"""Local Modal shim for the Kanban memo-evaluator lane.

The dispatcher runs the shim locally so the board database remains the sole
lifecycle authority. The shim may send a bounded task brief to Modal, but it
never exposes a Kanban DB path or lifecycle credentials to the remote worker.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_MODAL_LANE = "memo-evaluator"
_MAX_MODAL_BRIEF_CHARS = 64_000
_log = logging.getLogger(__name__)


class ModalUnsupportedTask(Exception):
    """A memo task cannot be evaluated remotely and must run on a mounted backend.

    The Modal worker has no Kanban attachments mount, so a task whose memo is
    supplied as an attachment (PDF / source document) would be evaluated with
    only its file *path* in the brief — the remote container cannot read the
    bytes and could return a hallucinated ``complete``. Surface this as a
    capability block so a human reroutes the task instead of trusting a verdict
    made against files the worker never saw.
    """


def resolve_worker_backend(assignee: str | None, kanban_config: dict[str, Any]) -> str:
    """Return the configured backend, restricting Modal to memo-evaluator.

    Every lane is local by default. Treating a non-memo lane's ``modal`` value
    as local prevents an accidental configuration edit from widening this
    phase-one integration before its remote workspace contract exists.
    """
    if (assignee or "").strip().lower() != _MODAL_LANE:
        return "local"
    raw_backends = kanban_config.get("worker_backends", {})
    configured = raw_backends.get(_MODAL_LANE, "local") if isinstance(raw_backends, dict) else "local"
    return "modal" if str(configured).strip().lower() == "modal" else "local"


def _audit_metadata(result: dict[str, Any]) -> dict[str, str]:
    call_id = result.get("modal_call_id")
    log_url = result.get("modal_log_url")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError("Modal completion is missing modal_call_id")
    if not isinstance(log_url, str) or not log_url.strip():
        raise ValueError("Modal completion is missing modal_log_url")
    return {
        "modal_call_id": call_id.strip(),
        "modal_log_url": log_url.strip(),
    }


def apply_modal_result(
    conn: Any,
    task_id: str,
    result: dict[str, Any],
    *,
    expected_run_id: int | None,
) -> bool:
    """Apply a remote result through the local Kanban lifecycle APIs only."""
    from hermes_cli import kanban_db as kb

    outcome = result.get("outcome")
    if outcome == "complete":
        audit = _audit_metadata(result)
        summary = result.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Modal completion is missing summary")
        remote_metadata = result.get("metadata")
        metadata = dict(remote_metadata) if isinstance(remote_metadata, dict) else {}
        metadata.update(audit)
        kb.add_comment(
            conn,
            task_id,
            "modal-shim",
            f"Modal audit: call {audit['modal_call_id']} — {audit['modal_log_url']}",
        )
        return kb.complete_task(
            conn,
            task_id,
            summary=summary.strip(),
            metadata=metadata,
            expected_run_id=expected_run_id,
        )

    if outcome == "block":
        reason = result.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Modal block is missing reason")
        kind = result.get("kind")
        if kind is not None and not isinstance(kind, str):
            raise ValueError("Modal block kind must be a string")
        return kb.block_task(
            conn,
            task_id,
            reason=reason.strip(),
            kind=kind,
            expected_run_id=expected_run_id,
        )

    raise ValueError("Modal result outcome must be 'complete' or 'block'")


def _modal_runner_path() -> str:
    runner = Path(__file__).with_name("kanban_modal_worker.py")
    if not runner.is_file():
        raise RuntimeError(f"Modal worker script is missing: {runner}")
    return str(runner)


def spawn_modal_worker(task: Any, workspace: str, *, board: str | None = None) -> int:
    """Launch a local shim that owns Modal invocation and local DB writes."""
    import subprocess
    from hermes_cli import kanban_db as kb

    env = dict(os.environ)
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    env["HERMES_KANBAN_DB"] = str(kb.kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(kb.workspaces_root(board=board))
    env["HERMES_KANBAN_BOARD"] = kb._normalize_board_slug(board) or kb.get_current_board()
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)

    log_dir = kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = kb.worker_log_rotation_config()
    kb._rotate_worker_log(log_path, rotate_bytes, backup_count)
    log_f = open(log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 -- fixed interpreter/module argv
            [sys.executable, "-m", "hermes_cli.kanban_modal", "--task-id", task.id],
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    finally:
        # The child inherits its own dup of the fd; the parent's handle is
        # only needed for the spawn and must be released so open descriptors
        # don't accumulate as more Modal workers are launched.
        log_f.close()
    return proc.pid


def _build_modal_request(task_id: str, workspace: str) -> tuple[dict[str, Any], int | None]:
    """Build the remote-safe payload without leaking local board credentials."""
    from hermes_cli import kanban_db as kb

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        if task is None or task.status != "running":
            raise ValueError(f"Kanban task {task_id} is not running")
        if (task.assignee or "").strip().lower() != _MODAL_LANE:
            raise ValueError(f"Kanban task {task_id} is not assigned to {_MODAL_LANE}")
        attachments = kb.list_attachments(conn, task.id)
        if attachments:
            names = ", ".join(sorted(a.filename for a in attachments))
            raise ModalUnsupportedTask(
                f"Kanban task {task_id} has attachments ({names}) that the "
                "mount-less Modal worker cannot read; run it on a backend with "
                "the attachments directory mounted."
            )
        brief = kb.build_worker_context(conn, task.id)
        if len(brief) > _MAX_MODAL_BRIEF_CHARS:
            raise ValueError(
                f"Kanban task brief is too large for the Modal invocation "
                f"({_MAX_MODAL_BRIEF_CHARS} character limit)"
            )
        return {
            "task_id": task.id,
            "brief": brief,
            "workspace": workspace,
        }, task.current_run_id


def _run_modal(request: dict[str, Any], *, timeout: int | None = None) -> dict[str, Any]:
    """Synchronously invoke the Modal app and parse its structured response."""
    modal_bin = shutil.which("modal")
    if modal_bin is None:
        raise RuntimeError("Modal CLI is not installed or not on PATH")
    fd, result_name = tempfile.mkstemp(prefix="hermes-kanban-modal-", suffix=".json")
    os.close(fd)
    result_path = Path(result_name)
    try:
        completed = subprocess.run(  # noqa: S603 -- fixed CLI plus serialized request
            [
                modal_bin,
                "run",
                "--write-result",
                str(result_path),
                _modal_runner_path(),
                "--request-json",
                json.dumps(request, separators=(",", ":")),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Modal run exited with status {completed.returncode}")
        try:
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Modal run did not return a structured result") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Modal run returned a non-object result")
        return parsed
    finally:
        result_path.unlink(missing_ok=True)


def run_modal_shim(task_id: str, workspace: str) -> bool:
    """Run Modal then map its response to a local Kanban completion or block."""
    from hermes_cli import kanban_db as kb

    expected_run_id: int | None = None
    try:
        request, expected_run_id = _build_modal_request(task_id, workspace)
        timeout = None
        with kb.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
            if task is not None and task.max_runtime_seconds:
                timeout = int(task.max_runtime_seconds)
        result = _run_modal(request, timeout=timeout)
        with kb.connect_closing() as conn:
            return apply_modal_result(
                conn, task_id, result, expected_run_id=expected_run_id
            )
    except ModalUnsupportedTask as exc:
        _log.warning("modal Kanban shim cannot run %s remotely: %s", task_id, exc)
        with kb.connect_closing() as conn:
            if expected_run_id is None:
                task = kb.get_task(conn, task_id)
                expected_run_id = task.current_run_id if task else None
            kb.add_comment(conn, task_id, "modal-shim", str(exc))
            return kb.block_task(
                conn,
                task_id,
                reason=str(exc),
                kind="capability",
                expected_run_id=expected_run_id,
            )
    except Exception as exc:
        _log.error("modal Kanban shim failed for %s: %s", task_id, exc)
        with kb.connect_closing() as conn:
            if expected_run_id is None:
                task = kb.get_task(conn, task_id)
                expected_run_id = task.current_run_id if task else None
            kb.add_comment(
                conn,
                task_id,
                "modal-shim",
                "Modal worker invocation failed; see the worker log for details.",
            )
            return kb.block_task(
                conn,
                task_id,
                reason="Modal worker invocation failed; see the worker log.",
                kind="transient",
                expected_run_id=expected_run_id,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Kanban memo evaluator via Modal")
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "")
    return 0 if run_modal_shim(args.task_id, workspace) else 1


if __name__ == "__main__":
    raise SystemExit(main())
