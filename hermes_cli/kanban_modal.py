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
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_MODAL_LANE = "memo-evaluator"
_MAX_MODAL_BRIEF_CHARS = 64_000
_log = logging.getLogger(__name__)

# The currently-running ``modal run`` child, tracked so a reclaim SIGTERM to the
# shim can forward termination to it. Without this the dispatcher would kill only
# the shim PID, orphaning the Modal CLI (and its paid remote call) to be
# duplicated when the task is requeued. Set while ``_run_modal`` blocks on the
# child; cleared in its ``finally``.
_active_modal_proc: subprocess.Popen | None = None

# Metadata keys the local Kanban lifecycle treats as trusted, host-side
# directives (they make ``complete_task`` copy/attach/unlink host files at the
# named paths). A remote Modal result is untrusted input — a prompt-injected or
# malformed worker response must never be able to smuggle these in and turn an
# arbitrary readable host file into a Kanban attachment (or get it unlinked).
# Strip them before applying the remote result.
_RESERVED_METADATA_KEYS = frozenset({"_staged_artifacts", "artifacts"})


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
        # The remote worker is untrusted: drop reserved host-side directive keys
        # so a malicious/hallucinated result cannot attach or unlink host files.
        stripped = _RESERVED_METADATA_KEYS.intersection(metadata)
        if stripped:
            _log.warning(
                "modal result for %s carried reserved metadata keys %s; stripped",
                task_id,
                sorted(stripped),
            )
            for key in stripped:
                metadata.pop(key, None)
        metadata.update(audit)
        # Complete first, then record the audit comment only if the run-validated
        # completion actually landed. A stale shim (task reclaimed while its Modal
        # call was in flight) has ``complete_task`` return False on the run-id
        # mismatch; writing the comment unconditionally would smear the old call
        # id / log url onto the new attempt's audit history.
        #
        # ``scan_prose_artifacts=False``: the remote worker is untrusted and
        # mount-less, so a prompt-injected summary must not be able to name a
        # workspace file (e.g. ``<workspace>/.env``) and have the host-side prose
        # scanner promote it into Kanban attachments. Stripping the reserved
        # metadata keys above does not cover the prose route.
        completed = kb.complete_task(
            conn,
            task_id,
            summary=summary.strip(),
            metadata=metadata,
            expected_run_id=expected_run_id,
            scan_prose_artifacts=False,
        )
        if completed:
            kb.add_comment(
                conn,
                task_id,
                "modal-shim",
                f"Modal audit: call {audit['modal_call_id']} — {audit['modal_log_url']}",
            )
        return completed

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
    """Synchronously invoke the Modal app and parse its structured response.

    The ``modal run`` child is launched in its own process group and tracked in
    ``_active_modal_proc`` so a reclaim/timeout SIGTERM delivered to this shim can
    be forwarded to the whole group (see ``_terminate_active_modal``). Killing
    only the shim would orphan the Modal CLI and its paid remote call, which a
    requeued attempt would then duplicate.
    """
    global _active_modal_proc

    modal_bin = shutil.which("modal")
    if modal_bin is None:
        raise RuntimeError(
            "Modal CLI is not installed or not on PATH; install it on the "
            "dispatcher host with `uv pip install modal` and run `modal setup` "
            "before enabling the memo-evaluator Modal backend"
        )
    fd, result_name = tempfile.mkstemp(prefix="hermes-kanban-modal-", suffix=".json")
    os.close(fd)
    result_path = Path(result_name)
    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(  # noqa: S603 -- fixed CLI, request piped via stdin
            [
                modal_bin,
                "run",
                "--write-result",
                str(result_path),
                _modal_runner_path(),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Own process group so the shim can forward SIGTERM/SIGKILL to the
            # Modal CLI (and any grandchild it spawns), not just its direct pid.
            start_new_session=True,
        )
        _active_modal_proc = proc
        try:
            # Pass the (potentially ~64KB) request over stdin, not as an argv
            # element: Windows caps the ENTIRE command line at 32,767 chars, so a
            # large brief in ``--request-json`` would overflow the spawn there.
            _stdout, _stderr = proc.communicate(
                input=json.dumps(request, separators=(",", ":")),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # The task's runtime cap elapsed. Tear the whole Modal group down so
            # the remote call can't keep billing after we've given up on it.
            _terminate_active_modal(proc)
            raise
        if proc.returncode != 0:
            raise RuntimeError(f"Modal run exited with status {proc.returncode}")
        try:
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Modal run did not return a structured result") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Modal run returned a non-object result")
        return parsed
    finally:
        if proc is not None and _active_modal_proc is proc:
            _active_modal_proc = None
        result_path.unlink(missing_ok=True)


def _terminate_active_modal(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the Modal CLI's whole process group.

    Called when the shim is being torn down (its own SIGTERM handler) or when
    the per-task timeout elapses. Signalling the group — not just ``proc.pid`` —
    stops any grandchild the Modal CLI spawned so no orphaned remote call keeps
    running after the shim exits.
    """
    if proc.poll() is not None:
        return
    _signal_process_group(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_group(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _log.warning("Modal CLI child %s did not exit after SIGKILL", proc.pid)


def _signal_process_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's process group, falling back to the bare pid.

    ``start_new_session=True`` makes the child a group leader, so ``killpg`` on
    its pid reaches the Modal CLI and any grandchild. On platforms without
    ``killpg`` (Windows), fall back to signalling the process directly.
    """
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    try:
        if killpg is not None and getpgid is not None:
            killpg(getpgid(proc.pid), sig)
        else:
            proc.send_signal(sig)
    except (ProcessLookupError, OSError):
        pass


def _spawned_run_id(task_id: str) -> int | None:
    """Return the run id pinned into the env when this shim was spawned.

    ``spawn_modal_worker`` records the claimed run in ``HERMES_KANBAN_RUN_ID``.
    A delayed/previous shim that survives a reclaim must complete only the run
    it was spawned for — not whatever run happens to be current when it finally
    starts — so a stale Modal response cannot land on a re-claimed new attempt.
    Mirrors the local worker's env-pinned guard (``_worker_run_id_for``).
    """
    if os.environ.get("HERMES_KANBAN_TASK") not in (None, task_id):
        return None
    raw = os.environ.get("HERMES_KANBAN_RUN_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def run_modal_shim(task_id: str, workspace: str) -> bool:
    """Run Modal then map its response to a local Kanban completion or block."""
    from hermes_cli import kanban_db as kb

    # Prefer the run id pinned at spawn; only fall back to the request's fresh
    # read when the env var is absent (e.g. a direct call in a test).
    spawned_run_id = _spawned_run_id(task_id)
    expected_run_id: int | None = spawned_run_id
    try:
        request, request_run_id = _build_modal_request(task_id, workspace)
        if expected_run_id is None:
            expected_run_id = request_run_id
        # Pre-invocation staleness guard: if this shim was spawned for a specific
        # run (env-pinned) but the task has since been reclaimed into a newer run,
        # skip the paid Modal call entirely. Without this the stale shim still
        # launches — and pays for — a duplicate remote evaluation against the new
        # task state; only the *result* was being rejected afterward. Leave the
        # current attempt untouched (no lifecycle write) so its own shim owns it.
        if (
            spawned_run_id is not None
            and request_run_id is not None
            and request_run_id != spawned_run_id
        ):
            _log.warning(
                "modal shim for %s was spawned for run %s but the task is now on "
                "run %s; skipping the remote call to avoid a duplicate paid "
                "evaluation",
                task_id,
                spawned_run_id,
                request_run_id,
            )
            return False
        timeout = None
        with kb.connect_closing() as conn:
            task = kb.get_task(conn, task_id)
            if task is not None and task.max_runtime_seconds:
                timeout = int(task.max_runtime_seconds)
        # Thread the per-task runtime into the remote function timeout too, so a
        # >1h or uncapped task isn't silently killed at the function default.
        request["max_runtime_seconds"] = timeout
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
            # Block first; only record the shim note if the run-validated
            # transition landed, so a stale shim (task reclaimed mid-call) can't
            # smear this attempt's audit onto the new one.
            blocked = kb.block_task(
                conn,
                task_id,
                reason=str(exc),
                kind="capability",
                expected_run_id=expected_run_id,
            )
            if blocked:
                kb.add_comment(conn, task_id, "modal-shim", str(exc))
            return blocked
    except Exception as exc:
        _log.error("modal Kanban shim failed for %s: %s", task_id, exc)
        with kb.connect_closing() as conn:
            if expected_run_id is None:
                task = kb.get_task(conn, task_id)
                expected_run_id = task.current_run_id if task else None
            # Same run-validated gating as the capability path above.
            blocked = kb.block_task(
                conn,
                task_id,
                reason="Modal worker invocation failed; see the worker log.",
                kind="transient",
                expected_run_id=expected_run_id,
            )
            if blocked:
                kb.add_comment(
                    conn,
                    task_id,
                    "modal-shim",
                    "Modal worker invocation failed; see the worker log for details.",
                )
            return blocked


def _install_shim_signal_forwarding() -> None:
    """Forward a reclaim/timeout SIGTERM to the in-flight Modal CLI child.

    The dispatcher (``enforce_max_runtime`` / ``reclaim_task``) signals only the
    recorded shim pid. Without this handler the shim would exit while its
    ``modal run`` child — and the paid remote call — kept running, so a requeued
    attempt could launch a duplicate evaluation. On SIGTERM/SIGINT we tear the
    Modal process group down first, then exit non-zero so the run is recorded as
    a failure/timeout rather than a phantom success.
    """
    def _handler(signum, _frame):
        proc = _active_modal_proc
        if proc is not None:
            _terminate_active_modal(proc)
        # 128 + signal number is the conventional "terminated by signal" code.
        raise SystemExit(128 + signum)

    for _sig_name in ("SIGTERM", "SIGINT"):
        _sig = getattr(signal, _sig_name, None)
        if _sig is not None:
            try:
                signal.signal(_sig, _handler)
            except (ValueError, OSError):
                # Not on the main thread (e.g. a direct in-process test call);
                # signal forwarding is a best-effort dispatcher-path safeguard.
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Kanban memo evaluator via Modal")
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)
    _install_shim_signal_forwarding()
    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "")
    return 0 if run_modal_shim(args.task_id, workspace) else 1


if __name__ == "__main__":
    raise SystemExit(main())
