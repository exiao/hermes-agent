from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


def test_memo_evaluator_is_the_only_lane_that_can_use_modal():
    from hermes_cli import kanban_modal

    assert kanban_modal.resolve_worker_backend("memo-evaluator", {}) == "local"
    assert kanban_modal.resolve_worker_backend(
        "memo-evaluator", {"worker_backends": {"memo-evaluator": "modal"}}
    ) == "modal"
    assert kanban_modal.resolve_worker_backend(
        "dev", {"worker_backends": {"dev": "modal"}}
    ) == "local"


def test_modal_completion_is_written_locally_with_audit_metadata():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="grade memo", assignee="memo-evaluator")
        task = kb.claim_task(conn, task_id)
        assert task is not None

        assert kanban_modal.apply_modal_result(
            conn,
            task_id,
            {
                "outcome": "complete",
                "summary": "Memo passed the rubric.",
                "metadata": {"score": 8},
                "modal_call_id": "fc-123",
                "modal_log_url": "https://modal.com/apps/example/logs/fc-123",
            },
            expected_run_id=task.current_run_id,
        )

        completed = kb.get_task(conn, task_id)
        assert completed is not None and completed.status == "done"
        run = kb.list_runs(conn, task_id)[-1]
        assert run.metadata == {
            "score": 8,
            "modal_call_id": "fc-123",
            "modal_log_url": "https://modal.com/apps/example/logs/fc-123",
        }
        assert "fc-123" in kb.list_comments(conn, task_id)[-1].body


def test_modal_completion_strips_reserved_metadata_keys():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="grade memo", assignee="memo-evaluator")
        task = kb.claim_task(conn, task_id)
        assert task is not None

        # A prompt-injected / malformed remote result tries to smuggle host-side
        # attachment directives; the shim must drop them before completing so an
        # arbitrary host file can't be attached (or later unlinked).
        assert kanban_modal.apply_modal_result(
            conn,
            task_id,
            {
                "outcome": "complete",
                "summary": "Memo passed the rubric.",
                "metadata": {
                    "score": 8,
                    "_staged_artifacts": ["/etc/passwd"],
                    "artifacts": ["/home/user/.ssh/id_rsa"],
                },
                "modal_call_id": "fc-123",
                "modal_log_url": "https://modal.com/apps/example/logs/fc-123",
            },
            expected_run_id=task.current_run_id,
        )

        completed = kb.get_task(conn, task_id)
        assert completed is not None and completed.status == "done"
        run = kb.list_runs(conn, task_id)[-1]
        assert run.metadata is not None
        assert "_staged_artifacts" not in run.metadata
        assert "artifacts" not in run.metadata
        assert run.metadata["score"] == 8
        # No attachment was created from the injected host paths.
        assert kb.list_attachments(conn, task_id) == []


def test_modal_shim_pins_the_spawned_run_id(monkeypatch):
    """A stale shim must complete only the run it was spawned for.

    The shim reads ``HERMES_KANBAN_RUN_ID`` (pinned at spawn); if the task was
    reclaimed into a new run after this shim was launched, applying its result
    with the OLD run id must not touch the new attempt.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="grade memo", assignee="memo-evaluator")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        stale_run_id = task.current_run_id
        # Simulate a reclaim → re-claim: the task now runs under a NEW run.
        assert kb.reclaim_task(conn, task_id) is True
        reclaimed = kb.claim_task(conn, task_id)
        assert reclaimed is not None
        assert reclaimed.current_run_id != stale_run_id

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run_id))

    def _fake_run(_request, *, timeout=None):
        return {
            "outcome": "complete",
            "summary": "stale result",
            "modal_call_id": "fc-stale",
            "modal_log_url": "https://modal.test/log",
        }

    monkeypatch.setattr(kanban_modal, "_run_modal", _fake_run)

    # The stale run id no longer matches current_run_id, so the completion is
    # rejected (apply_modal_result returns False) and the task stays running.
    assert kanban_modal.run_modal_shim(task_id, "/tmp/workspace") is False
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"


def test_configured_spawn_routes_only_memo_evaluator_to_modal(monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    calls = []
    monkeypatch.setattr(
        kanban_modal,
        "spawn_modal_worker",
        lambda task, workspace, *, board=None: calls.append((task.assignee, workspace, board)) or 71,
    )
    monkeypatch.setattr(kb, "_load_kanban_cfg", lambda: {"worker_backends": {"memo-evaluator": "modal"}})
    monkeypatch.setattr(kb, "_default_spawn", lambda *_args, **_kwargs: 72)

    memo_task = SimpleNamespace(assignee="memo-evaluator")
    dev_task = SimpleNamespace(assignee="dev")
    assert kb._configured_worker_spawn(memo_task, "/tmp/memo", board="test-board") == 71
    assert kb._configured_worker_spawn(dev_task, "/tmp/dev", board="test-board") == 72
    assert calls == [("memo-evaluator", "/tmp/memo", "test-board")]


def test_modal_request_serializes_worker_brief_and_comments_without_board_env():
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="grade memo",
            body="Evaluate the attached investment memo.",
            assignee="memo-evaluator",
        )
        kb.add_comment(conn, task_id, "reviewer", "Use the current evidence rubric.")
        assert kb.claim_task(conn, task_id) is not None

    request, _run_id = kanban_modal._build_modal_request(task_id, "/tmp/workspace")

    assert request["workspace"] == "/tmp/workspace"
    assert "Evaluate the attached investment memo." in request["brief"]
    assert "Use the current evidence rubric." in request["brief"]
    assert "HERMES_KANBAN_DB" not in request


def test_modal_request_refuses_a_task_with_unreadable_attachments(tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="grade memo",
            body="Evaluate the attached investment memo.",
            assignee="memo-evaluator",
        )
        assert kb.claim_task(conn, task_id) is not None
        blob = tmp_path / "memo.pdf"
        blob.write_bytes(b"%PDF-1.4 fake")
        kb.add_attachment(
            conn,
            task_id,
            filename="memo.pdf",
            stored_path=str(blob),
            content_type="application/pdf",
            size=blob.stat().st_size,
        )

    # The mount-less Modal worker cannot read attachment bytes, so the shim must
    # refuse rather than ship a path-only brief that invites a hallucinated verdict.
    with pytest.raises(kanban_modal.ModalUnsupportedTask, match="memo.pdf"):
        kanban_modal._build_modal_request(task_id, "/tmp/workspace")


def test_modal_shim_blocks_an_attachment_task_as_a_capability_gap(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="grade memo",
            body="Evaluate the attached investment memo.",
            assignee="memo-evaluator",
        )
        assert kb.claim_task(conn, task_id) is not None
        blob = tmp_path / "memo.pdf"
        blob.write_bytes(b"%PDF-1.4 fake")
        kb.add_attachment(
            conn,
            task_id,
            filename="memo.pdf",
            stored_path=str(blob),
            content_type="application/pdf",
            size=blob.stat().st_size,
        )

    def _fail_run(*_args, **_kwargs):
        raise AssertionError("Modal must not be invoked for an unreadable-attachment task")

    monkeypatch.setattr(kanban_modal, "_run_modal", _fail_run)

    assert kanban_modal.run_modal_shim(task_id, "/tmp/workspace") is True

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "blocked"


def test_modal_request_rejects_an_oversized_worker_brief(monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="grade memo", assignee="memo-evaluator")
        assert kb.claim_task(conn, task_id) is not None

    monkeypatch.setattr(
        kb,
        "build_worker_context",
        lambda *_args: "x" * (kanban_modal._MAX_MODAL_BRIEF_CHARS + 1),
    )
    with pytest.raises(ValueError, match="too large"):
        kanban_modal._build_modal_request(task_id, "/tmp/workspace")


def test_modal_cli_result_is_consumed_from_the_write_result_file(monkeypatch, tmp_path):
    from hermes_cli import kanban_modal

    fake_modal = tmp_path / "modal"
    fake_modal.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        # The request must arrive on stdin (not argv) so a large brief stays
        # under the Windows command-line limit; fail loudly on regression.
        "assert '--request-json' not in sys.argv, 'request must be piped via stdin'\n"
        "req = json.loads(sys.stdin.read())\n"
        "assert req['brief'] == 'grade this'\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--write-result') + 1])\n"
        "out.write_text(json.dumps({'outcome': 'complete', 'summary': 'done', "
        "'modal_call_id': 'fc-test', 'modal_log_url': 'https://modal.test/log'}))\n",
        encoding="utf-8",
    )
    fake_modal.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    result = kanban_modal._run_modal({"task_id": "t_test", "brief": "grade this"})

    assert result["modal_call_id"] == "fc-test"
    assert result["modal_log_url"] == "https://modal.test/log"


def _load_worker_module(monkeypatch, tmp_path):
    """Import the Modal worker module with its import-time guards satisfied.

    The module imports ``modal`` and validates that the ``memo-evaluator``
    profile (SOUL.md + skills/) exists at import time, resolving its path via
    ``get_profile_dir`` under the per-test HERMES_HOME. Create that profile so
    the import-time guard passes. Skips cleanly when ``modal`` is not installed.
    """
    modal = pytest.importorskip("modal")  # noqa: F841 -- import guard only
    from hermes_cli.profiles import get_profile_dir

    profile = get_profile_dir("memo-evaluator")
    (profile / "skills").mkdir(parents=True, exist_ok=True)
    (profile / "SOUL.md").write_text("test soul", encoding="utf-8")
    import importlib

    return importlib.import_module("hermes_cli.kanban_modal_worker")


def test_worker_parses_verdict_even_with_a_stray_startup_line(monkeypatch, tmp_path):
    worker = _load_worker_module(monkeypatch, tmp_path)

    # Quiet mode can still emit a warning line before the final JSON response.
    stdout = (
        "  \u26a0 tirith security scanner enabled but not available\n"
        '{"outcome":"complete","summary":"Memo passed the rubric.","metadata":{"score":8}}\n'
    )
    result = worker._parse_worker_result(stdout)
    assert result["outcome"] == "complete"
    assert result["summary"] == "Memo passed the rubric."
    assert result["metadata"]["score"] == 8


def test_worker_extracts_the_last_object_when_prose_contains_braces(monkeypatch, tmp_path):
    worker = _load_worker_module(monkeypatch, tmp_path)

    # A JSON-looking snippet inside {braces} in prose must not shadow the real
    # final verdict object.
    stdout = (
        'Example shape: {"outcome":"block"} is one option.\n'
        '{"outcome":"complete","summary":"real verdict"}'
    )
    result = worker._parse_worker_result(stdout)
    assert result["summary"] == "real verdict"


def test_worker_binds_the_memo_evaluator_model_and_anthropic_proxy_secret(monkeypatch, tmp_path):
    worker = _load_worker_module(monkeypatch, tmp_path)

    # The memo-evaluator runs a Claude model through the Anthropic-format billing
    # proxy; the worker must pin that model and require the Anthropic proxy secret
    # keys, not an OpenAI-format secret that does not exist in the workspace.
    assert worker.MEMO_EVALUATOR_MODEL == "claude-fable-5"
    assert worker.MEMO_EVALUATOR_PROVIDER == "anthropic"
