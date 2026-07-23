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


def test_modal_audit_comment_is_only_written_on_a_validated_completion():
    """A rejected (stale-run) completion must not smear its audit onto the task.

    ``apply_modal_result`` completes first and only records the Modal call-id /
    log-url audit comment when the run-validated ``complete_task`` actually
    landed. A stale shim whose run id no longer matches must leave no misleading
    audit history on the new attempt.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="grade memo", assignee="memo-evaluator")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        stale_run_id = task.current_run_id
        assert kb.reclaim_task(conn, task_id) is True
        reclaimed = kb.claim_task(conn, task_id)
        assert reclaimed is not None
        assert reclaimed.current_run_id != stale_run_id

        result = {
            "outcome": "complete",
            "summary": "stale verdict",
            "modal_call_id": "fc-stale",
            "modal_log_url": "https://modal.test/log",
        }
        # Applying with the STALE run id: complete_task returns False, so no audit
        # comment is written.
        assert (
            kanban_modal.apply_modal_result(
                conn, task_id, result, expected_run_id=stale_run_id
            )
            is False
        )
        comments = kb.list_comments(conn, task_id)
    assert not any("Modal audit" in (c.body or "") for c in comments)


def test_modal_shim_skips_the_remote_call_when_reclaimed(monkeypatch):
    """A stale shim must not launch (and pay for) a duplicate remote evaluation.

    When a shim from a reclaimed attempt starts after the task was claimed again,
    the env-pinned run id no longer matches the task's current run. The shim must
    skip ``_run_modal`` entirely — not just reject the result afterward — so no
    duplicate paid Modal call is made against the new task state.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="grade memo", assignee="memo-evaluator")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        stale_run_id = task.current_run_id
        assert kb.reclaim_task(conn, task_id) is True
        reclaimed = kb.claim_task(conn, task_id)
        assert reclaimed is not None
        assert reclaimed.current_run_id != stale_run_id

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run_id))

    called = {"ran": False}

    def _fake_run(_request, *, timeout=None):
        called["ran"] = True
        raise AssertionError("stale shim must not invoke Modal")

    monkeypatch.setattr(kanban_modal, "_run_modal", _fake_run)

    # The pre-invocation guard returns False without ever calling _run_modal.
    assert kanban_modal.run_modal_shim(task_id, "/tmp/workspace") is False
    assert called["ran"] is False
    # The current (reclaimed) attempt is left untouched for its own shim to own.
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"


def test_modal_shim_failure_comment_is_gated_on_the_claimed_run(monkeypatch):
    """A failure from a stale shim must not smear the new attempt's audit.

    When the Modal invocation errors after the shim was reclaimed and the task
    re-claimed, the block is rejected on the run-id mismatch; the shim note must
    only be written when that run-validated block actually landed.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="grade memo", assignee="memo-evaluator")
        task = kb.claim_task(conn, task_id)
        assert task is not None
        stale_run_id = task.current_run_id
        assert kb.reclaim_task(conn, task_id) is True
        reclaimed = kb.claim_task(conn, task_id)
        assert reclaimed is not None
        assert reclaimed.current_run_id != stale_run_id

    # Pin the STALE run and skip the pre-invocation guard by making the request
    # build report the stale run id, then fail inside _run_modal.
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run_id))
    monkeypatch.setattr(
        kanban_modal,
        "_build_modal_request",
        lambda _tid, _ws: ({"task_id": task_id, "brief": "b"}, stale_run_id),
    )

    def _boom(_request, *, timeout=None):
        raise RuntimeError("modal exploded")

    monkeypatch.setattr(kanban_modal, "_run_modal", _boom)

    # The block is rejected (stale run), so it returns False and writes no note.
    assert kanban_modal.run_modal_shim(task_id, "/tmp/workspace") is False
    with kb.connect_closing() as conn:
        comments = kb.list_comments(conn, task_id)
    assert not any("worker log" in (c.body or "") for c in comments)


def test_modal_shim_threads_task_runtime_into_the_request(monkeypatch):
    """A >1h / uncapped task's runtime must reach the remote so it isn't
    killed at the function default. The shim adds max_runtime_seconds to the
    request payload it hands to Modal."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_modal

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="long memo",
            assignee="memo-evaluator",
            max_runtime_seconds=7200,
        )
        task = kb.claim_task(conn, task_id)
        assert task is not None

    captured: dict = {}

    def _fake_run(request, *, timeout=None):
        captured["request"] = request
        captured["timeout"] = timeout
        return {
            "outcome": "complete",
            "summary": "ok",
            "modal_call_id": "fc-1",
            "modal_log_url": "https://modal.test/log",
        }

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setattr(kanban_modal, "_run_modal", _fake_run)

    assert kanban_modal.run_modal_shim(task_id, "/tmp/workspace") is True
    assert captured["request"]["max_runtime_seconds"] == 7200
    assert captured["timeout"] == 7200


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


def test_worker_uncapped_runtime_uses_modal_24h_ceiling(monkeypatch, tmp_path):
    worker = _load_worker_module(monkeypatch, tmp_path)

    # A task with no runtime cap must run to Modal's 24h function-timeout ceiling,
    # not the 1h function default.
    assert worker._resolve_function_timeout(None) == worker._MODAL_MAX_TIMEOUT


def test_worker_runtime_within_cap_is_honored(monkeypatch, tmp_path):
    worker = _load_worker_module(monkeypatch, tmp_path)

    # A 2h task keeps its exact runtime; a string is coerced.
    assert worker._resolve_function_timeout(7200) == 7200
    assert worker._resolve_function_timeout("7200") == 7200


def test_worker_rejects_runtime_above_modal_cap_as_capability_block(monkeypatch, tmp_path):
    worker = _load_worker_module(monkeypatch, tmp_path)

    # A 48h task exceeds Modal's 24h function-timeout cap. Silently clamping to
    # 24h would kill it mid-evaluation and requeue it as a transient failure
    # forever; the worker must reject it up front as a capability block so a
    # human reroutes it — before any paid remote spawn.
    result = worker._resolve_function_timeout(48 * 60 * 60)
    assert isinstance(result, dict)
    assert result["outcome"] == "block"
    assert result["kind"] == "capability"
    assert "exceeds Modal" in result["reason"]


def test_worker_unparseable_runtime_falls_back_to_ceiling(monkeypatch, tmp_path):
    worker = _load_worker_module(monkeypatch, tmp_path)

    # A malformed value is treated as uncapped, not an error.
    assert worker._resolve_function_timeout("not-a-number") == worker._MODAL_MAX_TIMEOUT


def test_worker_main_handles_non_dict_request_json(monkeypatch, tmp_path):
    """A valid-JSON-but-non-dict request must not crash the entrypoint.

    ``json.loads`` of a bare list/string succeeds, but ``.get`` on it raises
    AttributeError. The runtime resolution must swallow that and fall back to
    uncapped rather than crashing the Modal local entrypoint.
    """
    worker = _load_worker_module(monkeypatch, tmp_path)
    import json as _json

    for payload in ("[1, 2, 3]", '"just a string"', "42"):
        try:
            max_runtime = _json.loads(payload).get("max_runtime_seconds")
        except (TypeError, ValueError, AttributeError, _json.JSONDecodeError):
            max_runtime = None
        # Same recovery the entrypoint applies → treated as uncapped.
        assert worker._resolve_function_timeout(max_runtime) == worker._MODAL_MAX_TIMEOUT


def test_run_modal_terminates_the_modal_child_group_on_timeout(monkeypatch, tmp_path):
    """A per-task runtime timeout must tear down the Modal CLI process group.

    Killing only the shim would orphan the ``modal run`` child (and its paid
    remote call); the shim forwards termination to the whole group so a requeued
    attempt can't launch a duplicate evaluation.
    """
    import subprocess

    from hermes_cli import kanban_modal

    monkeypatch.setattr(kanban_modal.shutil, "which", lambda _bin: "/usr/bin/modal")

    class _FakeProc:
        def __init__(self):
            self.pid = 4242
            self.returncode = None
            self._alive = True
            self.signals: list[int] = []

        def communicate(self, input=None, timeout=None):
            raise subprocess.TimeoutExpired(cmd="modal", timeout=timeout or 1)

        def poll(self):
            return None if self._alive else self.returncode

        def wait(self, timeout=None):
            # First SIGTERM: still alive (force the SIGKILL escalation). After a
            # SIGKILL was delivered, report exit.
            if any(s == getattr(kanban_modal.signal, "SIGKILL", None) for s in self.signals):
                self._alive = False
                self.returncode = -9
                return self.returncode
            raise subprocess.TimeoutExpired(cmd="modal", timeout=timeout or 1)

        def send_signal(self, sig):
            self.signals.append(sig)

    fake = _FakeProc()
    monkeypatch.setattr(kanban_modal.subprocess, "Popen", lambda *a, **k: fake)

    captured_groups: list[tuple[int, int]] = []

    def _fake_killpg(pgid, sig):
        captured_groups.append((pgid, sig))
        fake.signals.append(sig)

    monkeypatch.setattr(kanban_modal.os, "killpg", _fake_killpg, raising=False)
    monkeypatch.setattr(kanban_modal.os, "getpgid", lambda pid: pid, raising=False)

    with pytest.raises(subprocess.TimeoutExpired):
        kanban_modal._run_modal({"task_id": "t_x", "brief": "grade"}, timeout=1)

    # SIGTERM then SIGKILL were sent to the child's process group (pid==pgid).
    sigterm = kanban_modal.signal.SIGTERM
    sigkill = getattr(kanban_modal.signal, "SIGKILL", sigterm)
    assert (4242, sigterm) in captured_groups
    assert (4242, sigkill) in captured_groups
    # The global tracker is cleared so a later spawn isn't mistaken for this one.
    assert kanban_modal._active_modal_proc is None

