"""Integration coverage for the kanban-worker rate-limit exit WIRING.

The pure decision helper ``_kanban_worker_rate_limit_exit_code`` is unit-tested
in ``test_kanban_worker_rate_limit_exit.py``. This module covers the wiring that
was actually broken — and is the whole point of the fix — which those unit tests
never exercise:

* ``HermesCLI.chat()`` stashing the classified ``failure_reason`` onto
  ``self._last_failure_reason`` (cli.py) — the non-quiet single-query caller
  doesn't see the result dict, so it reads the run outcome off this attribute.
* the deferred ``sys.exit(75)`` on the non-quiet ``-q`` branch of ``cli.main()``,
  applied AFTER the ``finally`` so ``_finalize_single_query`` (which releases the
  active-session lease) runs exactly once.
* the breaker safety contract: a real defect (non-transient ``failure_reason``)
  must NOT borrow the carve-out — the ``-q`` worker exits 0 (clean → dispatcher
  auto-blocks), never 75, so a genuine failure still trips the breaker and there
  is no infinite respawn.

Kanban workers are spawned with ``hermes chat -q "work kanban task <id>"`` which
parses to ``quiet=False`` (NON-quiet) — so the relevant path is the non-quiet
single-query branch, exercised here through the real ``cli.main()``.
"""

from __future__ import annotations

from types import SimpleNamespace

import cli as cli_mod
from cli import HermesCLI
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


# --------------------------------------------------------------------------- #
# A. Real chat() stashes the classified failure reason for the -q caller.
# --------------------------------------------------------------------------- #


def _make_chat_cli(result):
    """A near-real HermesCLI whose only stub is the agent's run_conversation.

    Everything the stash code path touches runs for real; only the heavyweight
    agent init / credential refresh / display plumbing is stubbed on the
    instance so we can drive the actual ``chat()`` body.
    """
    c = HermesCLI.__new__(HermesCLI)
    c.session_id = "sess-1"
    c.model = "m"
    c.provider = "p"
    c.base_url = ""
    c.api_key = ""
    c.api_mode = "chat_completions"
    c.conversation_history = []
    c.final_response_markdown = "strip"
    c._session_db = None
    c._voice_tts = False
    c._voice_mode = False
    c._voice_continuous = False
    c._voice_tts_done = None
    c._stream_started = False
    c._stream_box_opened = False
    c.show_reasoning = False
    c.bell_on_complete = False
    c.show_timestamps = False
    c._reasoning_shown_this_turn = False
    c._active_agent_route_signature = "sig"
    c._secret_capture_callback = None
    c._sudo_password_callback = None
    c._approval_callback = None
    c._pending_model_switch_note = None
    c._pending_skills_reload_note = None
    c._pending_moa_config = None
    c._pending_moa_disable_after_turn = False
    c._prompt_start_time = None
    c._ensure_runtime_credentials = lambda: True
    c._resolve_turn_agent_config = lambda msg: {
        "signature": "sig",
        "model": None,
        "runtime": None,
        "request_overrides": None,
    }
    c._init_agent = lambda **kw: True
    c._flush_stream = lambda: None
    c._flush_credit_notices = lambda: None
    c._scrollback_box_width = lambda *a, **k: 80
    c._invalidate = lambda *a, **k: None
    c.agent = SimpleNamespace(
        session_id="sess-1",
        platform="cli",
        max_iterations=90,
        run_conversation=lambda **kw: result,
    )
    return c


def test_chat_stashes_transient_failure_reason():
    """A failed run carrying a transient throttle reason is stashed verbatim
    so the non-quiet ``-q`` caller can map it to the EX_TEMPFAIL sentinel."""
    for reason in ["rate_limit", "billing"]:
        c = _make_chat_cli(
            {
                "final_response": "",
                "failed": True,
                "failure_reason": reason,
                "messages": [],
            }
        )
        c.chat("work kanban task t_x")
        assert c._last_failure_reason == reason


def test_chat_stashes_none_on_clean_turn():
    """A clean (non-failed) turn stashes ``None`` even if the result dict still
    carries a stray ``failure_reason`` — the stash is gated on ``failed`` so a
    healthy run can never be misread as a rate-limit and exit 75."""
    c = _make_chat_cli(
        {
            "final_response": "all good",
            "failed": False,
            "failure_reason": "rate_limit",  # stray key; must be ignored when not failed
            "messages": [],
        }
    )
    c.chat("work kanban task t_x")
    assert c._last_failure_reason is None


def test_chat_stashes_none_for_non_transient_failure_via_helper():
    """A real-defect failure is stashed as-is, but the shared helper refuses to
    map it to the sentinel — so the breaker can still trip on a genuine crash."""
    c = _make_chat_cli(
        {
            "final_response": "",
            "failed": True,
            "failure_reason": "context_overflow",
            "messages": [],
        }
    )
    c.chat("work kanban task t_x")
    assert c._last_failure_reason == "context_overflow"
    assert (
        cli_mod._kanban_worker_rate_limit_exit_code(
            c._last_failure_reason, is_kanban_worker=True
        )
        is None
    )


# --------------------------------------------------------------------------- #
# B/C. main()'s non-quiet -q branch: deferred exit, finalize-once, ordering.
# --------------------------------------------------------------------------- #


def _drive_non_quiet_q(monkeypatch, *, failure_reason, kanban_task):
    """Run cli.main() down the non-quiet ``-q`` single-query branch with a fake
    CLI whose chat() reports ``failure_reason``. Returns the ordered call log and
    the finalize-call count. Raising SystemExit is propagated to the caller.
    """
    calls: list = []
    finalize_count = {"n": 0}

    class _Console:
        def print(self, *_a, **_k):
            calls.append("query-label")

    class FakeCLI:
        def __init__(self, **_kw):
            self.console = _Console()
            self.session_id = "worker-session"
            self.agent = SimpleNamespace(session_id="worker-session", platform="cli")

        def _claim_active_session(self, surface, *, stderr=False):
            calls.append(("claim", surface, stderr))
            return True

        def _show_security_advisories(self):
            calls.append("advisories")

        def chat(self, query, images=None):
            calls.append(("chat", query))
            # Mirror the real chat() stash: a failed run records its reason.
            self._last_failure_reason = failure_reason
            return "done"

        def _print_exit_summary(self):
            calls.append("summary")

    if kanban_task:
        monkeypatch.setenv("HERMES_KANBAN_TASK", kanban_task)
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)

    # The kanban image-ref enrichment block is best-effort; force it to bail
    # fast and deterministically (it's wrapped in try/except in main()).
    def _boom(*_a, **_k):
        raise RuntimeError("no kanban db in test")

    monkeypatch.setattr("hermes_cli.kanban_db.connect", _boom)

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_k: None)

    def _fake_finalize(_cli):
        finalize_count["n"] += 1
        calls.append("finalize")

    monkeypatch.setattr(cli_mod, "_finalize_single_query", _fake_finalize)

    return calls, finalize_count


def _assert_system_exit_code(expected_code, func, /, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except SystemExit as exc:
        assert exc.code == expected_code
        return
    raise AssertionError(f"expected SystemExit({expected_code})")


def test_worker_q_rate_limit_exits_tempfail_and_finalizes_once(monkeypatch):
    """A kanban-worker ``-q`` run that died on a transient throttle exits with
    the EX_TEMPFAIL sentinel (75) AND finalizes exactly once."""
    for reason in ["rate_limit", "billing"]:
        calls, finalize_count = _drive_non_quiet_q(
            monkeypatch, failure_reason=reason, kanban_task="t_worker"
        )

        _assert_system_exit_code(
            KANBAN_RATE_LIMIT_EXIT_CODE,
            cli_mod.main,
            query="work kanban task t_worker",
            quiet=False,
            toolsets="terminal",
        )

        assert KANBAN_RATE_LIMIT_EXIT_CODE == 75
        assert finalize_count["n"] == 1  # finalize-once: lease released exactly once


def test_worker_q_non_transient_exits_clean_not_tempfail(monkeypatch):
    """A genuine defect (non-transient ``failure_reason``) on the worker ``-q``
    path exits 0 (clean → dispatcher auto-blocks), NOT 75 — so the breaker still
    trips on real failures and there is no infinite respawn."""
    calls, finalize_count = _drive_non_quiet_q(
        monkeypatch, failure_reason="context_overflow", kanban_task="t_worker"
    )

    # No SystemExit: the non-quiet branch returns normally (exit code 0).
    cli_mod.main(query="work kanban task t_worker", quiet=False, toolsets="terminal")

    assert finalize_count["n"] == 1


def test_non_worker_q_rate_limit_exits_clean(monkeypatch):
    """The carve-out is scoped to kanban workers: a plain ``hermes chat -q`` with
    no HERMES_KANBAN_TASK keeps the normal clean exit even on a rate-limit
    failure (it must not borrow the EX_TEMPFAIL sentinel)."""
    calls, finalize_count = _drive_non_quiet_q(
        monkeypatch, failure_reason="rate_limit", kanban_task=None
    )

    cli_mod.main(query="hello", quiet=False, toolsets="terminal")

    assert finalize_count["n"] == 1


def test_worker_q_deferred_exit_fires_after_finalize(monkeypatch):
    """Ordering contract: the rate-limit carve-out reads ``_last_failure_reason``
    AFTER ``_print_exit_summary()``, defers the exit past the ``finally`` so
    ``_finalize_single_query`` runs first, then exits 75. Proves the deferred
    exit still fires despite the assignment sitting after the summary."""
    calls, finalize_count = _drive_non_quiet_q(
        monkeypatch, failure_reason="rate_limit", kanban_task="t_worker"
    )

    _assert_system_exit_code(
        75,
        cli_mod.main,
        query="work kanban task t_worker",
        quiet=False,
        toolsets="terminal",
    )
    # summary → finalize ordering, and both ran before the deferred exit raised.
    assert "summary" in calls and "finalize" in calls
    assert calls.index("summary") < calls.index("finalize")
    assert calls.index(("chat", "work kanban task t_worker")) < calls.index("summary")
