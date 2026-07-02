"""Tests: kanban_block reason is capped to a one-line human summary.

The block ``reason`` is the board's human-readable summary; the schema and
KANBAN_GUIDANCE both tell workers to keep it short and push structured detail
into a kanban_comment. Workers ignore that and dump run ids / paths / API
routes / rule citations into the reason, making the board unreadable. The
handler caps the reason mechanically (truncate, never reject, so the block
always lands). The cap is configurable via ``kanban.block_reason_max_chars``.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with a running task; returns the task id."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def _stored_reason(task_id):
    from hermes_cli import kanban_db as kb
    conn = kb.connect()
    try:
        run = kb.latest_run(conn, task_id)
    finally:
        conn.close()
    assert run is not None
    return run.summary or ""


def test_short_reason_unchanged(worker_env):
    """A one-line reason under the cap passes through byte-for-byte."""
    from tools import kanban_tools as kt
    reason = "review-required: 9502.T memo staged; publish command in comment."
    kt._handle_block({"reason": reason})
    assert _stored_reason(worker_env) == reason


def test_long_reason_truncated_with_marker(worker_env):
    """An over-long reason is clipped to the cap and marked."""
    from tools import kanban_tools as kt
    reason = "review-required: " + ("x" * 2000)
    kt._handle_block({"reason": reason})
    stored = _stored_reason(worker_env)
    assert len(stored) <= kt._BLOCK_REASON_DEFAULT_MAX_CHARS
    assert stored.startswith("review-required: ")
    assert "kanban_comment" in stored
    assert stored != reason


def test_cap_configurable_via_config(worker_env, monkeypatch):
    """kanban.block_reason_max_chars overrides the default cap."""
    from tools import kanban_tools as kt

    def _fake_cfg_get(cfg, *keys, default=None):
        if keys == ("kanban", "block_reason_max_chars"):
            return 50
        return default

    monkeypatch.setattr(kt, "cfg_get", _fake_cfg_get)
    reason = "review-required: " + ("y" * 500)
    kt._handle_block({"reason": reason})
    stored = _stored_reason(worker_env)
    assert len(stored) <= 50


def test_cap_disabled_when_zero(worker_env, monkeypatch):
    """A cap of 0 disables clipping entirely (escape hatch)."""
    from tools import kanban_tools as kt

    def _fake_cfg_get(cfg, *keys, default=None):
        if keys == ("kanban", "block_reason_max_chars"):
            return 0
        return default

    monkeypatch.setattr(kt, "cfg_get", _fake_cfg_get)
    reason = "review-required: " + ("z" * 2000)
    kt._handle_block({"reason": reason})
    assert _stored_reason(worker_env) == reason
