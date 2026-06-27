"""Regression: kanban terminal-state notifications must not clip the worker's
handoff mid-sentence.

Before this fix, the notifier watcher sliced the ``blocked`` reason at 160 chars
and the ``gave_up`` error at 200 chars, so a worker's multi-paragraph
explanation of WHY it stopped arrived truncated on Signal/Telegram and the
reader had to query sqlite to read the rest. These tests drive the real
``_kanban_notifier_watcher`` against a temp DB and assert the full handoff
survives, while a pathological multi-KB reason still gets a bounded, visibly
truncated message.
"""

import asyncio

from gateway.config import Platform
from gateway.kanban_watchers import _NOTIFY_DETAIL_MAX
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _blocked_subscription(reason):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="block notify", assignee="dev")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (tid,))
        kb.block_task(conn, tid, reason=reason)
        return tid
    finally:
        conn.close()


def _gave_up_subscription(error):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="gaveup notify", assignee="dev")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, kind="gave_up", payload={"error": error})
        return tid
    finally:
        conn.close()


def test_blocked_reason_is_not_clipped_at_160(tmp_path, monkeypatch):
    db_path = tmp_path / "blocked-untruncate.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # A realistic worker handoff well past the old 160-char slice.
    reason = (
        "B1 XSS fix is staged + verified + byte-equal, ready for Eric to deploy. "
        "BLOCKING on the re-scoped work (comments #48/#49): the relayed "
        "'deploy green-lit' contradicts the card's own PROD GATE, and building "
        "a race-inference feature is not the security hotfix this card was for. "
        "Needs Eric to confirm directly and split the feature into its own card."
    )
    assert len(reason) > 160

    tid = _blocked_subscription(reason)
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "blocked" in text
    # The whole reason survives — including the tail that the 160-char slice
    # used to drop.
    assert reason in text
    assert "split the feature into its own card." in text


def test_gave_up_error_is_not_clipped_at_200(tmp_path, monkeypatch):
    db_path = tmp_path / "gaveup-untruncate.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    error = (
        "task t_48c55d21 worktree path '/Users/x/projects/cpe-research' is not "
        "inside a git repo and does not point at a git repo root. The dispatcher "
        "walked up looking for a repo root, found none, and could not materialize "
        "a linked worktree, so the worker never spawned across all retry attempts."
    )
    assert len(error) > 200

    tid = _gave_up_subscription(error)
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "gave up" in text
    assert error in text


def test_pathological_reason_is_bounded_with_visible_suffix(tmp_path, monkeypatch):
    db_path = tmp_path / "blocked-bounded.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    reason = "x" * (_NOTIFY_DETAIL_MAX + 500)
    tid = _blocked_subscription(reason)
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    # Bounded: the raw reason did not land whole.
    assert reason not in text
    # And the truncation is advertised, not silent.
    assert "more chars; see board" in text
