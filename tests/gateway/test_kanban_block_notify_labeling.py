"""Regression: blocked-task notifications are self-labeling by ``kind``.

A worker's typed block (kanban_db.VALID_BLOCK_KINDS) must produce a Signal/
Telegram push whose FIRST line says — at a glance — whether the reader must
act:

  needs_input → ``🔴 DECISION NEEDED — <id>: <title>``
  capability  → ``🟠 ROUTING — <id>: <title>``  (NOT a human-action item)
  transient   → ``🟡 RETRY — <id>: <title>``

An un-typed/legacy block keeps the historical ``⏸ … blocked: <reason>`` shape.
These drive the real ``_kanban_notifier_watcher`` against a temp DB so the whole
producer → notifier path is exercised (the header is normalized into the stored
reason by ``block_task`` and the typed ``kind`` rides in the event payload).
"""

import asyncio

from gateway.config import Platform
from gateway.kanban_watchers import _format_block_notification
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


def _typed_block_subscription(reason, kind, title="do the thing"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title=title, assignee="dev")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (tid,))
        kb.block_task(conn, tid, reason=reason, kind=kind)
        return tid
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pure formatter unit tests (no DB)
# ---------------------------------------------------------------------------


def test_format_needs_input_leads_with_decision_needed():
    msg = _format_block_notification(
        "needs_input", "t_abc", "merge CPE PRs #795-799", "which env first?"
    )
    assert msg.splitlines()[0] == "🔴 DECISION NEEDED — t_abc: merge CPE PRs #795-799"
    assert "which env first?" in msg


def test_format_capability_reads_as_routing_not_decision():
    msg = _format_block_notification(
        "capability", "t_def", "rotate prod token", "no access to the vault"
    )
    first = msg.splitlines()[0]
    assert first == "🟠 ROUTING — t_def: rotate prod token"
    # A routing/capability block must NOT read as a human-decision item.
    assert "DECISION NEEDED" not in msg


def test_format_transient_leads_with_retry():
    msg = _format_block_notification("transient", "t_g", "crawl feed", "429 from API")
    assert msg.splitlines()[0] == "🟡 RETRY — t_g: crawl feed"


def test_format_untyped_keeps_legacy_blocked_shape():
    msg = _format_block_notification(None, "t_legacy", "old task", "stuck on something")
    assert msg == "⏸ Kanban t_legacy blocked: stuck on something"


def test_format_includes_assignee_tag():
    msg = _format_block_notification(
        "needs_input", "t_a", "title", "reason", tag="@dev "
    )
    assert msg.splitlines()[0] == "🔴 DECISION NEEDED — @dev t_a: title"


# ---------------------------------------------------------------------------
# End-to-end through the real notifier watcher
# ---------------------------------------------------------------------------


def test_needs_input_block_pushes_decision_needed_header(tmp_path, monkeypatch):
    db_path = tmp_path / "needs-input.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _typed_block_subscription(
        "merge CPE migration PRs #795-799", "needs_input", title="ship CPE migration"
    )
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    first = text.splitlines()[0]
    assert first.startswith("🔴 DECISION NEEDED — ")
    assert tid in first
    assert "ship CPE migration" in first
    # The reason still rides in the body (block_task stamps the DECISION NEEDED:
    # header into the stored reason; that's fine — it's below the headline).
    assert "merge CPE migration PRs #795-799" in text


def test_capability_block_pushes_routing_header_not_decision(tmp_path, monkeypatch):
    db_path = tmp_path / "capability.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _typed_block_subscription(
        "no creds for the prod vault — wrong lane", "capability",
        title="rotate prod token",
    )
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    first = text.splitlines()[0]
    assert first.startswith("🟠 ROUTING — ")
    assert tid in first
    assert "rotate prod token" in first
    # A routing block is for the orchestrator, never a decision-needed headline.
    assert not first.startswith("🔴")
    assert "DECISION NEEDED" not in first


def test_untyped_block_keeps_legacy_line(tmp_path, monkeypatch):
    """An un-typed (legacy/dispatcher) block keeps the ⏸ … blocked shape."""
    db_path = tmp_path / "untyped.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="legacy task", assignee="dev")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (tid,))
        kb.block_task(conn, tid, reason="generic stuck")  # kind=None
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "blocked" in text
    assert "generic stuck" in text
    assert "DECISION NEEDED" not in text
