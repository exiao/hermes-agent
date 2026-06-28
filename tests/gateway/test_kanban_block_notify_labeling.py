"""Regression: blocked-task notifications are self-labeling.

The Signal/Telegram push for a blocked task leads with a header that says — at a
glance — whether the reader must act:

  🔴 DECISION NEEDED — <id>: <title>   (a human has to decide; the default)
  🟠 ROUTING — <id>: <title>           (a hard wall: creds/access/lane)
  🟡 RETRY — <id>: <title>             (flaky / transient)

The header needs NO agent-facing contract: a worker writes a plain ``reason``
and the watcher infers the tag from the text (``_infer_block_header``). When a
worker did pass the optional upstream ``kind`` param it is honored directly. An
empty reason keeps the historical ``⏸ … blocked`` shape.

Tests drive the real ``_kanban_notifier_watcher`` against a temp DB so the whole
producer → notifier path is exercised, plus pure-formatter and pure-classifier
units.
"""

import asyncio

from gateway.config import Platform
from gateway.kanban_watchers import (
    _format_block_notification,
    _infer_block_header,
)
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


def _block_subscription(reason, kind=None, title="do the thing"):
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
# Pure classifier units — the reason text drives the header, no kind needed
# ---------------------------------------------------------------------------


def test_infer_routing_from_access_wording():
    assert _infer_block_header("No access to the prod vault to rotate the token") == "🟠 ROUTING"
    assert _infer_block_header("403 forbidden hitting the deploy API") == "🟠 ROUTING"


def test_infer_retry_from_transient_wording():
    assert _infer_block_header("429 rate limit from the search API, try again later") == "🟡 RETRY"
    assert _infer_block_header("connection reset, looks flaky") == "🟡 RETRY"


def test_infer_defaults_to_decision_needed():
    # Anything that isn't clearly routing/transient is treated as a human
    # decision — the safe default (surface it rather than hide it).
    assert _infer_block_header("Should I key the limiter on IP or user_id?") == "🔴 DECISION NEEDED"


def test_infer_empty_reason_has_no_header():
    assert _infer_block_header("") is None
    assert _infer_block_header("   ") is None


# ---------------------------------------------------------------------------
# Pure formatter units
# ---------------------------------------------------------------------------


def test_format_infers_decision_needed_from_reason():
    msg = _format_block_notification(
        None, "t_abc", "merge CPE PRs #795-799", "which env should I migrate first?"
    )
    assert msg.splitlines()[0] == "🔴 DECISION NEEDED — t_abc: merge CPE PRs #795-799"
    assert "which env should I migrate first?" in msg


def test_format_infers_routing_from_reason():
    msg = _format_block_notification(
        None, "t_def", "rotate prod token", "no access to the vault"
    )
    assert msg.splitlines()[0] == "🟠 ROUTING — t_def: rotate prod token"
    assert "DECISION NEEDED" not in msg


def test_format_infers_retry_from_reason():
    msg = _format_block_notification(None, "t_g", "crawl feed", "429 from API, transient")
    assert msg.splitlines()[0] == "🟡 RETRY — t_g: crawl feed"


def test_format_explicit_kind_overrides_inference():
    # A worker that DID pass the optional kind gets exactly that header even if
    # the reason text would infer differently.
    msg = _format_block_notification(
        "capability", "t_h", "do thing", "should I pick A or B?"
    )
    assert msg.splitlines()[0] == "🟠 ROUTING — t_h: do thing"


def test_format_empty_reason_keeps_legacy_blocked_shape():
    msg = _format_block_notification(None, "t_i", "do thing", "")
    assert msg == "⏸ Kanban t_i blocked"


def test_format_includes_assignee_tag():
    msg = _format_block_notification(
        None, "t_a", "title", "pick a region", tag="@dev "
    )
    assert msg.splitlines()[0] == "🔴 DECISION NEEDED — @dev t_a: title"


# ---------------------------------------------------------------------------
# End-to-end: real notifier watcher against a temp DB
# ---------------------------------------------------------------------------


def test_decision_block_pushes_decision_needed_header(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "k.db"))
    kb.init_db()
    _block_subscription("Should I key the limiter on IP or user_id?")
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.sent, "expected a push"
    first = adapter.sent[0]["text"].splitlines()[0]
    assert first.startswith("🔴 DECISION NEEDED — ")


def test_routing_block_pushes_routing_header_not_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "k.db"))
    kb.init_db()
    _block_subscription("No access to the vault, cannot rotate the prod token")
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.sent, "expected a push"
    first = adapter.sent[0]["text"].splitlines()[0]
    assert first.startswith("🟠 ROUTING — ")
    assert "DECISION NEEDED" not in adapter.sent[0]["text"]


def test_empty_reason_block_keeps_legacy_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "k.db"))
    kb.init_db()
    _block_subscription("")
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.sent, "expected a push"
    text = adapter.sent[0]["text"]
    assert "⏸" in text and "blocked" in text
    assert "DECISION NEEDED" not in text
