"""Tests for live board-wide auto-subscribe target resolution.

``kanban.default_notify`` fans every ticket's terminal events out to the
listed chats (e.g. a Signal group) with zero manual ``notify-subscribe``.
``_resolve_default_notify_targets`` is called every notifier tick (like
``_resolve_auto_decompose_settings``) so editing the list takes effect on the
next tick without a gateway restart, and it must fail SAFE (empty list = no
auto-subscribe) on any malformed/erroring config so the notifier loop never
dies.

The second half is an INTEGRATION test that drives the real notifier mixin
(``GatewayRunner._kanban_notifier_watcher``) against a temp board with
``kanban.default_notify`` set and NO manual subscription, proving a completed
ticket is delivered to the configured target end-to-end through the real
``_collect`` default-subscribe block + the real delivery path.
"""

from __future__ import annotations

import asyncio

import gateway.kanban_watchers as kw
from gateway.config import Platform
from gateway.kanban_watchers import _resolve_default_notify_targets
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


def test_empty_when_key_absent():
    assert _resolve_default_notify_targets(lambda: {"kanban": {}}) == []


def test_empty_when_default_notify_empty():
    assert _resolve_default_notify_targets(
        lambda: {"kanban": {"default_notify": []}}
    ) == []


def test_single_target_normalized():
    targets = _resolve_default_notify_targets(
        lambda: {
            "kanban": {
                "default_notify": [
                    {"platform": "Signal", "chat_id": "group:abc="},
                ]
            }
        }
    )
    assert targets == [
        {"platform": "signal", "chat_id": "group:abc=", "thread_id": ""}
    ]


def test_thread_id_preserved():
    targets = _resolve_default_notify_targets(
        lambda: {
            "kanban": {
                "default_notify": [
                    {"platform": "telegram", "chat_id": "123", "thread_id": "7"},
                ]
            }
        }
    )
    assert targets == [
        {"platform": "telegram", "chat_id": "123", "thread_id": "7"}
    ]


def test_entries_missing_platform_or_chat_id_dropped():
    targets = _resolve_default_notify_targets(
        lambda: {
            "kanban": {
                "default_notify": [
                    {"platform": "signal"},            # no chat_id
                    {"chat_id": "group:x="},           # no platform
                    {"platform": "", "chat_id": "y="},  # blank platform
                    {"platform": "signal", "chat_id": "group:ok="},  # valid
                ]
            }
        }
    )
    assert targets == [
        {"platform": "signal", "chat_id": "group:ok=", "thread_id": ""}
    ]


def test_non_dict_entries_skipped():
    targets = _resolve_default_notify_targets(
        lambda: {
            "kanban": {
                "default_notify": [
                    "not-a-dict",
                    None,
                    {"platform": "signal", "chat_id": "group:ok="},
                ]
            }
        }
    )
    assert targets == [
        {"platform": "signal", "chat_id": "group:ok=", "thread_id": ""}
    ]


def test_non_list_default_notify_fails_safe():
    assert _resolve_default_notify_targets(
        lambda: {"kanban": {"default_notify": "group:x="}}
    ) == []


def test_non_dict_config_fails_safe():
    assert _resolve_default_notify_targets(lambda: None) == []
    assert _resolve_default_notify_targets(lambda: ["not", "a", "dict"]) == []


def test_malformed_kanban_section_fails_safe():
    # A live edit that leaves ``kanban`` as a non-dict (e.g. ``kanban: false``
    # or a YAML list) must not raise — it would break ALL per-task delivery.
    assert _resolve_default_notify_targets(lambda: {"kanban": False}) == []
    assert _resolve_default_notify_targets(lambda: {"kanban": ["a", "b"]}) == []
    assert _resolve_default_notify_targets(lambda: {"kanban": "nope"}) == []


def test_config_read_error_fails_safe_empty():
    def _boom():
        raise RuntimeError("config read failed")

    assert _resolve_default_notify_targets(_boom) == []


def test_live_edit_takes_effect_between_calls():
    state = {"kanban": {"default_notify": []}}
    assert _resolve_default_notify_targets(lambda: state) == []
    state["kanban"]["default_notify"] = [
        {"platform": "signal", "chat_id": "group:new="}
    ]
    assert _resolve_default_notify_targets(lambda: state) == [
        {"platform": "signal", "chat_id": "group:new=", "thread_id": ""}
    ]


# --------------------------------------------------------------------------
# Integration: real notifier mixin, real board, default_notify drives delivery
# with NO manual notify-subscribe.
# --------------------------------------------------------------------------


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter, platform=Platform.SIGNAL):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {platform: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def test_default_notify_delivers_without_manual_subscribe(tmp_path, monkeypatch):
    """A ticket fans out to the config-listed chat with zero manual
    ``notify-subscribe``. Drives the real ``_collect`` default-subscribe block
    + the real delivery path across the realistic two-tick lifecycle:
    tick #1 subscribes the active (``ready``) task; the task then completes;
    tick #2 delivers the completion to the auto-added target."""
    db_path = tmp_path / "default-notify.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    target_chat = "group:y94kMF95wnSq3UYCBM6Ihei1ViUakcoGlmZkcTafwjk="

    fake_cfg = {
        "kanban": {
            "dispatch_in_gateway": True,
            "default_notify": [
                {"platform": "signal", "chat_id": target_chat},
            ],
        }
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda *a, **k: fake_cfg, raising=False
    )

    # Create an active task (status `ready`) — NO manual notify subscription.
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="auto fanout", assignee="researcher")
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()

    adapter = RecordingAdapter()

    # Tick #1: the default-notify block subscribes the active task. Nothing to
    # deliver yet (no terminal event).
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert adapter.sent == []
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1 and subs[0]["chat_id"] == target_chat, (
            "active task must be auto-subscribed to the default-notify target"
        )
        # Worker finishes.
        kb.complete_task(conn, tid, summary="echo ok done")
    finally:
        conn.close()

    # Tick #2: the completed event is delivered to the auto-added target.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, (
        f"expected one auto-fanout delivery, got {adapter.sent}"
    )
    assert adapter.sent[0]["chat_id"] == target_chat
    assert tid in adapter.sent[0]["text"]
    assert "done" in adapter.sent[0]["text"].lower()


def test_default_notify_does_not_disturb_existing_per_task_sub(tmp_path, monkeypatch):
    """The per-task subscribe path is untouched: a pre-existing subscription to
    a different chat keeps delivering, and the default target is added
    alongside it (no clobber, no regression)."""
    db_path = tmp_path / "coexist.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    default_chat = "group:y94kMF95wnSq3UYCBM6Ihei1ViUakcoGlmZkcTafwjk="
    manual_chat = "group:manual-existing="

    fake_cfg = {
        "kanban": {
            "dispatch_in_gateway": True,
            "default_notify": [{"platform": "signal", "chat_id": default_chat}],
        }
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda *a, **k: fake_cfg, raising=False
    )

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="coexist", assignee="researcher")
        # Pre-existing manual subscription to a DIFFERENT chat.
        kb.add_notify_sub(conn, task_id=tid, platform="signal", chat_id=manual_chat)
    finally:
        conn.close()

    adapter = RecordingAdapter()

    # Tick #1: default-notify adds its target alongside the manual sub while
    # the task is still active.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    conn = kb.connect()
    try:
        chats = sorted(s["chat_id"] for s in kb.list_notify_subs(conn, tid))
        assert chats == sorted([default_chat, manual_chat]), (
            f"both subs should coexist after tick #1; got {chats}"
        )
        kb.complete_task(conn, tid, summary="ok")
    finally:
        conn.close()

    # Tick #2: both targets receive the completion.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    chats = sorted(d["chat_id"] for d in adapter.sent)
    assert chats == sorted([default_chat, manual_chat]), (
        f"both the manual and default targets should receive the completion; got {chats}"
    )


def test_add_default_notify_subs_bulk_active_only(tmp_path, monkeypatch):
    """The bulk helper subscribes every active task and skips final ones in a
    single transaction, and is idempotent on re-run (no duplicate rows)."""
    db_path = tmp_path / "bulk.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    chat = "group:bulk-target="
    conn = kb.connect()
    try:
        active1 = kb.create_task(conn, title="a1", assignee="researcher")
        active2 = kb.create_task(conn, title="a2", assignee="researcher")
        done = kb.create_task(conn, title="d", assignee="researcher")
        kb.complete_task(conn, done, summary="ok")

        kb.add_default_notify_subs(conn, platform="signal", chat_id=chat)
        subscriptions = kb.list_notify_subs(conn)
        subscribed = {s["task_id"] for s in subscriptions}
        assert subscribed == {active1, active2}, (
            f"only active tasks should be subscribed; got {subscribed}"
        )
        # Default subscriptions begin caught up, just like an explicit
        # subscribe, so enabling them never replays an active task's history.
        for sub in subscriptions:
            latest = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM task_events WHERE task_id = ?",
                (sub["task_id"],),
            ).fetchone()[0]
            assert sub["last_event_id"] == latest

        # Idempotent: a second call adds no duplicate rows.
        kb.add_default_notify_subs(conn, platform="signal", chat_id=chat)
        assert len(kb.list_notify_subs(conn)) == 2
    finally:
        conn.close()
