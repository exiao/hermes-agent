"""Tests for the on-completion adversarial-review reaction (constitution rule 2).

When a WRITE-LANE kanban task completes, the gateway notifier should:
  1. spawn a review card parented to the completed task (a separate reviewer /
     second-producer context — never self-review),
  2. post a short progress line to the configured "Kanban Master" chat.

Read-only lanes (researcher, reviewer) and the auto-spawned review cards
themselves must NOT trigger — no review-of-a-review loop. The reaction is
idempotent on the completed task id so the per-chat fan-out of ``default_notify``
spawns at most ONE review.

The first half unit-tests the config resolver + lane gating. The second half is
an INTEGRATION test that drives the real notifier mixin
(``GatewayRunner._kanban_notifier_watcher``) against a temp board.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import gateway.kanban_watchers as kw
from gateway.config import Platform
from gateway.kanban_watchers import (
    _AUTO_REACTION_MARKER,
    _orchestrator_assignee,
    _resolve_on_complete_review_config,
)
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


# --------------------------------------------------------------------------
# Resolver: kanban.on_complete_review config block
# --------------------------------------------------------------------------


def test_resolver_none_when_block_absent():
    assert _resolve_on_complete_review_config(lambda: {"kanban": {}}) is None


def test_resolver_none_when_not_enabled():
    assert _resolve_on_complete_review_config(
        lambda: {"kanban": {"on_complete_review": {"enabled": False}}}
    ) is None


def test_resolver_enabled_without_notify():
    cfg = _resolve_on_complete_review_config(
        lambda: {"kanban": {"on_complete_review": {"enabled": True}}}
    )
    assert cfg == {"notify": None}


def test_resolver_enabled_with_notify_normalized():
    cfg = _resolve_on_complete_review_config(
        lambda: {
            "kanban": {
                "on_complete_review": {
                    "enabled": True,
                    "notify": {"platform": "Signal", "chat_id": "group:km="},
                }
            }
        }
    )
    assert cfg == {
        "notify": {"platform": "signal", "chat_id": "group:km=", "thread_id": ""}
    }


def test_resolver_notify_dropped_when_incomplete():
    # platform present, chat_id blank -> notify is None (spawn silently).
    cfg = _resolve_on_complete_review_config(
        lambda: {
            "kanban": {
                "on_complete_review": {
                    "enabled": True,
                    "notify": {"platform": "signal", "chat_id": ""},
                }
            }
        }
    )
    assert cfg == {"notify": None}


def test_resolver_non_dict_block_fails_safe():
    assert _resolve_on_complete_review_config(
        lambda: {"kanban": {"on_complete_review": "yes"}}
    ) is None


def test_resolver_config_read_error_fails_safe():
    def _boom():
        raise RuntimeError("config read failed")

    assert _resolve_on_complete_review_config(_boom) is None


def test_resolver_non_dict_config_fails_safe():
    assert _resolve_on_complete_review_config(lambda: None) is None
    assert _resolve_on_complete_review_config(lambda: ["x"]) is None


# --------------------------------------------------------------------------
# Orchestrator assignee resolution
# --------------------------------------------------------------------------


def test_orchestrator_assignee_defaults_when_unset():
    assert _orchestrator_assignee(lambda: {}) == "orchestrator"
    assert _orchestrator_assignee(lambda: {"kanban": {}}) == "orchestrator"
    assert _orchestrator_assignee(lambda: {"kanban": {"orchestrator_profile": ""}}) == "orchestrator"


def test_orchestrator_assignee_honours_config():
    assert _orchestrator_assignee(
        lambda: {"kanban": {"orchestrator_profile": "router"}}
    ) == "router"


def test_orchestrator_assignee_fails_safe():
    def _boom():
        raise RuntimeError("nope")
    assert _orchestrator_assignee(_boom) == "orchestrator"


# --------------------------------------------------------------------------
# Gating: _qualifies_for_on_complete_review (the loop guard)
# --------------------------------------------------------------------------


def _runner():
    r = GatewayRunner.__new__(GatewayRunner)
    r._running = True
    r.adapters = {}
    r._kanban_sub_fail_counts = {}
    return r


def test_gate_every_completed_task_qualifies():
    """The new contract: any lane's completion qualifies. The orchestrator,
    not this gate, decides whether a follow-up is warranted."""
    r = _runner()
    for lane in ("cpe-dev", "bloom-dev", "infra-ops", "verifier",
                 "content-creator", "researcher", "reviewer", "random-lane"):
        task = SimpleNamespace(id="t1", assignee=lane, body="some deliverable")
        ok, _reason = r._qualifies_for_on_complete_review(task)
        assert ok, lane


def test_gate_auto_spawned_card_never_qualifies():
    """The loop guard: a decision card stamped with the marker — regardless of
    lane — must never trigger another reaction. This is the structural fix that
    stops review-of-a-review."""
    r = _runner()
    body = f"{_AUTO_REACTION_MARKER}\n# review of t_parent"
    task = SimpleNamespace(id="t_review", assignee="verifier", body=body)
    ok, reason = r._qualifies_for_on_complete_review(task)
    assert not ok
    assert "loop-guard" in reason


# --------------------------------------------------------------------------
# Integration: real notifier mixin, real board, default_notify carries the
# completed event, the reaction spawns a parented review + posts progress.
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


KM_CHAT = "group:KanbanMasterChatId="


def _fake_cfg(enabled=True, notify=True):
    block: dict = {"enabled": enabled}
    if notify:
        block["notify"] = {"platform": "signal", "chat_id": KM_CHAT}
    return {
        "kanban": {
            "dispatch_in_gateway": True,
            # default_notify carries the completed event to a chat so the
            # notifier loop processes it (and fires the reaction inside it).
            "default_notify": [{"platform": "signal", "chat_id": KM_CHAT}],
            "on_complete_review": block,
        }
    }


def _drive_two_ticks(monkeypatch, tmp_path, db_name, assignee, body, *, enabled=True, notify=True):
    """Subscribe (tick 1), complete the task, deliver+react (tick 2).

    Returns (tid, adapter, child_ids_after).
    """
    db_path = tmp_path / db_name
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    cfg = _fake_cfg(enabled=enabled, notify=notify)
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda *a, **k: cfg, raising=False
    )

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="write-lane deliverable", assignee=assignee, body=body)
    finally:
        conn.close()

    adapter = RecordingAdapter()
    # Tick 1: auto-subscribe the active task (no terminal event yet).
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    conn = kb.connect()
    try:
        kb.complete_task(conn, tid, summary="shipped a small PR https://example/pr/1")
    finally:
        conn.close()

    # Tick 2: deliver completion + fire the reaction.
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    conn = kb.connect()
    try:
        kids = kb.child_ids(conn, tid)
        children = [kb.get_task(conn, c) for c in kids]
    finally:
        conn.close()
    return tid, adapter, children


def test_completion_spawns_parented_orchestrator_card_and_posts_progress(tmp_path, monkeypatch):
    tid, adapter, children = _drive_two_ticks(
        monkeypatch, tmp_path, "react.db", assignee="infra-ops", body="opened PR #99",
    )
    # One orchestrator decision card was spawned and parented to the task.
    decision_cards = [c for c in children if c and _AUTO_REACTION_MARKER in (c.body or "")]
    assert len(decision_cards) == 1, f"expected one decision card, got {children}"
    decision = decision_cards[0]
    assert decision.assignee == "orchestrator"
    assert tid in (decision.body or "")

    # Progress landed in the Kanban Master chat.
    texts = [s["text"] for s in adapter.sent if s["chat_id"] == KM_CHAT]
    assert any("completed" in t for t in texts), texts
    assert any(decision.id in t for t in texts), texts


def test_content_lane_completion_also_routes_to_orchestrator(tmp_path, monkeypatch):
    _tid, _adapter, children = _drive_two_ticks(
        monkeypatch, tmp_path, "content.db", assignee="content-creator", body="draft copy",
    )
    decision_cards = [c for c in children if c and _AUTO_REACTION_MARKER in (c.body or "")]
    assert len(decision_cards) == 1
    # Every lane routes to the orchestrator; it decides the second-producer critique.
    assert decision_cards[0].assignee == "orchestrator"


def test_read_only_researcher_completion_also_routes_to_orchestrator(tmp_path, monkeypatch):
    """Under the new contract every completion hands to the orchestrator, which
    then decides a researcher report needs no follow-up. The gate no longer
    filters by lane."""
    _tid, adapter, children = _drive_two_ticks(
        monkeypatch, tmp_path, "researcher.db", assignee="researcher", body="findings",
    )
    decision_cards = [c for c in children if c and _AUTO_REACTION_MARKER in (c.body or "")]
    assert len(decision_cards) == 1
    assert decision_cards[0].assignee == "orchestrator"


def test_auto_spawned_review_card_completion_does_not_loop(tmp_path, monkeypatch):
    """A spawned review card (marker in body), even on a write lane, completing
    must NOT spawn a review-of-a-review."""
    body = f"{_AUTO_REACTION_MARKER}\n# review of t_parent"
    _tid, _adapter, children = _drive_two_ticks(
        monkeypatch, tmp_path, "loop.db", assignee="verifier", body=body,
    )
    review_cards = [c for c in children if c and _AUTO_REACTION_MARKER in (c.body or "")]
    assert review_cards == [], "review card must not spawn another review (loop guard)"


def test_reaction_is_idempotent_across_a_third_tick(tmp_path, monkeypatch):
    """Re-running the notifier (event redelivery / extra tick) must not spawn a
    second review card or a second progress message."""
    db_path = tmp_path / "idem.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    cfg = _fake_cfg()
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda *a, **k: cfg, raising=False
    )
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dep", assignee="bloom-dev", body="PR")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    conn = kb.connect()
    try:
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    # Force the completed event to be re-evaluated by rewinding the cursor so a
    # 3rd tick re-delivers it; the idempotency_key must still prevent a 2nd review.
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    conn = kb.connect()
    try:
        kids = kb.child_ids(conn, tid)
        children = [kb.get_task(conn, c) for c in kids]
    finally:
        conn.close()
    review_cards = [c for c in children if c and _AUTO_REACTION_MARKER in (c.body or "")]
    assert len(review_cards) == 1, f"idempotent: exactly one decision card, got {len(review_cards)}"
    routed = [s for s in adapter.sent if "completed" in s["text"]]
    assert len(routed) == 1, f"exactly one progress message, got {len(routed)}"


def test_disabled_feature_spawns_nothing(tmp_path, monkeypatch):
    _tid, adapter, children = _drive_two_ticks(
        monkeypatch, tmp_path, "off.db", assignee="infra-ops", body="PR", enabled=False,
    )
    review_cards = [c for c in children if c and _AUTO_REACTION_MARKER in (c.body or "")]
    assert review_cards == [], "feature off: no review"
    assert not any("Review routed" in s["text"] for s in adapter.sent)
