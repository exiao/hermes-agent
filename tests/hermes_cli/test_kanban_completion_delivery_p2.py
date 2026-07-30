"""Regression tests for the four open P2 findings on the corrected-handoff path.

Each test exercises the real ``kanban_db`` / gateway / TUI code against a real
temp ``HERMES_HOME`` SQLite board. No source-text inspection, no mocks that
assume the answer — every assertion is about observable behavior (which rows
survive, which events are deliverable, what a serializer returns).

Findings covered:

A. A correction that commits while the original completion is still being
   delivered must not lose its subscriber.
B. Delivery routing identifiers must never appear in the public audit stream.
C. A corrected completion must keep the verified child-card manifest.
D. A correction must be rejected once ANY newer run / reopening happened.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# (A) Preserve subscriptions during a concurrent correction
# ---------------------------------------------------------------------------

def test_terminal_unsub_skips_removal_when_a_newer_completion_landed(kanban_home):
    """The in-flight delivery window must not orphan a corrected completion.

    Sequence (exactly the production race):
      1. worker claims + completes the task with a bare handoff
      2. the notifier claims the ``completed`` event and starts sending
      3. *while that send is in flight* the owner commits a correction — no
         completion_delivery record exists yet, so restoration is a no-op and
         a second ``completed`` event is appended
      4. the send returns and the notifier finalizes the terminal delivery

    At step 4 the subscription must survive, because the corrected event is
    newer than the event that was actually delivered.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="concurrent correction", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert owner is not None
        assert kb.complete_task(conn, tid, summary="bare handoff")

        # 2. notifier claims the completed event (this is the delivered cursor)
        _old, delivered_cursor, events = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=("completed",),
        )
        assert [e.kind for e in events] == ["completed"]

        # 3. owner correction commits mid-flight
        assert kb.complete_task(
            conn, tid,
            summary="corrected handoff",
            metadata={"commit_sha": "abc123"},
            expected_run_id=owner.id,
        )

        # 4. notifier finalizes terminal delivery for the event it sent
        kb.finalize_terminal_delivery(
            conn,
            {"task_id": tid, "platform": "telegram", "chat_id": "chat-1"},
            delivered_through_event_id=delivered_cursor,
        )

        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive when a newer completion appeared after "
            f"the delivered event; got {subs!r}"
        )
        _o, _n, pending = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=("completed",),
        )
        assert [e.kind for e in pending] == ["completed"], (
            "The corrected completion must still be deliverable to the "
            "surviving subscriber"
        )
        assert pending[0].payload.get("summary") == "corrected handoff"
    finally:
        conn.close()


def test_terminal_unsub_records_and_removes_atomically(kanban_home):
    """With no concurrent correction, the terminal delivery still finalizes."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="clean delivery", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert kb.complete_task(conn, tid, summary="bare handoff")

        _o, cursor, events = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=("completed",),
        )
        assert events
        kb.finalize_terminal_delivery(
            conn,
            {"task_id": tid, "platform": "telegram", "chat_id": "chat-1"},
            delivered_through_event_id=cursor,
        )
        assert kb.list_notify_subs(conn, tid) == []

        # And the recorded target is restored by a later owner correction.
        assert kb.complete_task(
            conn, tid,
            summary="corrected handoff",
            metadata={"commit_sha": "abc123"},
            expected_run_id=owner.id,
        )
        subs = kb.list_notify_subs(conn, tid)
        assert [(s["platform"], s["chat_id"]) for s in subs] == [("telegram", "chat-1")]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_gateway_notifier_keeps_sub_when_correction_races_the_send(kanban_home):
    """The real gateway notifier tick must not orphan a mid-send correction."""
    from unittest.mock import MagicMock
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="gateway race", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert kb.complete_task(conn, tid, summary="bare handoff")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}

    async def _send_and_race(*a, **kw):
        # The owner correction commits while this send is in flight.
        c2 = kb.connect()
        try:
            kb.complete_task(
                c2, tid,
                summary="corrected handoff",
                metadata={"commit_sha": "abc123"},
                expected_run_id=owner.id,
            )
        finally:
            c2.close()
        runner._running = False
        return None

    fake_adapter = MagicMock()
    fake_adapter.send = _send_and_race
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    original_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await original_sleep(0)

    from unittest.mock import patch
    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(runner._kanban_notifier_watcher(interval=1), timeout=20.0)

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Gateway notifier dropped the subscription even though a corrected "
            f"completion landed during the send; got {subs!r}"
        )
        _o, _n, pending = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=("completed",),
        )
        assert [e.payload.get("summary") for e in pending] == ["corrected handoff"]
    finally:
        conn.close()


def test_tui_poller_keeps_sub_when_correction_races_the_delivery(kanban_home, monkeypatch):
    """The TUI notification poller shares the race and must behave the same."""
    import tui_gateway.server as tui_server

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="tui race", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="tui", chat_id="tui-session")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert kb.complete_task(conn, tid, summary="bare handoff")
    finally:
        conn.close()

    real_format = tui_server._format_kanban_event_text
    raced: list[bool] = []

    def _format_and_race(sub, task, ev, slug):
        if not raced:
            raced.append(True)
            c2 = kb.connect()
            try:
                kb.complete_task(
                    c2, tid,
                    summary="corrected handoff",
                    metadata={"commit_sha": "abc123"},
                    expected_run_id=owner.id,
                )
            finally:
                c2.close()
        return real_format(sub, task, ev, slug)

    monkeypatch.setattr(tui_server, "_format_kanban_event_text", _format_and_race)
    texts = tui_server._collect_kanban_notifications({"session_key": "tui-session"})
    assert texts, "the original completion should still have been delivered"

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "TUI poller dropped the subscription even though a corrected "
            f"completion landed during delivery; got {subs!r}"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (B) Keep delivery routing data out of serialized events
# ---------------------------------------------------------------------------

_ROUTING_SECRETS = (
    "chat-secret-1", "user-secret-1", "thread-secret-1",
    "profile-secret-1", "anchor-secret-1",
)


def _task_with_recorded_delivery(conn) -> tuple[str, int]:
    tid = kb.create_task(conn, title="routing leak", assignee="worker")
    kb.add_notify_sub(
        conn, task_id=tid, platform="telegram", chat_id="chat-secret-1",
        thread_id="thread-secret-1", user_id="user-secret-1",
        chat_type="private", notifier_profile="profile-secret-1",
        delivery_metadata={"reply_anchor": "anchor-secret-1"},
    )
    kb.claim_task(conn, tid)
    owner = kb.latest_run(conn, tid)
    assert kb.complete_task(conn, tid, summary="bare handoff")
    sub = kb.list_notify_subs(conn, tid)[0]
    _o, cursor, _ev = kb.claim_unseen_events_for_sub(
        conn, task_id=tid, platform="telegram", chat_id="chat-secret-1",
        thread_id="thread-secret-1", kinds=("completed",),
    )
    kb.finalize_terminal_delivery(conn, sub, delivered_through_event_id=cursor)
    return tid, owner.id


def test_recorded_delivery_never_enters_the_public_event_stream(kanban_home):
    conn = kb.connect()
    try:
        tid, _run_id = _task_with_recorded_delivery(conn)
        events = kb.list_events(conn, tid)
        blob = json.dumps(
            [{"kind": e.kind, "payload": e.payload} for e in events],
            ensure_ascii=False,
        )
        for secret in _ROUTING_SECRETS:
            assert secret not in blob, (
                f"routing identifier {secret!r} leaked into the task event "
                f"stream: {blob}"
            )
        assert not any(e.kind == "completion_delivery" for e in events)
    finally:
        conn.close()


def test_kanban_show_does_not_leak_delivery_routing(kanban_home, monkeypatch):
    from tools import kanban_tools

    conn = kb.connect()
    try:
        tid, _run_id = _task_with_recorded_delivery(conn)
    finally:
        conn.close()

    out = kanban_tools._handle_show({"task_id": tid})
    for secret in _ROUTING_SECRETS:
        assert secret not in out, (
            f"kanban_show leaked routing identifier {secret!r} to the model: {out}"
        )


def test_recorded_delivery_still_restores_after_owner_correction(kanban_home):
    """Moving the state out of task_events must not break the feature."""
    conn = kb.connect()
    try:
        tid, run_id = _task_with_recorded_delivery(conn)
        assert kb.list_notify_subs(conn, tid) == []
        assert kb.complete_task(
            conn, tid,
            summary="corrected handoff",
            metadata={"commit_sha": "abc123"},
            expected_run_id=run_id,
        )
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["chat_id"] == "chat-secret-1"
        assert subs[0]["thread_id"] == "thread-secret-1"
        assert subs[0]["user_id"] == "user-secret-1"
        assert subs[0]["notifier_profile"] == "profile-secret-1"
        assert subs[0]["delivery_metadata"] == {"reply_anchor": "anchor-secret-1"}
    finally:
        conn.close()


def test_delivery_records_are_cascaded_on_task_delete(kanban_home):
    conn = kb.connect()
    try:
        tid, _run_id = _task_with_recorded_delivery(conn)
        kb.delete_task(conn, tid)
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM kanban_completion_deliveries WHERE task_id = ?",
            (tid,),
        ).fetchone()
        assert rows["n"] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (C) Preserve verified cards in corrected completion events
# ---------------------------------------------------------------------------

def test_corrected_completion_keeps_verified_cards(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="parent with children", assignee="worker")
        child = kb.create_task(conn, title="child card", parents=[tid], created_by="worker")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert kb.complete_task(conn, tid, summary="bare handoff")

        assert kb.complete_task(
            conn, tid,
            summary="corrected handoff",
            metadata={"commit_sha": "abc123"},
            created_cards=[child],
            expected_run_id=owner.id,
        )
        completed = [
            e for e in kb.list_events(conn, tid)
            if e.kind == "completed"
        ]
        assert completed, "expected a re-emitted completed event"
        latest = completed[-1]
        assert latest.payload.get("verified_cards") == [child], (
            "the corrected completion event lost the verified child-card "
            f"manifest: {latest.payload!r}"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (D) Reject corrections after any newer task run
# ---------------------------------------------------------------------------

def test_correction_rejected_after_a_newer_reclaimed_run(kanban_home):
    """A newer run that ended 'reclaimed' still invalidates the old correction."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="reclaimed newer run", assignee="worker")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert owner is not None
        assert kb.complete_task(conn, tid, summary="bare handoff")

        # Reopen, a second worker claims it, then a dashboard drag moves it
        # straight to done — the second run ends with outcome='reclaimed'.
        assert kb.set_status_direct(conn, tid, "ready")
        kb.claim_task(conn, tid)
        second = kb.latest_run(conn, tid)
        assert second is not None and second.id != owner.id
        assert kb.set_status_direct(conn, tid, "done")
        assert kb.latest_run(conn, tid).outcome == "reclaimed"

        before = len([e for e in kb.list_events(conn, tid) if e.kind == "completed"])
        assert not kb.complete_task(
            conn, tid,
            result="stale owner result",
            summary="stale owner handoff",
            metadata={"commit_sha": "abc123"},
            expected_run_id=owner.id,
        ), "a lingering first worker must not rewrite a superseded run"
        after = len([e for e in kb.list_events(conn, tid) if e.kind == "completed"])
        assert after == before, "no extra completed event may be emitted"
        run = conn.execute(
            "SELECT summary FROM task_runs WHERE id = ?", (owner.id,)
        ).fetchone()
        assert run["summary"] != "stale owner handoff"
    finally:
        conn.close()


def test_correction_rejected_after_a_reopen_and_manual_close(kanban_home):
    """A reopen + manual re-close with no new run also invalidates it."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="reopened manually", assignee="worker")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert kb.complete_task(conn, tid, summary="bare handoff")

        assert kb.set_status_direct(conn, tid, "ready")
        assert kb.set_status_direct(conn, tid, "done")

        assert not kb.complete_task(
            conn, tid,
            summary="stale owner handoff",
            metadata={"commit_sha": "abc123"},
            expected_run_id=owner.id,
        ), "a reopening between completion and correction must invalidate it"
    finally:
        conn.close()


def test_correction_still_accepted_without_any_newer_lifecycle(kanban_home):
    """The feature itself must keep working — no over-rejection."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="happy path", assignee="worker")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert kb.complete_task(conn, tid, summary="bare handoff")
        assert kb.complete_task(
            conn, tid,
            result="real result",
            summary="corrected handoff",
            metadata={"commit_sha": "abc123"},
            expected_run_id=owner.id,
        )
        assert kb.get_task(conn, tid).result == "real result"
    finally:
        conn.close()


def test_correction_preserves_original_completion_evidence(kanban_home):
    """A correction adds the missing push field; it must not erase prior evidence.

    The owner's first completion can already carry changed_files/artifacts/
    created_cards but lack commit_sha. A later correction supplying ONLY the
    push field previously replaced the run's whole metadata object and rebuilt
    the event from correction args alone, silently dropping everything the
    original recorded unless the worker happened to repeat it.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="evidence merge", assignee="worker")
        kb.claim_task(conn, tid)
        owner = kb.latest_run(conn, tid)
        assert owner is not None
        assert kb.complete_task(
            conn, tid,
            summary="did the work",
            metadata={"changed_files": ["a.py", "b.py"], "artifacts": ["/tmp/report.md"]},
        )

        # Correction supplies ONLY the push evidence that was missing.
        assert kb.complete_task(
            conn, tid,
            summary="did the work",
            metadata={"commit_sha": "abc123", "pr_url": "https://example/pr/1"},
            expected_run_id=owner.id,
        )

        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (owner.id,)
        ).fetchone()
        merged = json.loads(row["metadata"])
        assert merged["commit_sha"] == "abc123"
        assert merged["changed_files"] == ["a.py", "b.py"], "prior evidence erased"
        assert merged["artifacts"] == ["/tmp/report.md"], "prior artifacts erased"
    finally:
        conn.close()
