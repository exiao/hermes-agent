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
    """Trip the real retry breaker so the gave_up event payload is built by
    the production path (`_record_task_failure`), not hand-written. With
    ``failure_limit=1`` a single failure crosses the threshold and emits the
    ``gave_up`` event the notifier reads."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="gaveup notify", assignee="dev")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.claim_task(conn, tid)
        kb._record_task_failure(
            conn, tid, error=error,
            outcome="spawn_failed", release_claim=True, end_run=True,
            failure_limit=1,
        )
        return tid
    finally:
        conn.close()


def _completed_subscription(summary):
    """Complete via the real `complete_task` producer so the event payload is
    built (and capped) exactly as production does, not hand-written."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="done notify", assignee="dev")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary=summary)
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
    # A reason with no routing/retry hints self-labels as the human-decision
    # header (the legacy "⏸ … blocked" shape is now only used for empty reasons).
    assert "🔴 DECISION NEEDED" in text
    # The whole reason survives — including the tail that the 160-char slice
    # used to drop.
    assert reason in text
    assert "split the feature into its own card." in text


def test_gave_up_error_survives_producer_and_notifier(tmp_path, monkeypatch):
    """A long gave_up error must survive BOTH the producer
    (`_record_task_failure`, which used to slice the event payload at 500) and
    the notifier (which used to slice at 200). Uses a >500-char error driven
    through the real breaker so it regression-guards the producer pre-truncation
    Codex flagged, not just the notifier slice."""
    db_path = tmp_path / "gaveup-untruncate.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    error = (
        "task t_48c55d21 worktree path '/Users/x/projects/cpe-research' is not "
        "inside a git repo and does not point at a git repo root. The dispatcher "
        "walked up looking for a repo root, found none, and could not materialize "
        "a linked worktree, so the worker never spawned. "
    ) * 3  # >500 chars: exceeds the old producer cap, not just the notifier cap
    assert len(error) > 500

    tid = _gave_up_subscription(error)
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "gave up" in text
    # Full error survives end to end — the tail past 500 chars (which the
    # producer used to drop before the notifier ever saw it) is present.
    assert error.strip() in text


def test_completed_long_summary_survives_producer_400_cap(tmp_path, monkeypatch):
    """A completed summary whose first line exceeds 400 chars must survive the
    real `complete_task` producer (which used to slice the event payload first
    line at 400) and reach the notifier intact. Guards the completed-path
    pre-truncation Codex flagged: the notifier fix alone couldn't help because
    the producer dropped the tail before the notifier saw it."""
    db_path = tmp_path / "done-long.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # Single line, >400 chars, well under the notifier's 3500 visible cap.
    summary = (
        "Landed the security hotfix and verified it end to end: "
        + "escaped every leaderboard field, added a clamp helper, re-synced "
        "200.html byte-equal, ran the targeted suite green, and confirmed the "
        "stored-XSS payload no longer executes in the rendered leaderboard. "
        * 3
    ).replace("\n", " ")
    assert len(summary) > 400 and len(summary) < 3500

    tid = _completed_subscription(summary)
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "done" in text
    # The tail past 400 chars (which the producer used to drop) is present.
    assert summary.strip() in text


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


def test_gave_up_run_row_stores_full_error_for_retry(tmp_path, monkeypatch):
    """The durable attempt record (`task_runs.error`) must keep the full error,
    not a 500-char stub. `build_worker_context` feeds prior-attempt errors to
    the NEXT retry (capped at 4KB there), so slicing to 500 in `_end_run` would
    hand a retrying worker a truncated prior failure — while the completed path
    stores its summary in full. This guards that asymmetry."""
    db_path = tmp_path / "gaveup-runrow.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    error = (
        "spawn failed: worktree path is not inside a git repo and does not point "
        "at a git repo root; the dispatcher walked up, found no repo, and could "
        "not materialize a linked worktree. "
    ) * 4  # >500 chars
    assert len(error) > 500

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="runrow", assignee="dev")
        kb.claim_task(conn, tid)
        kb._record_task_failure(
            conn, tid, error=error,
            outcome="spawn_failed", release_claim=True, end_run=True,
            failure_limit=1,
        )
        runs = kb.list_runs(conn, tid, include_active=False)
    finally:
        conn.close()

    assert runs, "a closed run should exist after gave_up"
    run_error = runs[-1].error
    assert run_error is not None
    # The full error survives on the run row — the tail past 500 chars (which
    # the old _end_run slice dropped before any retry could read it) is present.
    assert error.strip() in run_error


def test_completed_whitespace_only_summary_has_no_trailing_newline(tmp_path, monkeypatch):
    """A whitespace-only summary must not leave a dangling ``\\n`` on the headline.

    ``_clip_notify_detail`` strips its input, so a whitespace-only summary clips
    to ``""``; prepending ``\\n`` unconditionally would emit ``"... done — title\\n"``.
    The handoff must be omitted entirely when the clipped detail is empty.
    """
    db_path = tmp_path / "done-whitespace.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _completed_subscription("   \n  \t ")
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert "done" in text
    # No empty handoff line tacked on the end.
    assert not text.endswith("\n")
    assert "\n" not in text


def test_capped_handoff_plus_envelope_fits_under_4000_char_adapter(tmp_path, monkeypatch):
    """The whole notification (envelope + clipped detail + suffix) must stay under
    the tightest common adapter cap (4000, e.g. WeCom) so the advertised "see
    board" suffix is never silently re-clipped off the tail on capped platforms.
    """
    db_path = tmp_path / "done-capped.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # A summary well past the inline cap forces the visible-suffix path.
    summary = "x" * (_NOTIFY_DETAIL_MAX + 1000)
    tid = _completed_subscription(summary)
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    # The advertised suffix is present...
    assert "more chars; see board" in text
    # ...and the full message — envelope included — fits under a 4000-char
    # adapter, so ``content[:4000]`` would not chop the suffix off.
    assert len(text) <= 4000
