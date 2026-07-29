"""Block-loop tasks must not be laundered back into the work pool.

Regression for the 296-blocked-run loop on t_4e495923 (babysit bloom#2130),
plus seven sibling cards at 76-288 blocked runs each.

Two guards that each work alone and defeated each other:

* ``block_task`` routes a task to ``status='triage'`` after
  ``BLOCK_RECURRENCE_LIMIT`` re-blocks for the same cause, emitting
  ``block_loop_detected``. This fired 295 times on that card.
* ``specify_triage_task`` exists to flesh out unspecified one-liner triage
  cards and flips them ``triage -> todo``, then calls ``recompute_ready``.

``triage`` is where the breaker PARKS a looping task and where the specifier
HARVESTS work, so the observed cycle every ~4 minutes was:

    block_loop_detected -> specified -> promoted -> claimed -> spawned -> ...

``block_recurrences`` read 0 on all eight cards because each lap reset it.
"""

import pytest

from hermes_cli import kanban_db as kb


def _fresh_board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "block-loop.db"))
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb.connect()


def _drive_into_block_loop(conn, task_id):
    """Re-block for the same cause until the breaker routes to triage.

    Uses a truly-blocked kind (``needs_input``). ``kind="dependency"`` is a
    SOFT park: it emits ``dependency_wait`` and lands in ``todo``, never
    reaching the recurrence breaker.
    """
    for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 2):
        task = kb.get_task(conn, task_id)
        if task.status == "triage":
            break
        if task.status not in ("running", "ready"):
            kb.set_status_direct(conn, task_id, "ready")
        kb.claim_task(conn, task_id, claimer="test")
        kb.block_task(conn, task_id, reason="needs a human call", kind="needs_input")
        if kb.get_task(conn, task_id).status == "blocked":
            kb.unblock_task(conn, task_id)
    return kb.get_task(conn, task_id)


def test_block_loop_routes_to_triage(tmp_path, monkeypatch):
    """Precondition: the breaker still parks a same-cause re-blocker in triage."""
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(
            conn, title="babysit a PR whose base is broken", assignee="pr-babysitter",
        )
        task = _drive_into_block_loop(conn, tid)
        assert task.status == "triage"
        kinds = [
            e.kind for e in kb.list_events(conn, tid)
            if e.kind == "block_loop_detected"
        ]
        assert kinds, "breaker must emit block_loop_detected"
    finally:
        conn.close()


def test_specifier_refuses_a_block_loop_task(tmp_path, monkeypatch):
    """The specifier must NOT flip a block-loop triage task back to todo.

    This is the actual 296-run bug: specifying it re-enters the work pool.
    """
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(
            conn, title="babysit a PR whose base is broken", assignee="pr-babysitter",
        )
        _drive_into_block_loop(conn, tid)
        assert kb.get_task(conn, tid).status == "triage"

        ok = kb.specify_triage_task(
            conn, tid,
            title="Babysit bloom#2130 to merge-ready",
            body="A real spec with plenty of detail so the spec-less guard passes.",
            assignee="pr-babysitter",
            author="specifier",
        )

        assert ok is False, "specifier must refuse a block-loop task"
        assert kb.get_task(conn, tid).status == "triage", "must stay parked in triage"
    finally:
        conn.close()


def test_operator_status_move_clears_the_loop(tmp_path, monkeypatch):
    """An operator move out of triage is the escape hatch — not a dead end.

    ``unblock_task`` only accepts ``blocked``/``scheduled``, so it CANNOT exit
    ``triage``. If the guard keyed on ``unblocked`` alone the card would be
    trapped forever, so a ``status`` move must clear it.
    """
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(
            conn, title="babysit a PR whose base is broken", assignee="pr-babysitter",
        )
        _drive_into_block_loop(conn, tid)
        assert kb._has_unresolved_block_loop(conn, tid) is True
        assert kb.unblock_task(conn, tid) is False, (
            "precondition: unblock_task cannot exit triage"
        )

        kb.set_status_direct(conn, tid, "todo")

        assert kb._has_unresolved_block_loop(conn, tid) is False, (
            "an operator status move must release the guard"
        )
    finally:
        conn.close()


def test_blocked_run_hard_cap_stops_auto_promotion(tmp_path, monkeypatch):
    """Past BLOCKED_RUN_HARD_CAP, recompute_ready must not re-promote.

    Independent of which state machine routed the task — this is the ceiling
    that cannot be laundered by flipping status.
    """
    conn = _fresh_board(tmp_path, monkeypatch)
    try:
        tid = kb.create_task(conn, title="loops forever", assignee="worker")
        for _ in range(kb.BLOCKED_RUN_HARD_CAP + 2):
            kb._synthesize_ended_run(
                conn, tid, outcome="blocked", summary="same cause again",
            )
        assert _count_blocked(conn, tid) > kb.BLOCKED_RUN_HARD_CAP

        kb.set_status_direct(conn, tid, "blocked")
        kb.recompute_ready(conn)

        assert kb.get_task(conn, tid).status == "blocked", (
            "hard cap must stop auto-promotion"
        )
    finally:
        conn.close()


def _count_blocked(conn, task_id):
    return kb._blocked_run_count(conn, task_id)
