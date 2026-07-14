"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------


def test_first_typed_block_lands_in_blocked(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="which key?", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_kind == "needs_input"
        assert t.block_recurrences == 1


def test_unblock_does_not_reset_recurrence_counter(kanban_home: Path) -> None:
    """The crux of the fix: unblock must preserve the loop counter."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="needs_input")
        assert kb.get_task(conn, tid).block_recurrences == 1
        assert kb.unblock_task(conn, tid)
        t = kb.get_task(conn, tid)
        assert t.status == "ready"
        assert t.block_recurrences == 1  # NOT reset to 0
        assert t.block_kind == "needs_input"  # kind preserved for comparison


def test_same_cause_reblock_routes_to_triage(kanban_home: Path) -> None:
    """Dale's loop: block → unblock → re-block same kind → triage."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="need creds", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="still need creds", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "triage"
        assert t.block_recurrences == 2


def test_untyped_block_loop_also_protected(kanban_home: Path) -> None:
    """Legacy un-typed blocks (kind=None) still trip the breaker."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="a")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="a again")
        assert kb.get_task(conn, tid).status == "triage"


def test_different_kinds_do_not_compound(kanban_home: Path) -> None:
    """A re-block for a DIFFERENT reason resets the counter to 1."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="a", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="b", kind="capability")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_recurrences == 1


def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_block_routes_to_todo(kanban_home: Path) -> None:
    """Dependency waits never enter the human 'blocked' bucket."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="need X first", kind="dependency")
        t = kb.get_task(conn, tid)
        assert t.status == "todo"
        assert t.block_kind == "dependency"


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_no_parent_does_not_repromote(kanban_home: Path) -> None:
    """Regression (t_e85f0abe): a dependency-wait with NO parent link must
    PARK in ``todo`` and never auto-re-promote — the <1s
    dependency_wait→promoted→claimed→spawned loop that burned a paid worker
    run per tick.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="waiting on a sibling", kind="dependency")
        assert kb.get_task(conn, tid).status == "todo"
        # Simulate many dispatcher ticks. Without the park guard, the very
        # first recompute_ready would flip it back to 'ready'.
        for _ in range(20):
            kb.recompute_ready(conn)
            assert kb.get_task(conn, tid).status == "todo", (
                "dependency-wait with no parent must not auto-re-promote"
            )
        # No 'promoted' event fired after the dependency_wait.
        evs = kb.list_events(conn, tid)
        dep_idx = max(i for i, e in enumerate(evs) if e.kind == "dependency_wait")
        assert not any(
            e.kind == "promoted" for e in evs[dep_idx + 1:]
        ), "no re-promotion event should follow the dependency_wait"


def test_dependency_already_done_parent_does_not_repromote(
    kanban_home: Path,
) -> None:
    """Regression (t_e85f0abe, mirrors live t_309aaeb8): a dependency-wait
    whose parent was ALREADY done at block time must PARK — the named
    dependency isn't the parent, so nothing genuinely resolved and
    re-promoting just re-runs the worker that declared itself blocked.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # Finish the parent FIRST, then the child declares a dependency wait.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.block_task(conn, child, reason="waiting on something else", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        for _ in range(20):
            kb.recompute_ready(conn)
            assert kb.get_task(conn, child).status == "todo", (
                "dependency-wait with an already-done parent must not "
                "auto-re-promote"
            )


def test_dependency_unblock_recovers_parked_wait(kanban_home: Path) -> None:
    """A parked dependency-wait (no parent) recovers on an explicit
    kanban_unblock — the sanctioned exit from the park state.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="need a sibling artifact", kind="dependency")
        assert kb.get_task(conn, tid).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "todo"
        # Explicit operator unblock. block_task on a 'todo' dependency-wait set
        # status='todo' (not 'blocked'), so drive it via the normal recovery:
        # unblock only acts on blocked/scheduled, so a parked todo recovers by
        # a manual promote_task instead.
        ok, _ = kb.promote_task(conn, tid, actor="operator", force=True)
        assert ok
        assert kb.get_task(conn, tid).status == "ready"


def test_dependency_parent_completes_after_park_promotes(
    kanban_home: Path,
) -> None:
    """A dependency-wait that parks (parent still in flight) DOES resume once
    the parent actually completes after the block — the healthy auto-recover
    path must survive the park guard.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait for parent", kind="dependency")
        # Parent not done yet — child parks and does not promote.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Now finish the parent; the next recompute must promote the child.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_rerun_after_completion_not_parked(kanban_home: Path) -> None:
    """A stale dependency_wait from a PRIOR run must not park a task that has
    since completed and been re-activated. The task's own 'completed' event
    after the dependency_wait supersedes the parked state (there is no
    'unblocked' after a completion), otherwise the revived task parks forever.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        # First life: worker declares a dependency wait (no parent) → parks.
        kb.block_task(conn, tid, reason="wait", kind="dependency")
        assert kb.get_task(conn, tid).status == "todo"
        # Task is then driven to completion (e.g. operator promote + finish).
        ok, _ = kb.promote_task(conn, tid, actor="operator", force=True)
        assert ok
        kb.claim_task(conn, tid, claimer="worker")
        kb.complete_task(conn, tid, result="done")
        assert kb.get_task(conn, tid).status == "done"
        # Second life: re-activate the finished task back into todo and run the
        # gate. The stale dependency_wait must NOT re-park it.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (tid,))
        for _ in range(5):
            kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "ready", (
            "a completed-then-reactivated task must not be re-parked by a "
            "stale dependency_wait from its previous run"
        )


def test_dependency_link_done_parent_recovers(kanban_home: Path) -> None:
    """Graph repair: a parked wait with no parent recovers when an
    ALREADY-done parent is linked AFTER the block via `kanban link`. No new
    'completed' event fires for the finished parent, so the recovery must key
    off the post-wait 'linked' event + the parent's current terminal status.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        # Finish the parent BEFORE it is ever linked.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        # Child declares a dependency wait with NO parent link → parks.
        kb.block_task(conn, child, reason="wait on parent", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Operator repairs the graph by linking the already-done parent.
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready", (
            "linking an already-done parent after the wait must resolve it"
        )


def test_dependency_unlink_all_parents_recovers(kanban_home: Path) -> None:
    """Graph repair (mirror of link): a child that parked with a mistaken
    parent edge recovers when that edge is removed via `kanban unlink` after
    the block — leaving no parents means nothing to wait on, so the park is
    released rather than stuck forever. A wait that NEVER had a parent still
    parks (no post-wait 'unlinked' event).
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        # A mistaken edge is added, then the child declares a dependency wait.
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait on wrong parent", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Operator removes the mistaken edge — recompute must now release it.
        kb.unlink_tasks(conn, parent_id=parent, child_id=child)
        assert kb.get_task(conn, child).status == "ready", (
            "unlinking the last parent after the wait must release the park"
        )


def test_dependency_delete_parent_releases_last_parent_park(
    kanban_home: Path,
) -> None:
    """Hard-deleting the parent must behave like unlinking the last edge.

    delete_task removes task_links internally; if it does not emit the same
    post-wait 'unlinked' child event as kanban unlink, the child looks like a
    never-linked dependency wait and remains parked in todo forever.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait on wrong parent", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"

        assert kb.delete_task(conn, parent)

        assert kb.get_task(conn, child).status == "ready", (
            "deleting the last parent after the wait must release the park"
        )


def test_dependency_idempotent_link_does_not_release_existing_done_parent(
    kanban_home: Path,
) -> None:
    """A no-op re-link must not masquerade as a post-wait graph repair.

    If an already-done parent was linked before the dependency wait, the child
    must park. Re-running the same kanban link is idempotent, so it should not
    emit a fresh post-wait 'linked' event that releases the park.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.block_task(conn, child, reason="waiting on something else", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"

        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.recompute_ready(conn)

        assert kb.get_task(conn, child).status == "todo", (
            "idempotent re-link of an existing done parent must not release "
            "a dependency wait"
        )


def test_dependency_archive_of_already_done_parent_does_not_promote(
    kanban_home: Path,
) -> None:
    """Routine cleanup must not reintroduce the respawn loop.

    A child dependency-waits while its only parent is already ``done``. The
    parent was terminal at block time, so the child must park. Later archiving
    that already-satisfied parent emits a post-wait ``archived`` event, but no
    dependency became *newly* satisfied — the child must STAY parked, not get
    promoted by ``archive_task``'s ``recompute_ready``.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # Parent finishes BEFORE the wait → terminal at block time.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.block_task(conn, child, reason="waiting on something else", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"

        # Routine cleanup archives the already-done parent.
        assert kb.archive_task(conn, parent)

        assert kb.get_task(conn, child).status == "todo", (
            "archiving an already-terminal parent is not a NEW resolution and "
            "must not promote a parked dependency wait"
        )


def test_dependency_parent_completing_after_wait_still_promotes(
    kanban_home: Path,
) -> None:
    """The healthy path stays intact: a parent that reaches a terminal state
    AFTER the wait (a genuine new resolution) must release the park."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        # Child waits while the parent is still in flight.
        kb.block_task(conn, child, reason="wait on parent", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"

        # Parent finishes AFTER the wait → genuine new resolution.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)

        assert kb.get_task(conn, child).status == "ready", (
            "a parent reaching terminal state after the wait must release the park"
        )


def test_dependency_purge_archived_parent_releases_last_parent_park(
    kanban_home: Path,
) -> None:
    """Purging an archived parent must behave like unlinking the last edge.

    ``delete_archived_task`` (the ``kanban archive --rm`` purge path) removes
    ``task_links`` internally; if it does not emit the same post-wait
    ``unlinked`` child event as ``delete_task``/``kanban unlink``, the child
    looks like a never-linked dependency wait and stays parked in ``todo``
    forever.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait on parent", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"

        # Archive then purge the parent (kanban archive --rm).
        assert kb.archive_task(conn, parent)
        assert kb.delete_archived_task(conn, parent)

        assert kb.get_task(conn, child).status == "ready", (
            "purging the last (archived) parent after the wait must release the park"
        )


# ---------------------------------------------------------------------------
# Worker self-block with a rotated run-claim (t_e85f0abe Part B)
# ---------------------------------------------------------------------------


def test_self_block_reconcile_closes_stale_run(kanban_home: Path) -> None:
    """When the run-claim rotated, reconciling the block to the live run must
    also CLOSE the worker's superseded run so it doesn't linger as a phantom
    running/ended_at-NULL attempt.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        live_run = kb.get_task(conn, tid).current_run_id
        assert live_run is not None
        # Rotate the claim: a new run row becomes current, worker holds the old.
        with kb.write_txn(conn):
            cur = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) "
                "VALUES (?, 'worker', 'running', 0)",
                (tid,),
            )
            rotated_run = cur.lastrowid
            conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?",
                (rotated_run, tid),
            )
        assert kb.block_task(
            conn, tid, reason="need creds", kind="needs_input",
            expected_run_id=live_run,
        )
        assert kb.get_task(conn, tid).status == "blocked"
        # The worker's old run must no longer be an open running row.
        old = conn.execute(
            "SELECT status, ended_at FROM task_runs WHERE id = ?",
            (live_run,),
        ).fetchone()
        assert old["ended_at"] is not None, "stale worker run must be closed"
        assert old["status"] != "running", (
            "reconciled stale run must not remain 'running'"
        )
        # No phantom running row with ended_at IS NULL should survive.
        phantom = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs "
            "WHERE task_id = ? AND status = 'running' AND ended_at IS NULL",
            (tid,),
        ).fetchone()
        assert phantom["n"] == 0, "no phantom active run should remain"


def test_self_block_with_stale_expected_run_id(kanban_home: Path) -> None:
    """A live worker whose env run-id is stale (claim rotated) can still
    self-block against a running row instead of failing with
    "not in running/ready".
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        # The row's real current_run_id is the live run.
        t = kb.get_task(conn, tid)
        live_run = t.current_run_id
        assert live_run is not None
        stale_run = live_run + 999  # a run id that isn't the current one
        # Passing a run id that doesn't belong to the task must still fail-safe.
        assert not kb.block_task(
            conn, tid, reason="x", kind="needs_input", expected_run_id=stale_run
        )
        # But an expected_run_id that IS a real (prior) run for this task, while
        # current_run_id has rotated, must reconcile to the live run and block.
        # Simulate rotation: insert a second run row and point current_run_id
        # at it, leaving the worker holding the older (real) run id.
        with kb.write_txn(conn):
            cur = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) "
                "VALUES (?, 'worker', 'running', 0)",
                (tid,),
            )
            rotated_run = cur.lastrowid
            conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?",
                (rotated_run, tid),
            )
        assert kb.block_task(
            conn, tid, reason="need creds", kind="needs_input",
            expected_run_id=live_run,
        ), "worker with a real-but-non-current run id should reconcile + block"
        assert kb.get_task(conn, tid).status == "blocked"


def test_completion_clears_block_memory(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).block_recurrences == 1
        kb.complete_task(conn, tid, result="done")
        t = kb.get_task(conn, tid)
        assert t.status == "done"
        assert t.block_recurrences == 0
        assert t.block_kind is None


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------


def test_invalid_kind_rejected(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError):
            kb.block_task(conn, tid, reason="x", kind="bogus")


def test_block_without_kind_is_backward_compatible(kanban_home: Path) -> None:
    """Existing callers that pass no kind keep the old single-block behaviour."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="legacy")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_kind is None
