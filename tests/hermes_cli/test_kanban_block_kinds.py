"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking —
  unless no parent is open, in which case the wait can never be satisfied
  and the block is recorded as ``needs_input`` (sticky, loop-counted).
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from plugins.kanban.dashboard.plugin_api import _set_status_direct
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t", parents=()):
    """Create a task (linked under ``parents`` first) and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    for parent in parents:
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
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


def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
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


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn,
            parent_id=parent,
            child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_parent_completes_after_park_promotes(
    kanban_home: Path,
) -> None:
    """A dependency-wait that parks (parent still in flight) DOES resume once
    the parent actually completes after the block — the healthy auto-recover
    path must survive the park guard.
    """
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn, parent_id=parent, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
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


def test_dependency_unlink_all_parents_recovers(kanban_home: Path) -> None:
    """Graph repair (mirror of link): a child that parked with a mistaken
    parent edge recovers when that edge is removed via `kanban unlink` after
    the block — leaving no parents means nothing to wait on, so the park is
    released rather than stuck forever. A wait that NEVER had a parent still
    parks (no post-wait 'unlinked' event).
    """
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        # A mistaken edge is added, then the child declares a dependency wait.
        kb.link_tasks(
            conn, parent_id=parent, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
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
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn, parent_id=parent, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.block_task(conn, child, reason="wait on wrong parent", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"

        assert kb.delete_task(conn, parent)

        assert kb.get_task(conn, child).status == "ready", (
            "deleting the last parent after the wait must release the park"
        )


def test_dependency_parent_completing_after_wait_still_promotes(
    kanban_home: Path,
) -> None:
    """The healthy path stays intact: a parent that reaches a terminal state
    AFTER the wait (a genuine new resolution) must release the park."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn, parent_id=parent, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
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


def test_dependency_reopened_parent_recompletion_after_wait_promotes(
    kanban_home: Path,
) -> None:
    """A parent that completed, was REOPENED, then completed again after the
    wait must release the park.

    The already-terminal check must key off the parent's lifecycle relative to
    its last reopen — not any historical terminal event. Otherwise the old
    pre-reopen 'completed' event marks the parent already-terminal and its
    genuine post-wait recompletion is ignored, stranding the child in todo.
    """
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn, parent_id=parent, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        # Parent completes once.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        # Parent is reopened (dashboard drag done -> ready).
        assert _set_status_direct(conn, parent, "ready")
        # Child dependency-blocks while the reopened parent is in flight.
        kb.block_task(conn, child, reason="wait on reopened parent", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Parent completes AGAIN, after the wait → genuine new resolution.
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done again")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready", (
            "a reopened parent's recompletion after the wait must release the park"
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
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn, parent_id=parent, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
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


def test_dependency_partial_unlink_of_unresolved_parent_releases_park(
    kanban_home: Path,
) -> None:
    """Removing only the unresolved edge after the wait, leaving an
    already-terminal parent, must release the park.

    Child waits on parent A (already done) + parent B (in flight). An operator
    unlinks the mistaken in-flight B. All remaining parents (just A) are now
    terminal, so the dependency graph is satisfied and the child must promote
    rather than stay stranded in todo.
    """
    with kbc.connect_closing() as conn:
        parent_a = kb.create_task(conn, title="parent-A-done", assignee="worker")
        parent_b = kb.create_task(conn, title="parent-B-inflight", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn, parent_id=parent_a, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.link_tasks(
            conn, parent_id=parent_b, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        # A finishes before the wait; B stays in flight.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent_a,))
        kb.claim_task(conn, parent_a, claimer="worker")
        kb.complete_task(conn, parent_a, result="done")
        kb.block_task(conn, child, reason="wait on A and B", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Operator unlinks the mistaken in-flight parent B.
        kb.unlink_tasks(conn, parent_id=parent_b, child_id=child)
        assert kb.get_task(conn, child).status == "ready", (
            "unlinking the only unresolved parent while an already-done parent "
            "remains must release the park"
        )


def test_dependency_partial_unlink_leaving_inflight_parent_still_parks(
    kanban_home: Path,
) -> None:
    """Unlinking one parent while another is STILL in flight must keep parking.

    Guards against the partial-unlink release firing when an unresolved parent
    remains — the child must wait for that parent, not promote early.
    """
    with kbc.connect_closing() as conn:
        parent_a = kb.create_task(conn, title="parent-A", assignee="worker")
        parent_b = kb.create_task(conn, title="parent-B", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn, parent_id=parent_a, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.link_tasks(
            conn, parent_id=parent_b, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.block_task(conn, child, reason="wait on A and B", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Unlink A, but B is still in flight → must stay parked.
        kb.unlink_tasks(conn, parent_id=parent_a, child_id=child)
        assert kb.get_task(conn, child).status == "todo", (
            "an unresolved in-flight parent still gates the child after a "
            "partial unlink"
        )


def test_dependency_parent_marked_done_via_status_after_wait_promotes(
    kanban_home: Path,
) -> None:
    """A parent driven terminal via set_status_direct(parent, 'done') AFTER the
    wait must release the park.

    set_status_direct accepts 'done' and calls recompute_ready, so this manual
    dashboard drag-to-done is a genuine completion path — the child must promote
    rather than stay stranded in todo.
    """
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn, parent_id=parent, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        # Child waits while the parent is still in flight (todo/ready).
        kb.block_task(conn, child, reason="wait on parent", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Parent is dragged straight to 'done' on the board after the wait.
        assert _set_status_direct(conn, parent, "done")
        assert kb.get_task(conn, child).status == "ready", (
            "a parent marked done via set_status_direct after the wait is a "
            "genuine new resolution and must release the park"
        )


def test_dependency_block_with_terminal_parents_parks_then_escalates(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``dependency`` block whose parents are all terminal can never be
    satisfied by ``recompute_ready``: it must park in ``blocked`` as
    ``needs_input`` (no ``dependency_wait``, no re-promotion), say so on the
    CLI, and count toward the loop breaker so a re-block after an unblock
    reaches ``triage``."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="already-done-parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        # The edge exists before the run: link_tasks refuses to gate a running child retroactively.
        child = _running_task(conn, title="child-of-done", parents=(parent,))

        # `hermes kanban block <child> --kind dependency waiting on upstream`
        args = argparse.Namespace(task_id=child, ids=None, reason=["waiting", "on", "upstream"], kind="dependency")
        assert kanban_cli._cmd_block(args) == 0
        assert "needs_input" in capsys.readouterr().out
        parked = kb.get_task(conn, child)
        assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", "needs_input", 1)
        events = kb.list_events(conn, child)
        assert not [e for e in events if e.kind == "dependency_wait"]
        blocked = [e for e in events if e.kind == "blocked"][-1].payload
        assert (blocked["requested_kind"], blocked["rekind_reason"]) == ("dependency", "no_open_parent")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"

        # A cron/human unblocks; the worker re-declares the same impossible wait.
        assert kb.unblock_task(conn, child)
        assert kb.claim_task(conn, child, claimer="worker") is not None
        assert kb.block_task(conn, child, reason="still waiting", kind="dependency")
        assert kb.get_task(conn, child).status == "triage"
        loop = [e for e in kb.list_events(conn, child) if e.kind == "block_loop_detected"][-1].payload
        assert loop["recurrences"] == kb.BLOCK_RECURRENCE_LIMIT


def test_dependency_block_with_open_parent_stays_parked_across_dispatch_tick(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Control: a genuine wait on an incomplete parent parks in ``todo``
    without counting a recurrence, survives a dispatch tick unspawned, and
    resumes once the parent finishes."""
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="open-parent", assignee="alice")
        # Linked while todo (a running child cannot be gated retroactively); the parent then stays
        # open while the child runs — the reopened-parent shape, forced the same way the loop below does.
        child = kb.create_task(conn, title="waiter", assignee="worker")
        kb.link_tasks(
            conn, parent_id=parent, child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 1):
            assert kb.block_task(conn, child, reason="wait", kind="dependency")
            parked = kb.get_task(conn, child)
            assert (parked.status, parked.block_kind, parked.block_recurrences) == ("todo", "dependency", 0)
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        assert kb.block_task(conn, child, reason="wait", kind="dependency")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert kb.get_task(conn, child).status == "todo"
        assert child not in spawns
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="alice")
        kb.complete_task(conn, parent, result="done")
        res2 = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert child in [row[0] for row in res2.spawned]


# ---------------------------------------------------------------------------
# Worker self-block with a rotated run-claim (t_e85f0abe Part B)
# ---------------------------------------------------------------------------


def test_self_block_reconcile_closes_stale_run(kanban_home: Path) -> None:
    """When the run-claim rotated, reconciling the block to the live run must
    also CLOSE the worker's superseded run so it doesn't linger as a phantom
    running/ended_at-NULL attempt.
    """
    with kbc.connect_closing() as conn:
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
    with kbc.connect_closing() as conn:
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
    with kbc.connect_closing() as conn:
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
