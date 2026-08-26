"""A card parked at creation must stay parked.

`--initial-status blocked` is how you file a ticket for later pickup rather
than for immediate work. It set `status='blocked'` but wrote no `blocked`
event, so `_has_sticky_block` saw nothing, `recompute_ready` treated the card
as circuit-breaker debris and promoted it, and the dispatcher spawned a worker
on a ticket nobody had scheduled. Observed live on t_7a8459f5: created at
1787773142, promoted and claimed at 1787773146.
"""
import pytest

from hermes_cli import kanban_db


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    c = kanban_db.connect(tmp_path / "kanban.db")
    yield c
    c.close()


def _kinds(c, task_id):
    return [r["kind"] for r in c.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,)
    ).fetchall()]


def _status(c, task_id):
    return c.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()["status"]


def test_preexisting_initial_status_blocked_stays_parked(conn):
    """Legacy cards have only their original ``created`` event."""
    tid = kanban_db.create_task(
        conn, title="legacy parked ticket", assignee="dev", initial_status="blocked",
    )
    # Simulate a board created before the sticky ``blocked`` event was added.
    conn.execute(
        "DELETE FROM task_events WHERE task_id = ? AND kind = 'blocked'", (tid,)
    )
    conn.commit()

    assert kanban_db._has_sticky_block(conn, tid) is True
    kanban_db.recompute_ready(conn)

    assert _status(conn, tid) == "blocked"


def test_manual_promotion_releases_creation_park(conn):
    """A deliberate manual promotion must clear the legacy marker."""
    tid = kanban_db.create_task(
        conn, title="released ticket", assignee="dev", initial_status="blocked",
    )
    ok, reason = kanban_db.promote_task(conn, tid, actor="operator", force=True)

    assert (ok, reason) == (True, None)
    assert kanban_db._has_sticky_block(conn, tid) is False

    # A later circuit-breaker block has no explicit ``blocked`` event and may
    # recover because the operator already released the creation-time park.
    conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
    conn.commit()
    kanban_db.recompute_ready(conn)
    assert _status(conn, tid) == "ready"


def test_initial_status_blocked_survives_recompute_ready(conn):
    """The whole point of the flag: no lane picks the card up."""
    tid = kanban_db.create_task(
        conn, title="parked ticket", assignee="dev", initial_status="blocked",
    )
    assert _status(conn, tid) == "blocked"

    kanban_db.recompute_ready(conn)

    assert _status(conn, tid) == "blocked", (
        "a card parked at creation was auto-promoted; a worker spawns on it"
    )


def test_initial_status_blocked_records_a_blocked_event(conn):
    """`_has_sticky_block` reads events, so the park must leave one."""
    tid = kanban_db.create_task(
        conn, title="parked ticket", assignee="dev", initial_status="blocked",
    )
    assert "blocked" in _kinds(conn, tid)
    assert kanban_db._has_sticky_block(conn, tid) is True


def test_unblock_still_releases_a_card_parked_at_creation(conn):
    """Sticky must not mean stuck: unblock is the intended exit."""
    tid = kanban_db.create_task(
        conn, title="parked ticket", assignee="dev", initial_status="blocked",
    )
    kanban_db.unblock_task(conn, tid)
    kanban_db.recompute_ready(conn)

    assert _status(conn, tid) == "ready"


def test_a_normal_card_is_unaffected(conn):
    """The control: without it the first test passes on a board that parks
    everything."""
    tid = kanban_db.create_task(conn, title="normal", assignee="dev")
    kanban_db.recompute_ready(conn)

    assert _status(conn, tid) == "ready"
    assert "blocked" not in _kinds(conn, tid)
