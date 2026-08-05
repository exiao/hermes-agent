"""Re-persist dedup in ``append_messages_batch`` must ignore ``active``.

``active`` is mutable lifecycle state, not identity:

* compaction   -> ``active=0, compacted=1``
* rewind/undo  -> ``active=0, compacted=0``
* undo restore -> flips ``active`` back to 1 **by id** (never re-inserts)

The batch guard used to match on ``AND active = 1``, so any re-persist that
arrived after the original row was deactivated missed its own original and
inserted a fresh copy — once per rotation flush. Long-lived sessions
accumulated thousands of byte-identical rows (same session, role, content and
timestamp), and the FTS trigram index multiplied the cost.

``append_message`` (single-row) always omitted ``active``; this locks the batch
path to the same rule.
"""

from hermes_state import SessionDB


def _rows(db, session_id):
    with db._read_ctx() as conn:
        return conn.execute(
            "SELECT role, content, timestamp, active FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()


def _deactivate(db, session_id):
    """Simulate compaction: active=0, compacted=1."""
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET active = 0, compacted = 1 WHERE session_id = ?",
            (session_id,),
        )
    )


def test_batch_repersist_after_deactivation_does_not_duplicate(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-batch-dedup"
    db.create_session(session_id=session_id, source="test")

    # Fixed timestamps: a re-persist carries the ORIGINAL timestamp, which is
    # what makes it distinguishable from a genuinely new message.
    messages = [
        {"role": "user", "content": "first", "timestamp": 1000.0},
        {"role": "assistant", "content": "second", "timestamp": 1001.0},
    ]

    db.append_messages_batch(session_id=session_id, messages=messages)
    assert len(_rows(db, session_id)) == 2

    # Re-persisting the same history is already a no-op while rows are live.
    db.append_messages_batch(session_id=session_id, messages=messages)
    assert len(_rows(db, session_id)) == 2, "guard must dedupe live rows"

    # Compaction deactivates the transcript.
    _deactivate(db, session_id)

    # The rotation flush re-persists the same reloaded history. Before the fix
    # the guard's `AND active = 1` missed the deactivated originals and this
    # inserted two more rows (and would again on every later flush).
    db.append_messages_batch(session_id=session_id, messages=messages)

    rows = _rows(db, session_id)
    assert len(rows) == 2, (
        f"re-persist after compaction duplicated rows: {len(rows)} != 2 -> {rows}"
    )
    # The originals stay deactivated — dedup must not resurrect compacted
    # history into the live context window.
    assert all(r[3] == 0 for r in rows), f"compacted rows were reactivated: {rows}"


def test_repeated_compaction_cycles_stay_flat(tmp_path):
    """Growth is one duplicate per DEACTIVATION, not per flush.

    Once the old code inserted a duplicate it was ``active=1``, so later
    flushes in the same cycle matched that copy and stopped. The next
    compaction deactivated it too, and the cycle repeated — which is how a
    long-lived session reaches thousands of copies of one message.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-batch-dedup-repeat"
    db.create_session(session_id=session_id, source="test")

    messages = [{"role": "user", "content": "hello", "timestamp": 2000.0}]
    db.append_messages_batch(session_id=session_id, messages=messages)

    # Five compaction cycles, each followed by rotation flushes.
    for _ in range(5):
        _deactivate(db, session_id)
        db.append_messages_batch(session_id=session_id, messages=messages)
        db.append_messages_batch(session_id=session_id, messages=messages)

    assert len(_rows(db, session_id)) == 1


def test_distinct_message_after_compaction_still_inserts(tmp_path):
    """Dropping the active filter must not suppress genuinely new messages."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "sess-batch-dedup-new"
    db.create_session(session_id=session_id, source="test")

    db.append_messages_batch(
        session_id=session_id,
        messages=[{"role": "user", "content": "old", "timestamp": 3000.0}],
    )
    _deactivate(db, session_id)

    # New message: different timestamp and content -> must be inserted live.
    db.append_messages_batch(
        session_id=session_id,
        messages=[{"role": "user", "content": "new", "timestamp": 3001.0}],
    )

    rows = _rows(db, session_id)
    assert len(rows) == 2, f"new message was wrongly deduped: {rows}"
    assert rows[1][1] == "new"
    assert rows[1][3] == 1, "new message must be active"
