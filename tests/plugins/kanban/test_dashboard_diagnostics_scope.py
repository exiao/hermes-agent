"""Tests for the kanban dashboard board-load diagnostics scoping.

``plugin_api._compute_task_diagnostics(conn, task_ids=None)`` runs on every
board load. The board-load path must NOT scan the full ``done`` column's event
history (the hotspot: hundreds of finished cards, ~15K-and-growing events, zero
badges). It scopes to:

  (a) every non-terminal card, PLUS
  (b) the small set of terminal (done/archived) cards that carry an active
      ``_WARNING_EVENT_KINDS`` event — because ``hallucinated_cards`` and the
      ``prose_phantom_refs`` advisory legitimately fire on those.

A plain ``done`` card with no warning event must be excluded. The explicit
``task_ids=[...]`` path (drawer open) must stay unscoped.

These tests build a real on-disk board DB via ``kanban_db`` (sqlite3.Row
factory, matching what the dashboard hands the rule engine).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import pytest  # type: ignore[unresolved-import]  # dev-only dep; ty env lacks it

from hermes_cli import kanban_db as kb


def _load_plugin_module():
    """Dynamically load plugins/kanban/dashboard/plugin_api.py as a module.

    Mirrors the loader in test_kanban_dashboard_plugin.py — the dashboard
    plugin dir is not an importable package, so we load it by file path.
    """
    repo_root = Path(__file__).resolve().parents[3]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_diag_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


plugin_api = _load_plugin_module()


@pytest.fixture
def board_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path=db_path)
    return conn


def _set_status(conn, task_id, status):
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
    conn.commit()


def _emit_event(conn, task_id, kind, payload, *, after=False):
    """Insert a task_events row. ``after`` bumps created_at so it sorts last
    (the rule engine treats a warning event as active only when no clean
    completed/edited event follows it)."""
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, NULL, ?, ?, ?)",
        (task_id, kind, json.dumps(payload), int(time.time()) + (10 if after else 0)),
    )
    conn.commit()


def test_board_load_skips_plain_done_card_but_keeps_live(board_db):
    conn = board_db
    # Live blocked card with an active hallucination signal — must be flagged.
    blocked = kb.create_task(conn, title="blocked", assignee="x",
                             initial_status="blocked")
    _emit_event(conn, blocked, "completion_blocked_hallucination",
                {"phantom_cards": ["t_ghost1"]})

    # A plain done card with NO warning event — must NOT be scanned/flagged.
    done_plain = kb.create_task(conn, title="done plain", assignee="x")
    _set_status(conn, done_plain, "done")

    diags = plugin_api._compute_task_diagnostics(conn, task_ids=None)

    assert blocked in diags, "live blocked card lost its diagnostic badge"
    assert done_plain not in diags, "plain done card was flagged on board load"


def test_board_load_keeps_done_card_with_phantom_ref_advisory(board_db):
    """A ``done`` card whose completion summary referenced an unresolved id
    carries an active ``suspected_hallucinated_references`` event AFTER its
    ``completed`` event. ``prose_phantom_refs`` fires on it, and the board-load
    pass must still surface that advisory (don't regress the feature)."""
    conn = board_db
    done = kb.create_task(conn, title="done w/ phantom ref", assignee="x")
    _set_status(conn, done, "done")
    # completed first, then the advisory after it (so it stays active).
    _emit_event(conn, done, "completed", {})
    _emit_event(conn, done, "suspected_hallucinated_references",
                {"phantom_refs": ["t_deadbeef1234"]}, after=True)

    diags = plugin_api._compute_task_diagnostics(conn, task_ids=None)

    assert done in diags, "done card lost its prose_phantom_refs advisory"
    kinds = {d["kind"] for d in diags[done]}
    assert "prose_phantom_refs" in kinds


def test_board_load_excludes_archived_without_signal(board_db):
    conn = board_db
    archived = kb.create_task(conn, title="archived", assignee="x")
    _set_status(conn, archived, "archived")
    diags = plugin_api._compute_task_diagnostics(conn, task_ids=None)
    assert archived not in diags


def test_drawer_path_unscoped(board_db):
    """The explicit task_ids path (drawer open) still computes whatever id it's
    handed, including a plain done card — it does not apply the board-load
    status/event scoping."""
    conn = board_db
    done = kb.create_task(conn, title="done", assignee="x")
    _set_status(conn, done, "done")
    result = plugin_api._compute_task_diagnostics(conn, task_ids=[done])
    assert isinstance(result, dict)


def test_board_load_excludes_archived_with_warning_event(board_db):
    """Archived cards stay out of the default (task_ids=None) diagnostics even
    when they carry an active warning event. The prior fleet query used
    ``status != 'archived'``; archived is hidden from the default board view, so
    a stale phantom-ref advisory on an archived card must not keep the default
    diagnostics non-empty. (A ``done`` card with the same event IS re-included —
    see test_board_load_keeps_done_card_with_phantom_ref_advisory.)"""
    conn = board_db
    archived = kb.create_task(conn, title="archived w/ advisory", assignee="x")
    _set_status(conn, archived, "archived")
    _emit_event(conn, archived, "completed", {})
    _emit_event(conn, archived, "suspected_hallucinated_references",
                {"phantom_refs": ["t_deadbeef1234"]}, after=True)

    diags = plugin_api._compute_task_diagnostics(conn, task_ids=None)

    assert archived not in diags, "archived card surfaced in default diagnostics"


def _scanned_task_ids(conn):
    """Return the set of task ids whose event/run history the board-load pass
    materialises. Uses sqlite3's ``set_trace_callback`` to capture executed
    statements, then reads the ids bound into the ``task_events WHERE task_id
    IN (...)`` scan — those are exactly the candidate ``row_ids`` the pass
    materialises, i.e. the hotspot the scoping is meant to bound.
    """
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        plugin_api._compute_task_diagnostics(conn, task_ids=None)
    finally:
        conn.set_trace_callback(None)

    captured: set[str] = set()
    for stmt in statements:
        if "FROM task_events WHERE task_id IN" in stmt:
            # sqlite3 expands bound params into the traced statement text;
            # the ids are the quoted 't_<hex>' literals in the IN clause.
            captured.update(re.findall(r"'(t_[0-9a-f]+)'", stmt))
    return captured


def _emit_event_at(conn, task_id, kind, payload, ts):
    """Insert a task_events row with an explicit ``created_at``.

    Used to build a deterministic blocked→unblocked cycle sequence inside the
    ``block_unblock_cycling`` rule's sliding window (the rule counts
    ``blocked`` events that follow an ``unblocked`` event, chronological by id).
    """
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, NULL, ?, ?, ?)",
        (task_id, kind, json.dumps(payload), ts),
    )
    conn.commit()


def test_board_load_keeps_done_card_with_block_cycle_diagnostic(board_db):
    """A ``done`` card that cycled blocked→unblocked enough times to trigger
    ``block_unblock_cycling`` and then completed *without* any
    hallucination-warning event must still surface on the board-load pass.

    ``_rule_block_unblock_cycling`` has no current-status gate (it counts
    recent ``blocked``/``unblocked`` events regardless of the card's current
    status), so the drawer path (``task_ids=[id]``) still shows the diagnostic
    on a completed card. The board-load candidate query must re-include such a
    done card, otherwise ``/board`` badges and ``/diagnostics`` disagree for
    the same task. Re-inclusion keys off ``blocked``/``unblocked`` events (NOT
    ``_WARNING_EVENT_KINDS``), so the done-history scan stays bounded to the
    small set of done cards with a recent block-cycle signal.
    """
    conn = board_db
    now = int(time.time())
    cycled = kb.create_task(conn, title="done, cycled", assignee="x")
    _set_status(conn, cycled, "done")
    # 3 blocked-after-unblocked cycles within the 24h window (default
    # threshold). Sequence by ascending id (insertion order): blocked,
    # unblocked, blocked (cycle 1), unblocked, blocked (cycle 2), unblocked,
    # blocked (cycle 3). Then a clean completed event — no hallucination
    # warning, so the ``_WARNING_EVENT_KINDS`` branch does NOT catch this card.
    seq = [
        ("blocked", now - 700),
        ("unblocked", now - 690),
        ("blocked", now - 680),   # cycle 1
        ("unblocked", now - 670),
        ("blocked", now - 660),   # cycle 2
        ("unblocked", now - 650),
        ("blocked", now - 640),   # cycle 3
        ("completed", now - 600),
    ]
    for kind, ts in seq:
        _emit_event_at(conn, cycled, kind, {}, ts)

    # Control: a plain done card with no block-cycle and no warning event —
    # must stay excluded from the board-load candidate scan.
    plain = kb.create_task(conn, title="done plain", assignee="x")
    _set_status(conn, plain, "done")

    # Drawer path confirms the rule engine still fires on the done card.
    drawer = plugin_api._compute_task_diagnostics(conn, task_ids=[cycled])
    assert cycled in drawer, "drawer path lost the block_unblock_cycling badge"
    assert "block_unblock_cycling" in {d["kind"] for d in drawer[cycled]}

    # Board-load path must agree: the cycled done card stays in scope.
    diags = plugin_api._compute_task_diagnostics(conn, task_ids=None)
    assert cycled in diags, (
        "done card with active block_unblock_cycling was dropped by the "
        "board-load candidate query — /board and /diagnostics disagree"
    )
    assert "block_unblock_cycling" in {d["kind"] for d in diags[cycled]}
    assert plain not in diags, "plain done card was flagged on board load"

    # The cycled card's history must be materialised (it's a candidate); the
    # plain done card's history must NOT be materialised.
    scanned = _scanned_task_ids(conn)
    assert cycled in scanned, "cycled done card was dropped from the scan"
    assert plain not in scanned, "plain done card's history was materialised"


def test_board_load_skips_done_card_whose_block_cycle_aged_out(board_db):
    """A ``done`` card whose block-cycle events all fell outside the sliding
    window (so ``_rule_block_unblock_cycling`` no longer fires) must NOT be
    re-included by the block-cycle branch — the board-load pass must not
    re-materialise a done card's history for a diagnostic that can't fire.
    """
    conn = board_db
    now = int(time.time())
    # 3 cycles, but all ~48h ago — outside the default 24h window.
    aged = kb.create_task(conn, title="done, aged cycles", assignee="x")
    _set_status(conn, aged, "done")
    seq = [
        ("blocked", now - 48 * 3600 - 700),
        ("unblocked", now - 48 * 3600 - 690),
        ("blocked", now - 48 * 3600 - 680),
        ("unblocked", now - 48 * 3600 - 670),
        ("blocked", now - 48 * 3600 - 660),
        ("unblocked", now - 48 * 3600 - 650),
        ("blocked", now - 48 * 3600 - 640),
        ("completed", now - 48 * 3600 - 600),
    ]
    for kind, ts in seq:
        _emit_event_at(conn, aged, kind, {}, ts)

    diags = plugin_api._compute_task_diagnostics(conn, task_ids=None)
    assert aged not in diags, (
        "done card with aged-out block cycles was flagged on board load"
    )
    scanned = _scanned_task_ids(conn)
    assert aged not in scanned, (
        "done card with aged-out block cycles had its history materialised"
    )


def test_board_load_skips_done_card_whose_warning_was_cleared(board_db):
    """A ``done`` card that once hit ``completion_blocked_hallucination`` but was
    then completed/edited (clearing the warning, per the rule engine's
    ``_active_hallucination_events``) yields no badge. Re-including it would just
    re-materialise its event/run history on every board load for nothing, so the
    pass must NOT pull it into the candidate scan — it stays excluded just like a
    plain done card.
    """
    conn = board_db
    # Done card: blocked-completion warning, THEN a clean completed event after
    # it (greater id) — the warning is no longer active.
    cleared = kb.create_task(conn, title="done, warning cleared", assignee="x")
    _set_status(conn, cleared, "done")
    _emit_event(conn, cleared, "completion_blocked_hallucination",
                {"phantom_cards": ["t_ghost1"]})
    _emit_event(conn, cleared, "completed", {}, after=True)

    # Control: a done card whose advisory is still active (completed first, then
    # the warning after) — must stay in scope.
    active = kb.create_task(conn, title="done, warning active", assignee="x")
    _set_status(conn, active, "done")
    _emit_event(conn, active, "completed", {})
    _emit_event(conn, active, "suspected_hallucinated_references",
                {"phantom_refs": ["t_deadbeef1234"]}, after=True)

    diags = plugin_api._compute_task_diagnostics(conn, task_ids=None)
    assert cleared not in diags, "cleared done card was flagged on board load"
    assert active in diags, "active-warning done card lost its advisory"

    scanned = _scanned_task_ids(conn)
    assert cleared not in scanned, (
        "cleared done card's history was materialised on board load"
    )
    assert active in scanned, "active-warning done card was dropped from the scan"
