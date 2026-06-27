"""Tests for create-time assignee validation (issue: silent dispatcher drop).

``kanban_create`` (both the model tool and the ``hermes kanban create`` CLI)
now rejects an assignee that names a lane which doesn't exist, instead of
accepting a card the dispatcher will then silently never spawn. The check is a
single shared validator, ``kanban_db.validate_assignee``, so the two surfaces
can't drift.

Covers:
  - validate_assignee() directly: real profile / unknown / sentinels / default.
  - tool _handle_create: real assignee OK, unknown rejected (no row created),
    'none' sentinel OK.
  - CLI _cmd_create / run_slash: unknown --assignee fails loud (no row created).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


def _seed_profiles(home: Path, *names: str) -> None:
    for name in names:
        pdir = home / "profiles" / name
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "config.yaml").write_text("model: {}\n")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with two real profiles on disk (researcher,
    writer) plus the implicit ``default``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    _seed_profiles(home, "researcher", "writer")
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# validate_assignee() unit cases
# ---------------------------------------------------------------------------

def test_validate_assignee_accepts_real_profile(kanban_home):
    assert kb.validate_assignee("researcher") is None
    assert kb.validate_assignee("writer") is None


def test_validate_assignee_accepts_default(kanban_home):
    # `default` is always a valid profile when the default root exists.
    assert kb.validate_assignee("default") is None


def test_validate_assignee_normalizes_case(kanban_home):
    # create_task canonicalizes via normalize_profile_name; the validator must
    # accept the same case-folded form so a title-cased label still passes.
    assert kb.validate_assignee("Researcher") is None


def test_validate_assignee_rejects_unknown(kanban_home):
    err = kb.validate_assignee("equity-analyzer")
    assert err is not None
    assert "equity-analyzer" in err
    # Lists the valid profiles so the model can self-correct.
    assert "researcher" in err and "writer" in err


@pytest.mark.parametrize("sentinel", [None, "", "  ", "none", "None", "-", "null"])
def test_validate_assignee_accepts_unassigned_sentinels(kanban_home, sentinel):
    # A deliberately-unassigned / triage card is valid — don't over-tighten.
    assert kb.validate_assignee(sentinel) is None


# ---------------------------------------------------------------------------
# Tool surface: _handle_create
# ---------------------------------------------------------------------------

def _board_task_count() -> int:
    conn = kb.connect()
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()
        return int(row["n"])
    finally:
        conn.close()


def test_tool_create_real_assignee_succeeds(kanban_home):
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "do work", "assignee": "researcher"}))
    assert out["ok"] is True
    assert out["task_id"]


def test_tool_create_unknown_assignee_rejected_no_row(kanban_home):
    from tools import kanban_tools as kt

    before = _board_task_count()
    out = json.loads(kt._handle_create({"title": "bad route", "assignee": "equity-analyzer"}))
    assert "error" in out
    assert "equity-analyzer" in out["error"]
    # The board must be unchanged: no row was created for the bad assignee.
    assert _board_task_count() == before


def test_tool_create_unassigned_sentinel_succeeds(kanban_home):
    """An explicit 'none' assignee is a recognized unassign sentinel and must
    still pass the validator — guards against over-tightening. (The validator
    deliberately only gates 'names a lane that doesn't exist'; it does not
    coerce how the sentinel is stored — that's out of scope.)"""
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "park it", "assignee": "none"}))
    assert out["ok"] is True
    assert out["task_id"]


# ---------------------------------------------------------------------------
# CLI surface: hermes kanban create
# ---------------------------------------------------------------------------

def test_cli_create_real_assignee_succeeds(kanban_home):
    out = kc.run_slash("create 'cli ok' --assignee writer")
    assert "Created" in out


def test_cli_create_unknown_assignee_rejected_no_row(kanban_home):
    before = _board_task_count()
    out = kc.run_slash("create 'cli bad' --assignee equity-analyzer")
    assert "Created" not in out
    assert "equity-analyzer" in out
    assert "not a known profile" in out
    # No row created for the bad assignee.
    assert _board_task_count() == before
