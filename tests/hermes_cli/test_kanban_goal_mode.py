"""Tests for kanban goal_mode — per-card Ralph-style goal loop.

Covers three layers:

1. DB: goal_mode / goal_max_turns persist through create_task + from_row,
   and a legacy DB (without the columns) migrates cleanly.
2. Spawn: _default_spawn sets the HERMES_KANBAN_GOAL_MODE env vars only
   when the card opts in.
3. Loop: goals.run_kanban_goal_loop continuation / completion / budget
   behaviour, driven entirely through injected callbacks (no live model).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import goals


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# DB layer
# ---------------------------------------------------------------------------

def test_goal_mode_defaults_off(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain task", assignee="worker")
        task = kb.get_task(conn, tid)
    assert task.goal_mode is False
    assert task.goal_max_turns is None


def test_goal_mode_persists(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="open-ended task",
            assignee="worker",
            goal_mode=True,
            goal_max_turns=7,
        )
        task = kb.get_task(conn, tid)
    assert task.goal_mode is True
    assert task.goal_max_turns == 7


def test_goal_mode_without_max_turns(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="t", assignee="worker", goal_mode=True
        )
        task = kb.get_task(conn, tid)
    assert task.goal_mode is True
    assert task.goal_max_turns is None


def test_legacy_db_migrates_goal_columns(tmp_path, monkeypatch):
    """A tasks table created without goal columns must gain them on init."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal legacy schema: tasks table missing goal_mode / goal_max_turns.
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind) "
        "VALUES ('legacy1', 'old', 'ready', 0, 1, 'scratch')"
    )
    legacy.commit()
    legacy.close()

    # init_db runs the additive migration.
    kb.init_db()
    with kb.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "goal_mode" in cols
        assert "goal_max_turns" in cols
        task = kb.get_task(conn, "legacy1")
    # Existing row keeps the safe default.
    assert task.goal_mode is False
    assert task.goal_max_turns is None


# ---------------------------------------------------------------------------
# Spawn env
# ---------------------------------------------------------------------------

def test_spawn_sets_goal_env_only_when_enabled(kanban_home, monkeypatch):
    captured = {}

    class _FakeProc:
        pid = 4242

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="goal task",
            assignee="default",
            goal_mode=True,
            goal_max_turns=5,
        )
        task = kb.get_task(conn, tid)

    kb._default_spawn(task, str(kanban_home))
    env = captured["env"]
    assert env.get("HERMES_KANBAN_GOAL_MODE") == "1"
    assert env.get("HERMES_KANBAN_GOAL_MAX_TURNS") == "5"


def test_spawn_no_goal_env_for_plain_task(kanban_home, monkeypatch):
    captured = {}

    class _FakeProc:
        pid = 4243

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="default")
        task = kb.get_task(conn, tid)

    kb._default_spawn(task, str(kanban_home))
    env = captured["env"]
    assert "HERMES_KANBAN_GOAL_MODE" not in env
    assert "HERMES_KANBAN_GOAL_MAX_TURNS" not in env


# ---------------------------------------------------------------------------
# Lane goal-mode defaults + worker_max_iterations resolver
# ---------------------------------------------------------------------------

def _db_task(kanban_home, **create_kw):
    """Create a real task row and return its Task view (resolver inputs)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, **create_kw)
        return kb.get_task(conn, tid)


def test_lane_default_enables_goal_mode_for_checkable_lane(kanban_home):
    # `dev` is a default goal-mode lane; no explicit task opt-in needed.
    task = _db_task(kanban_home, title="t", assignee="dev")
    gm, turns = kb._resolve_lane_goal_defaults(task, {})
    assert gm is True
    assert turns == kb.DEFAULT_GOAL_MODE_MAX_TURNS


def test_lane_default_off_for_judgment_lane(kanban_home):
    task = _db_task(kanban_home, title="t", assignee="designer")
    gm, turns = kb._resolve_lane_goal_defaults(task, {})
    assert gm is False
    assert turns is None


def test_explicit_task_goal_mode_wins_for_non_lane(kanban_home):
    # A non-default lane that explicitly asked for goal mode still gets it.
    task = _db_task(kanban_home, title="t", assignee="designer", goal_mode=True)
    gm, turns = kb._resolve_lane_goal_defaults(task, {})
    assert gm is True
    assert turns == kb.DEFAULT_GOAL_MODE_MAX_TURNS


def test_explicit_max_turns_overrides_lane_default(kanban_home):
    task = _db_task(kanban_home, title="t", assignee="dev", goal_mode=True, goal_max_turns=12)
    gm, turns = kb._resolve_lane_goal_defaults(task, {})
    assert gm is True
    assert turns == 12


def test_config_goal_mode_lanes_overrides_builtin(kanban_home):
    cfg = {"goal_mode_lanes": ["pm"], "goal_mode_default_max_turns": 3}
    # `pm` now defaults on...
    pm_task = _db_task(kanban_home, title="t", assignee="pm")
    assert kb._resolve_lane_goal_defaults(pm_task, cfg) == (True, 3)
    # ...and `dev` no longer does (config list replaces the built-in).
    dev_task = _db_task(kanban_home, title="t2", assignee="dev")
    gm2, _ = kb._resolve_lane_goal_defaults(dev_task, cfg)
    assert gm2 is False


def test_config_empty_lane_list_disables_all_defaults(kanban_home):
    task = _db_task(kanban_home, title="t", assignee="dev")
    gm, _ = kb._resolve_lane_goal_defaults(task, {"goal_mode_lanes": []})
    assert gm is False


def test_worker_max_iterations_only_for_non_goal():
    cfg = {"worker_max_iterations": 150}
    assert kb._resolve_worker_max_iterations(False, cfg) == 150
    # Goal-mode dispatches keep the 90 per-run cap regardless of the knob.
    assert kb._resolve_worker_max_iterations(True, cfg) is None


def test_worker_max_iterations_unset_or_invalid():
    assert kb._resolve_worker_max_iterations(False, {}) is None
    assert kb._resolve_worker_max_iterations(False, {"worker_max_iterations": 0}) is None
    assert kb._resolve_worker_max_iterations(False, {"worker_max_iterations": "nope"}) is None


# ---------------------------------------------------------------------------
# Spawn env: lane defaults + worker_max_iterations injection
# ---------------------------------------------------------------------------

def _capture_spawn_env(kanban_home, monkeypatch, task, kanban_cfg):
    captured = {}

    class _FakeProc:
        pid = 5000

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    monkeypatch.setattr(kb, "_load_kanban_cfg", lambda: kanban_cfg)
    kb._default_spawn(task, str(kanban_home))
    return captured["env"]


def test_spawn_lane_default_sets_goal_env(kanban_home, monkeypatch):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ci sweep", assignee="dev")
        task = kb.get_task(conn, tid)
    env = _capture_spawn_env(kanban_home, monkeypatch, task, {})
    assert env.get("HERMES_KANBAN_GOAL_MODE") == "1"
    assert env.get("HERMES_KANBAN_GOAL_MAX_TURNS") == str(kb.DEFAULT_GOAL_MODE_MAX_TURNS)
    # Goal-mode dispatch must NOT also inject HERMES_MAX_ITERATIONS.
    assert "HERMES_MAX_ITERATIONS" not in env


def test_spawn_worker_max_iterations_for_non_goal_lane(kanban_home, monkeypatch):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="design pass", assignee="designer")
        task = kb.get_task(conn, tid)
    env = _capture_spawn_env(
        kanban_home, monkeypatch, task, {"worker_max_iterations": 150}
    )
    assert env.get("HERMES_MAX_ITERATIONS") == "150"
    assert "HERMES_KANBAN_GOAL_MODE" not in env


def test_spawn_goal_lane_ignores_worker_max_iterations(kanban_home, monkeypatch):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ci sweep", assignee="dev")
        task = kb.get_task(conn, tid)
    env = _capture_spawn_env(
        kanban_home, monkeypatch, task, {"worker_max_iterations": 150}
    )
    assert env.get("HERMES_KANBAN_GOAL_MODE") == "1"
    assert "HERMES_MAX_ITERATIONS" not in env


def test_spawn_goal_lane_drops_inherited_max_iterations(kanban_home, monkeypatch):
    # A HERMES_MAX_ITERATIONS leaking from the dispatcher's own env must NOT
    # raise a goal-mode worker's per-run cap — goal mode keeps the 90 cap.
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "200")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ci sweep", assignee="dev")
        task = kb.get_task(conn, tid)
    env = _capture_spawn_env(kanban_home, monkeypatch, task, {})
    assert env.get("HERMES_KANBAN_GOAL_MODE") == "1"
    assert "HERMES_MAX_ITERATIONS" not in env


def test_spawn_non_goal_drops_inherited_max_iterations_when_unset(kanban_home, monkeypatch):
    # No config knob → a non-goal worker's per-run cap comes from config/profile,
    # not an ambient HERMES_MAX_ITERATIONS inherited from the dispatcher.
    monkeypatch.setenv("HERMES_MAX_ITERATIONS", "200")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="design pass", assignee="designer")
        task = kb.get_task(conn, tid)
    env = _capture_spawn_env(kanban_home, monkeypatch, task, {})
    assert "HERMES_KANBAN_GOAL_MODE" not in env
    assert "HERMES_MAX_ITERATIONS" not in env


# ---------------------------------------------------------------------------
# Goal loop logic (callback-injected, no live model)
# ---------------------------------------------------------------------------

def _patch_judge(monkeypatch, verdicts):
    """Make judge_goal return a scripted sequence of verdicts."""
    seq = list(verdicts)

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        v = seq.pop(0) if seq else "done"
        # 4-tuple contract: (verdict, reason, parse_failed, wait_directive)
        return v, f"scripted:{v}", False, None

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)


def test_loop_stops_when_worker_already_completed(monkeypatch):
    # Worker called kanban_complete on its first turn — no judging needed.
    _patch_judge(monkeypatch, ["continue"])  # should never be consulted
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "x",
        task_status_fn=lambda: "done",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="done already",
    )
    assert res["outcome"] == "completed_by_worker"
    assert turns == []  # no extra turns


def test_loop_continues_then_worker_completes(monkeypatch):
    _patch_judge(monkeypatch, ["continue", "continue"])
    statuses = iter(["running", "running", "done"])
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t2",
        goal_text="ship feature",
        run_turn=lambda p: turns.append(p) or f"turn{len(turns)}",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="started",
    )
    assert res["outcome"] == "completed_by_worker"
    # Two continuation turns fed before the worker completed.
    assert len(turns) == 2
    assert all("not done yet" in p for p in turns)


def test_loop_blocks_on_budget_exhaustion(monkeypatch):
    _patch_judge(monkeypatch, ["continue"] * 10)
    blocked = {}

    def _block(reason):
        blocked["reason"] = reason

    res = goals.run_kanban_goal_loop(
        task_id="t3",
        goal_text="endless task",
        run_turn=lambda p: "still going",
        task_status_fn=lambda: "running",
        block_fn=_block,
        max_turns=3,
        first_response="turn1",
    )
    assert res["outcome"] == "blocked_budget"
    assert res["turns_used"] == 3
    assert "turn budget" in blocked["reason"].lower()


def test_loop_finalize_nudge_when_judge_done_but_open(monkeypatch):
    # Judge says done, but worker never terminated → one finalize nudge,
    # then worker completes.
    _patch_judge(monkeypatch, ["done", "done"])
    statuses = iter(["running", "done"])
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t4",
        goal_text="task",
        run_turn=lambda p: turns.append(p) or "ok",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="looks done",
    )
    assert res["outcome"] == "completed_by_worker"
    assert len(turns) == 1
    assert "still open" in turns[0]


def test_loop_blocks_when_judge_done_but_never_finalizes(monkeypatch):
    # Judge keeps saying done, worker never calls kanban_complete → block
    # after the single finalize nudge.
    _patch_judge(monkeypatch, ["done", "done"])
    blocked = {}

    res = goals.run_kanban_goal_loop(
        task_id="t5",
        goal_text="task",
        run_turn=lambda p: "still not finalizing",
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=10,
        first_response="looks done",
    )
    assert res["outcome"] == "blocked_budget"
    assert "finalize" in blocked["reason"].lower()


def test_loop_stops_if_task_reclaimed(monkeypatch):
    _patch_judge(monkeypatch, ["continue"])
    res = goals.run_kanban_goal_loop(
        task_id="t6",
        goal_text="task",
        run_turn=lambda p: pytest.fail("should not run a turn"),
        task_status_fn=lambda: "archived",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="x",
    )
    assert res["outcome"] == "stopped"
