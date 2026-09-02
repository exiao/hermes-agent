"""Tests for the Kanban DB layer (hermes_cli.kanban_db)."""

from __future__ import annotations

import concurrent.futures
import os
import sqlite3
import subprocess
import sys
import time
import types
import unittest.mock
from pathlib import Path

import pytest

import hermes_state
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kanban@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kanban Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Schema / init
# ---------------------------------------------------------------------------





def test_terminal_window_index_matches_board_order(kanban_home):
    """The board's done/archived window should not sort terminal history with a temp B-tree."""

    with kb.connect() as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT * FROM tasks WHERE status = ? "
            "ORDER BY (CASE WHEN status = 'archived' "
            "THEN COALESCE(archived_at, completed_at) "
            "ELSE completed_at END IS NULL), "
            "CASE WHEN status = 'archived' "
            "THEN COALESCE(archived_at, completed_at) "
            "ELSE completed_at END DESC, created_at DESC, id DESC "
            "LIMIT ?",
            ("done", 50),
        ).fetchall()

    details = [row["detail"] for row in plan]
    assert any("idx_tasks_terminal_window" in detail for detail in details), details
    assert not any("USE TEMP B-TREE" in detail.upper() for detail in details), details


def test_connect_honors_kanban_busy_timeout_env(kanban_home, monkeypatch):
    """All kanban connections should use the explicit busy-timeout knob.

    A worker stampede should wait for SQLite's writer lock instead of failing
    immediately with ``database is locked`` during first-connect/WAL/schema
    setup.  The timeout must be queryable via PRAGMA so CLI, gateway, and tool
    connections behave the same way.
    """
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "123456")

    with kb.connect() as conn:
        row = conn.execute("PRAGMA busy_timeout").fetchone()

    assert row[0] == 123456


@pytest.mark.windows_only
def test_cross_process_init_lock_uses_windows_byte_range_lock(tmp_path, monkeypatch):
    """Windows must use a real (non-blocking) process lock, not a no-op open.

    The init lock acquires with LK_NBLCK in a bounded retry loop (#36644) so a
    wedged holder can never block connect() forever; a clean acquire takes the
    lock once and releases it once.

    ``windows_only``: ``msvcrt`` does not exist off Windows, so faking
    ``_IS_WINDOWS`` on Linux meant injecting a fake ``msvcrt`` module too —
    the test then asserted against its own stub rather than the byte-range
    locking API. Here the platform is real; only ``msvcrt.locking`` is
    instrumented so the call sequence is observable.
    """
    calls: list[tuple[int, int, int]] = []
    import msvcrt as _msvcrt

    fake_msvcrt = types.SimpleNamespace(
        LK_NBLCK=_msvcrt.LK_NBLCK,
        LK_UNLCK=_msvcrt.LK_UNLCK,
        locking=lambda fd, mode, nbytes: calls.append((fd, mode, nbytes)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    db_path = tmp_path / "kanban.db"
    with kb._cross_process_init_lock(db_path):
        # Acquired exactly once via the non-blocking byte-range lock.
        assert [call[1:] for call in calls] == [(fake_msvcrt.LK_NBLCK, 1)]

    # Released once on exit.
    assert [call[1:] for call in calls] == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]


def test_connect_migrates_legacy_db_before_optional_column_indexes(tmp_path):
    """Legacy DBs missing additive indexed columns must migrate cleanly.

    SCHEMA_SQL runs in ``connect()`` before ``_migrate_add_optional_columns``.
    Indexes over additive columns therefore must be created after the
    migration adds those columns, or boards predating the column fail to
    open before migration can run.

    Covers all four indexes that sit on additive columns:
    - ``tasks.session_id``       -> ``idx_tasks_session_id``    (#28447)
    - ``tasks.tenant``           -> ``idx_tasks_tenant``        (#16081)
    - ``tasks.idempotency_key``  -> ``idx_tasks_idempotency``   (#17805)
    - ``task_events.run_id``     -> ``idx_events_run``          (#17805)
    """
    db_path = tmp_path / "legacy-kanban.db"
    conn = sqlite3.connect(str(db_path))
    # Pre-#16081 ``tasks`` shape: missing tenant, idempotency_key, session_id.
    conn.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
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
    """)
    # Pre-#17805 ``task_events`` shape: missing run_id. Required because
    # ``_migrate_add_optional_columns`` unconditionally runs PRAGMA on
    # ``task_events`` for run_id back-fill.
    conn.execute("""
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        )
    """)
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy', 'old board task', 'ready', 1)"
    )
    conn.commit()
    conn.close()

    with kb.connect(db_path) as migrated:
        task_columns = {
            row["name"] for row in migrated.execute("PRAGMA table_info(tasks)")
        }
        event_columns = {
            row["name"]
            for row in migrated.execute("PRAGMA table_info(task_events)")
        }
        indexes = {
            row["name"]
            for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }

    # Additive columns added by migration:
    assert "session_id" in task_columns
    assert "tenant" in task_columns
    assert "idempotency_key" in task_columns
    assert "run_id" in event_columns
    # And their indexes — the regression scope of this test:
    assert "idx_tasks_session_id" in indexes
    assert "idx_tasks_tenant" in indexes
    assert "idx_tasks_idempotency" in indexes
    assert "idx_events_run" in indexes


# ---------------------------------------------------------------------------
# Task creation + status inference
# ---------------------------------------------------------------------------

def test_create_task_no_parents_is_ready(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship it", assignee="alice")
        t = kb.get_task(conn, tid)
    assert t is not None
    assert t.status == "ready"
    assert t.assignee == "alice"
    assert t.workspace_kind == "scratch"


def test_create_task_persists_model_override(kanban_home):
    """A per-task model override round-trips through create_task -> row.

    The dispatcher spawns the worker with ``-m <model_override>`` when set,
    so an override must survive the INSERT. Omitting it (or passing blank)
    must leave the column NULL so the worker falls back to the profile model.
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="cheap card", assignee="alice",
            model_override="claude-sonnet-5-0",
        )
        default_tid = kb.create_task(conn, title="normal card", assignee="alice")
        blank_tid = kb.create_task(
            conn, title="blank override", assignee="alice", model_override="  ",
        )
        overridden = kb.get_task(conn, tid)
        defaulted = kb.get_task(conn, default_tid)
        blanked = kb.get_task(conn, blank_tid)
    assert overridden.model_override == "claude-sonnet-5-0"
    assert defaulted.model_override is None
    assert blanked.model_override is None


def test_create_task_with_parent_is_todo_until_parent_done(kanban_home):
    with kb.connect() as conn:
        p = kb.create_task(conn, title="parent")
        c = kb.create_task(conn, title="child", parents=[p])
        assert kb.get_task(conn, c).status == "todo"
        kb.complete_task(conn, p, result="ok")
        assert kb.get_task(conn, c).status == "ready"


def test_create_task_unknown_parent_errors(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="unknown parent"):
        kb.create_task(conn, title="orphan", parents=["t_ghost"])


def test_workspace_kind_validation(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="workspace_kind"):
        kb.create_task(conn, title="bad ws", workspace_kind="cloud")


def test_create_strips_worktree_scheme_from_workspace_path(kanban_home, tmp_path):
    """A 'worktree:<path>' value jammed into workspace_path must split into
    workspace_kind='worktree' + a BARE absolute workspace_path (regression for
    the scheme prefix that tripped the dispatcher circuit breaker)."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="scheme in path",
            workspace_path=f"worktree:{repo}",
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "worktree"
    workspace_path = task.workspace_path
    assert workspace_path is not None
    assert workspace_path == str(repo)
    assert not workspace_path.startswith("worktree:")


def test_create_strips_dir_scheme_from_workspace_path(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="dir scheme", workspace_path="dir:/abs/work/dir"
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "dir"
    assert task.workspace_path == "/abs/work/dir"


def test_create_explicit_kind_not_overridden_by_stray_prefix(kanban_home):
    """An explicit non-default workspace_kind wins; only the prefix is stripped."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="explicit dir",
            workspace_kind="dir",
            workspace_path="dir:/abs/keep",
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "dir"
    assert task.workspace_path == "/abs/keep"


def test_strip_scheme_worktree_kind_with_prefix_keeps_kind():
    """A row already 'worktree' kind but carrying a 'worktree:' prefix in the
    path (Codex P2: legacy worktree row) must strip the prefix and keep the
    kind, so dispatch routes it through worktree materialization with a bare
    absolute path instead of failing the absolute-path check."""
    assert kb._strip_workspace_scheme("worktree", "worktree:/abs/repo") == (
        "worktree",
        "/abs/repo",
    )


def test_create_bare_path_unchanged(kanban_home, tmp_path):
    """A healthy bare absolute path must pass through untouched."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="bare path",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(repo)


def test_create_task_persists_worktree_branch_name(kanban_home, tmp_path):
    # Anchor the worktree target inside a real git repo so it passes the
    # create-time repo-root guard (the target itself need not exist yet — its
    # parent being inside the repo is enough for the materialize semantics).
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "t6-wire"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="ship worktree",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name=" wt/t6-wire ",
        )
        task = kb.get_task(conn, tid)
        events = kb.list_events(conn, tid)
        context = kb.build_worker_context(conn, tid)

    assert task.branch_name == "wt/t6-wire"
    assert events[0].payload["branch_name"] == "wt/t6-wire"
    assert "Branch:   wt/t6-wire" in context


def test_branch_name_requires_worktree_workspace(kanban_home):
    with kb.connect() as conn, pytest.raises(ValueError, match="worktree"):
        kb.create_task(
            conn,
            title="bad branch",
            workspace_kind="scratch",
            branch_name="wt/bad",
        )


# ---------------------------------------------------------------------------
# Links + dependency resolution
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Atomic claim (CAS)
# ---------------------------------------------------------------------------



def test_schedule_task_parks_time_delay_without_dispatching(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="delayed recheck", assignee="ops")
        assert kb.schedule_task(conn, t, reason="run next week") is True
        task = kb.get_task(conn, t)
        assert task.status == "scheduled"
        assert kb.claim_task(conn, t) is None

        events = kb.list_events(conn, t)
        assert any(e.kind == "scheduled" and e.payload == {"reason": "run next week"} for e in events)








def test_stale_claim_reclaim_event_records_diagnostic_payload(
    kanban_home, monkeypatch,
):
    """``reclaimed`` events should carry claim_expires, last_heartbeat_at,
    and worker_pid so operators can diagnose why a claim went stale
    (#23025: previous payload only had ``stale_lock`` which gives no
    timing context)."""
    import json
    import hermes_cli.kanban_db as _kb

    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        host = _kb._claimer_id().split(":", 1)[0]
        kb.claim_task(conn, t, claimer=f"{host}:worker")
        kb._set_worker_pid(conn, t, 12345)
        old_expires = int(time.time()) - 3600
        hb_at = int(time.time()) - 1800
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, last_heartbeat_at = ? "
            "WHERE id = ?",
            (old_expires, hb_at, t),
        )

        monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
        kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None)
        row = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'reclaimed'",
            (t,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload"])
        assert payload["claim_expires"] == old_expires
        assert payload["last_heartbeat_at"] == hb_at
        assert payload["worker_pid"] == 12345
        assert payload["host_local"] is True






# ---------------------------------------------------------------------------
# Rate-limit requeue: a worker that bails on a provider quota wall must be
# released back to ``ready`` WITHOUT counting a failure, so a long (e.g.
# 5-hour) quota window can't trip the circuit breaker and permanently block
# the card. The respawn guard then defers it on a cooldown until quota
# returns. Regression coverage for the kanban-rate-limit-failure report.
# ---------------------------------------------------------------------------


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8




def test_rate_limit_exit_requeues_without_counting_failure(
    kanban_home, monkeypatch,
):
    """A rate-limit sentinel exit releases the task to ``ready`` and leaves
    ``consecutive_failures`` untouched — the breaker must never trip on a
    transient throttle, even across many quota-wall hits."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="rl", assignee="a")

        # Simulate FAR more quota-wall hits than DEFAULT_FAILURE_LIMIT (2).
        # If any of these counted as a failure the task would be blocked.
        for i in range(6):
            pid = 70000 + i
            # Claim to open a real run (so detect_crashed_workers can close
            # it with a rate_limited outcome), then point the claim at this
            # host + a dead pid so the crash path acts on it.
            kb.claim_task(conn, tid, claimer=f"{host}:w{i}")
            conn.execute(
                "UPDATE tasks SET worker_pid=?, consecutive_failures=? "
                "WHERE id=?",
                (pid, 0, tid),
            )
            conn.commit()
            _kb._record_worker_exit(
                pid, _exited_status(_kb.KANBAN_RATE_LIMIT_EXIT_CODE)
            )

            crashed = kb.detect_crashed_workers(conn)
            # Rate-limited requeues are NOT crashes.
            assert tid not in crashed
            rl = getattr(_kb.detect_crashed_workers, "_last_rate_limited", [])
            assert tid in rl

            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                f"hit {i}: should requeue ready, got {task.status}"
            )
            assert task.consecutive_failures == 0, (
                f"hit {i}: rate-limit must not count a failure, "
                f"got {task.consecutive_failures}"
            )

        # Last failure error stamped so the respawn guard recognizes the
        # quota wall.
        assert task.last_failure_error and "rate-limited" in task.last_failure_error

        # A ``rate_limited`` run outcome was recorded (not ``crashed``).
        outcomes = [
            r["outcome"] for r in conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id=?", (tid,),
            ).fetchall()
        ]
        assert "rate_limited" in outcomes
        assert "crashed" not in outcomes




def test_respawn_guard_defers_rate_limited_within_cooldown(
    kanban_home, monkeypatch,
):
    """Within the cooldown after a rate-limit requeue, the guard defers the
    respawn; after the cooldown it allows a probe — and crucially does NOT
    fall into ``blocker_auth`` (which would defer forever)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = 5_000_000

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="rl-guard", assignee="a")
        # Seed a rate_limited run that just ended + the stamped error.
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        conn.execute(
            "UPDATE task_runs SET outcome='rate_limited', status='rate_limited', "
            "ended_at=? WHERE id=?",
            (now, run_id),
        )
        conn.execute(
            "UPDATE tasks SET status='ready', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
            "last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued", tid),
        )
        conn.commit()

        # Inside cooldown → defer with the rate-limit-specific reason.
        monkeypatch.setattr(_kb.time, "time", lambda: now + 100)
        assert kb.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        # Past cooldown → allowed (None), NOT trapped by blocker_auth even
        # though last_failure_error contains "rate-limited".
        monkeypatch.setattr(_kb.time, "time", lambda: now + 400)
        assert kb.check_respawn_guard(conn, tid) is None








# ---------------------------------------------------------------------------
# Complete / block / unblock / archive / assign
# ---------------------------------------------------------------------------





def test_unblock_resets_failure_counters(kanban_home):
    """unblock_task must reset consecutive_failures and last_failure_error."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="a")
        kb.claim_task(conn, t)
        assert kb.block_task(conn, t, reason="need input")
        # Simulate accumulated failures from the circuit breaker
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 5, "
            "last_failure_error = 'test error' WHERE id = ?",
            (t,),
        )
        conn.commit()
        assert kb.unblock_task(conn, t)
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None


def test_recompute_ready_skips_tasks_at_failure_limit(kanban_home):
    """recompute_ready must not auto-recover tasks whose consecutive_failures
    has reached the circuit-breaker limit (#35072).

    Without this guard, a task that repeatedly exhausts its iteration
    budget would cycle forever: block → auto-recover (counter reset)
    → respawn → budget exhausted → block → …
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(conn, title="child", assignee="a",
                               parents=[parent])
        # Complete the parent so the child's dependencies are satisfied.
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, summary="done")

        # Simulate the child having exhausted its budget twice,
        # hitting the default failure limit (2).
        kb.claim_task(conn, child)
        kb._record_task_failure(
            conn, child, error="budget exhausted 1",
            outcome="timed_out", release_claim=True, end_run=True,
            failure_limit=2,
        )
        kb._record_task_failure(
            conn, child, error="budget exhausted 2",
            outcome="timed_out", release_claim=True, end_run=True,
            failure_limit=2,
        )
        task = kb.get_task(conn, child)
        assert task.status == "blocked"
        assert task.consecutive_failures >= 2

        # recompute_ready must NOT promote this task — the circuit
        # breaker has tripped and it should stay blocked.
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child).status == "blocked"

        # Explicit unblock should still work and reset the counter.
        assert kb.unblock_task(conn, child)
        task = kb.get_task(conn, child)
        assert task.status == "ready"
        assert task.consecutive_failures == 0


def test_recompute_ready_recovers_below_limit(kanban_home):
    """recompute_ready auto-recovers blocked tasks that haven't hit the
    failure limit yet — the counter is preserved across recovery."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="real work", body="do the thing", assignee="a")
        kb.claim_task(conn, t)
        # One failure, below the default limit of 2.
        kb._record_task_failure(
            conn, t, error="budget exhausted 1",
            outcome="timed_out", release_claim=True, end_run=True,
            failure_limit=2,
        )
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.consecutive_failures == 1

        # Simulate being blocked by something else (not circuit breaker).
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?", (t,),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        # Counter must be preserved, not reset.
        assert task.consecutive_failures == 1


def test_recompute_ready_honours_dispatcher_failure_limit(kanban_home):
    """The guard's effective limit must follow the same resolution order
    as the circuit breaker (#35072): per-task max_retries → dispatcher
    failure_limit → DEFAULT_FAILURE_LIMIT.

    Without threading the dispatcher's ``kanban.failure_limit`` through,
    the guard falls back to DEFAULT_FAILURE_LIMIT and disagrees with the
    breaker — sticking a task prematurely (config limit > default) or
    letting a tripped task escape (config limit < default).
    """
    with kb.connect() as conn:
        # Config allows MORE retries than the default. A task blocked
        # with failures below the configured limit must still recover.
        t = kb.create_task(conn, title="lenient", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=? "
            "WHERE id=?",
            (kb.DEFAULT_FAILURE_LIMIT, t),
        )
        conn.commit()
        # Default-limit call would stick it (failures >= default).
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, t).status == "blocked"
        # Dispatcher configured a higher limit → recover, preserve counter.
        promoted = kb.recompute_ready(
            conn, failure_limit=kb.DEFAULT_FAILURE_LIMIT + 2
        )
        assert promoted == 1
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.consecutive_failures == kb.DEFAULT_FAILURE_LIMIT

        # Config allows FEWER retries than the default. A task at the
        # stricter limit must stay blocked even though it's below default.
        t2 = kb.create_task(conn, title="strict", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1 "
            "WHERE id=?",
            (t2,),
        )
        conn.commit()
        # Default-limit (2) would recover it (1 < 2).
        # Stricter config limit (1) must keep it blocked (1 >= 1).
        assert kb.recompute_ready(conn, failure_limit=1) == 0
        assert kb.get_task(conn, t2).status == "blocked"




# ---------------------------------------------------------------------------
# Parent-completion invariant at the claim gate (RCA t_a6acd07d)
# ---------------------------------------------------------------------------














def test_archive_task_stamps_archive_time_without_rewriting_completion(kanban_home):
    """Archiving a done card updates the archive-window timestamp, not completed_at."""

    with kb.connect() as conn:
        t = kb.create_task(conn, title="done long ago")
        kb.complete_task(conn, t)
        old_completed_at = 1_000
        conn.execute(
            "UPDATE tasks SET completed_at = ? WHERE id = ?",
            (old_completed_at, t),
        )

        before_archive = int(time.time())
        assert kb.archive_task(conn, t)

        row = conn.execute(
            "SELECT completed_at, archived_at FROM tasks WHERE id = ?",
            (t,),
        ).fetchone()
        assert row["completed_at"] == old_completed_at
        assert row["archived_at"] >= before_archive


def test_archive_task_refreshes_archive_time_on_rearchive(kanban_home):
    """Re-archiving a reopened card should move it to the front of archived windows."""

    with kb.connect() as conn:
        t = kb.create_task(conn, title="reopened archived card")
        kb.complete_task(conn, t)
        assert kb.archive_task(conn, t)
        conn.execute(
            "UPDATE tasks SET status = 'ready', archived_at = ? WHERE id = ?",
            (100, t),
        )

        before_rearchive = int(time.time())
        assert kb.archive_task(conn, t)

        archived_at = conn.execute(
            "SELECT archived_at FROM tasks WHERE id = ?",
            (t,),
        ).fetchone()["archived_at"]
        assert archived_at >= before_rearchive
        assert archived_at != 100


def test_migration_backfills_archived_at_from_latest_archive_event(kanban_home):
    """Legacy archived cards should sort by their archived event after upgrade."""

    with kb.connect() as conn:
        t = kb.create_task(conn, title="legacy archived card")
        kb.complete_task(conn, t)
        conn.execute(
            "UPDATE tasks SET status = 'archived', completed_at = ?, archived_at = NULL WHERE id = ?",
            (1_000, t),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
            (t, "archived", None, 20_000),
        )

        kb._migrate_add_optional_columns(conn)

        archived_at = conn.execute(
            "SELECT archived_at FROM tasks WHERE id = ?",
            (t,),
        ).fetchone()["archived_at"]
        assert archived_at == 20_000


def test_delete_archived_task_removes_related_rows(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        tid = kb.create_task(conn, title="child", parents=[parent], assignee="worker")
        kb.add_comment(conn, tid, "user", "cleanup me")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="done")
        assert kb.archive_task(conn, tid)
        conn.execute(
            "INSERT INTO kanban_notify_subs(task_id, platform, chat_id, thread_id, user_id, created_at, last_event_id) "
            "VALUES (?, 'telegram', '123', '', 'u', 0, 0)",
            (tid,),
        )
        conn.commit()

        assert kb.delete_archived_task(conn, tid) is True
        assert kb.get_task(conn, tid) is None
        assert conn.execute("SELECT COUNT(*) FROM task_links WHERE child_id = ? OR parent_id = ?", (tid, tid)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_comments WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (tid,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?", (tid,)).fetchone()[0] == 0


def test_delete_task_removes_task_and_cascades(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="to-delete", assignee="alice")
        kb.add_comment(conn, t, "user", "comment")
        kb.add_comment(conn, t, "user", "another")
        assert kb.delete_task(conn, t)
        assert kb.get_task(conn, t) is None
        assert len(kb.list_comments(conn, t)) == 0
        assert len(kb.list_events(conn, t)) == 0
        assert len(kb.list_runs(conn, t)) == 0




# ---------------------------------------------------------------------------
# Comments / events / worker context
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------



def test_dispatch_skips_unassigned(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="floater")
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_unassigned
    assert t not in res.skipped_nonspawnable
    assert not res.spawned


def test_dispatch_skips_nonspawnable_into_separate_bucket(kanban_home, monkeypatch):
    """Tasks whose assignee fails profile_exists() must NOT land in
    ``skipped_unassigned`` (which is operator-actionable) — they go in
    the dedicated ``skipped_nonspawnable`` bucket so health telemetry
    can suppress false-positive "stuck" warnings."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        t = kb.create_task(conn, title="for-terminal", assignee="orion-cc")
        res = kb.dispatch_once(conn, dry_run=True)
    assert t in res.skipped_nonspawnable
    assert t not in res.skipped_unassigned
    assert not res.spawned


def test_has_spawnable_ready_false_when_only_terminal_lanes(kanban_home, monkeypatch):
    """``has_spawnable_ready`` returns False when every ready task is
    assigned to a control-plane lane — used by gateway/CLI dispatchers
    to silence the stuck-warn while terminals still have queued work."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    with kb.connect() as conn:
        kb.create_task(conn, title="t1", assignee="orion-cc")
        kb.create_task(conn, title="t2", assignee="orion-research")
        assert kb.has_spawnable_ready(conn) is False


def test_has_spawnable_ready_true_when_real_profile_present(kanban_home, monkeypatch):
    """``has_spawnable_ready`` returns True as soon as ANY ready task
    has an assignee that maps to a real Hermes profile — preserves the
    real "stuck" signal when a daily/agent task is queued."""
    from hermes_cli import profiles
    monkeypatch.setattr(
        profiles, "profile_exists", lambda name: name == "daily"
    )
    with kb.connect() as conn:
        kb.create_task(conn, title="terminal-task", assignee="orion-cc")
        kb.create_task(conn, title="hermes-task", assignee="daily")
        assert kb.has_spawnable_ready(conn) is True


def test_has_spawnable_ready_false_on_empty_queue(kanban_home):
    """Empty queue is the trivial false case — no ready tasks at all."""
    with kb.connect() as conn:
        assert kb.has_spawnable_ready(conn) is False


def test_dispatch_promotes_ready_and_spawns(kanban_home, all_assignees_spawnable):
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append((task.id, task.assignee, workspace))

    with kb.connect() as conn:
        p = kb.create_task(conn, title="p", assignee="alice")
        c = kb.create_task(conn, title="c", assignee="bob", parents=[p])
        # Finish parent outside dispatch; promotion happens inside.
        kb.complete_task(conn, p)
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)
    # Spawned c (a was already done when dispatch was called).
    assert len(spawns) == 1
    assert spawns[0][0] == c
    assert spawns[0][1] == "bob"
    # c is now running
    with kb.connect() as conn:
        assert kb.get_task(conn, c).status == "running"


def test_dispatch_self_heals_persisted_scheme_prefix(
    kanban_home, all_assignees_spawnable, tmp_path
):
    """A task already persisted with a scheme prefix in workspace_path (written
    by a create surface that bypassed the guard) must be self-healed AND the
    correction persisted at claim time, so the spawned task and the DB both
    carry the promoted workspace_kind. Regression for the Gemini HIGH finding:
    healing only a local copy in resolve_workspace left workspace_kind='scratch'
    in the DB, breaking branch wiring and scratch-tip emission."""
    target = tmp_path / "healed-dir"
    spawned = []

    def fake_spawn(task, workspace):
        spawned.append((task.workspace_kind, task.workspace_path, workspace))

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="healme", assignee="alice")
        # Simulate a malformed persisted value: jam 'dir:<abs>' into
        # workspace_path with the default scratch kind, bypassing create_task's
        # guard (direct UPDATE, as a third-party create surface would).
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'scratch', workspace_path = ? "
                "WHERE id = ?",
                (f"dir:{target}", tid),
            )
        kb.dispatch_once(conn, spawn_fn=fake_spawn)
        healed = kb.get_task(conn, tid)

    # The spawned task object carries the healed values...
    assert spawned and spawned[0][0] == "dir"
    assert spawned[0][1] == str(target)
    # ...and the correction is persisted to the DB (not just a local copy).
    assert healed is not None
    assert healed.workspace_kind == "dir"
    assert healed.workspace_path == str(target)


def test_dispatch_review_self_heals_persisted_scheme_prefix(
    kanban_home, all_assignees_spawnable, tmp_path
):
    """The review-queue claim path must self-heal a persisted scheme prefix the
    same way the ready-queue path does. Regression for the Codex P2: a legacy
    task already in status='review' with workspace_path='<scheme>:<abs>' bypassed
    the heal, branched on the unpromoted workspace_kind, and could trip the
    spawn-failure circuit breaker."""
    target = tmp_path / "healed-dir"
    spawned = []

    def fake_spawn(task, workspace):
        spawned.append((task.workspace_kind, task.workspace_path, workspace))

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review-heal", assignee="alice")
        _set_task_status(conn, tid, "review")
        # Malformed persisted value on a review-status task: 'dir:<abs>'
        # jammed into workspace_path with the default scratch kind.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'scratch', workspace_path = ? "
                "WHERE id = ?",
                (f"dir:{target}", tid),
            )
        kb.dispatch_once(conn, spawn_fn=fake_spawn)
        healed = kb.get_task(conn, tid)

    # The spawned task object carries the healed (promoted) values...
    assert spawned and spawned[0][0] == "dir"
    assert spawned[0][1] == str(target)
    # ...and the correction is persisted to the DB, not just a local copy.
    assert healed is not None
    assert healed.workspace_kind == "dir"
    assert healed.workspace_path == str(target)


def test_resolve_workspace_persists_healed_kind_with_conn(kanban_home, tmp_path):
    """resolve_workspace(conn=...) must persist BOTH the promoted kind and the
    bare path for a legacy row, so callers that only write back the resolved
    path (the manual ``hermes kanban claim`` path) don't leave a stale
    workspace_kind='scratch' in the DB. Codex P2: persist healed kinds for
    manual claims."""
    target = tmp_path / "legacy-dir"
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="manual-heal", assignee="alice")
        # Malformed persisted value: 'dir:<abs>' with default scratch kind.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'scratch', workspace_path = ? "
                "WHERE id = ?",
                (f"dir:{target}", tid),
            )
        task = kb.get_task(conn, tid)
        assert task is not None
        resolved = kb.resolve_workspace(task, conn=conn)
        healed = kb.get_task(conn, tid)

    assert resolved == target
    # The correction is persisted (not just a local copy): kind promoted to dir.
    assert healed is not None
    assert healed.workspace_kind == "dir"
    assert healed.workspace_path == str(target)


def test_resolve_workspace_without_conn_does_not_persist(kanban_home, tmp_path):
    """Without a conn, resolve_workspace still heals its local copy (back-compat
    for read-only callers) but must NOT touch the DB."""
    target = tmp_path / "legacy-dir2"
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="noconn-heal", assignee="alice")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'scratch', workspace_path = ? "
                "WHERE id = ?",
                (f"dir:{target}", tid),
            )
        task = kb.get_task(conn, tid)
        assert task is not None
        resolved = kb.resolve_workspace(task)  # no conn
        unhealed = kb.get_task(conn, tid)

    assert resolved == target
    # DB row is unchanged because no conn was supplied.
    assert unhealed is not None
    assert unhealed.workspace_kind == "scratch"
    assert unhealed.workspace_path == f"dir:{target}"


def test_dispatch_spawn_failure_releases_claim(kanban_home, all_assignees_spawnable):
    def boom(task, workspace):
        raise RuntimeError("spawn failed")

    with kb.connect() as conn:
        t = kb.create_task(conn, title="boom", assignee="alice")
        kb.dispatch_once(conn, spawn_fn=boom)
        # Must return to ready so the next tick can retry.
        assert kb.get_task(conn, t).status == "ready"
        assert kb.get_task(conn, t).claim_lock is None


def test_dispatch_max_spawn_counts_existing_running_tasks(
    kanban_home, all_assignees_spawnable
):
    """max_spawn is a live concurrency cap, not a per-tick spawn cap.

    Without counting tasks already in ``running``, every dispatcher tick can
    launch up to ``max_spawn`` more workers while previous workers are still
    alive. Long-running boards then accumulate unbounded worker subprocesses.
    """
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running_a = kb.create_task(conn, title="running-a", assignee="alice")
        running_b = kb.create_task(conn, title="running-b", assignee="bob")
        ready = kb.create_task(conn, title="ready", assignee="carol")
        kb.claim_task(conn, running_a)
        kb.claim_task(conn, running_b)

        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)

        assert res.spawned == []
        assert spawns == []
        assert kb.get_task(conn, ready).status == "ready"


def test_dispatch_max_spawn_fills_remaining_capacity(
    kanban_home, all_assignees_spawnable
):
    """When below cap, dispatch only fills available worker slots."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        ready_a = kb.create_task(conn, title="ready-a", assignee="bob")
        ready_b = kb.create_task(conn, title="ready-b", assignee="carol")
        kb.claim_task(conn, running)

        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)

        assert len(res.spawned) == 1
        assert spawns == [ready_a]
        assert kb.get_task(conn, ready_a).status == "running"
        assert kb.get_task(conn, ready_b).status == "ready"


def test_dispatch_reclaims_stale_before_spawning(kanban_home):
    with kb.connect() as conn:
        t = kb.create_task(conn, title="x", assignee="alice")
        kb.claim_task(conn, t)
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (int(time.time()) - 1, t),
        )
        res = kb.dispatch_once(conn, dry_run=True)
    assert res.reclaimed == 1


# ---------------------------------------------------------------------------
# Respawn guard (check_respawn_guard + dispatch_once integration)
# ---------------------------------------------------------------------------







def test_respawn_guard_blocker_auth_on_authentication_error(kanban_home):
    """Full word 'Authentication' triggers blocker_auth (regex covers auth\\w*)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="authn-task", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("Authentication failed: invalid credentials", t),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "blocker_auth"


def test_respawn_guard_blocker_auth_on_authorization_error(kanban_home):
    """Full word 'authorization' triggers blocker_auth (regex covers auth\\w*)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="authz-task", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("authorization denied for scope repo", t),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "blocker_auth"


def test_respawn_guard_recent_success(kanban_home):
    """A completed run within the guard window triggers recent_success."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="already-done", assignee="alice")
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, now - 120, now - 60),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "recent_success"


def test_respawn_guard_recent_success_bypassed_by_requeue(kanban_home):
    """An explicit re-queue after a recent success (operator done->ready,
    promote, unblock, reclaim) is a deliberate re-run and must bypass the
    recent_success guard — otherwise a manual done->ready just sits there
    until the window elapses."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="rerun-me", assignee="alice")
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, now - 120, now - 60),
        )
        # Baseline: a recent completion defers the respawn.
        assert kb.check_respawn_guard(conn, t) == "recent_success"
        # Operator drags done -> ready: a 'status' event after completion.
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) "
            "VALUES (?, 'status', ?)",
            (t, now - 10),
        )
        assert kb.check_respawn_guard(conn, t) is None


def test_respawn_guard_stale_success_not_guarded(kanban_home):
    """A completed run outside the guard window does not block re-spawn."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="old-done", assignee="alice")
        old_end = int(time.time()) - kb._RESPAWN_GUARD_SUCCESS_WINDOW - 60
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, old_end - 300, old_end),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_active_pr_in_comment(kanban_home):
    """A GitHub PR URL in a recent comment BY THE TASK'S OWN LANE triggers
    active_pr."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(
            conn, t, "alice",
            "PR created: https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "active_pr"


def test_respawn_guard_active_pr_fallback_worker_comment(kanban_home):
    """A dispatcher worker without ``HERMES_PROFILE`` records its task-owned
    PR handoff as ``worker``; that legacy fallback must still guard an
    assigned task from opening a duplicate PR."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(
            conn, t, "worker",
            "PR created: https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "active_pr"


def test_respawn_guard_cross_author_pr_comment_not_guarded(kanban_home):
    """A PR URL cross-posted by a DIFFERENT lane (context from a sibling card,
    reviewer notes, an operator pasting a link) does not strand the task.

    Regression: a grade-only memo-evaluator card was respawn_guarded for the
    full 24h PR window because a dev lane cross-posted its (already-merged)
    PR URL as context (live incident 2026-07-17, t_622d5a37)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="grade-only", assignee="memo-evaluator")
        kb.add_comment(
            conn, t, "dev",
            "Context: fix merged in https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_active_pr_released_by_requeue(kanban_home):
    """An explicit re-queue event STRICTLY after the qualifying PR comment
    bypasses active_pr, mirroring the recent_success exception."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        past = int(time.time()) - 120
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'alice', "
            "'PR created: https://github.com/totemx-AI/subsidysmart/pull/42', ?)",
            (t, past),
        )
        kb._append_event(conn, t, "unblocked", {})
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_active_pr_same_second_requeue_still_guarded(kanban_home):
    """A requeue event in the SAME one-second bucket as the PR comment cannot
    prove after-ordering (auto-promotion → spawn → PR can land within 1s), so
    the tie keeps the guard — fail safe toward not duplicating a PR."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        now_ts = int(time.time())
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'alice', "
            "'PR created: https://github.com/totemx-AI/subsidysmart/pull/42', ?)",
            (t, now_ts),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'unblocked', '{}', ?)",
            (t, now_ts),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "active_pr"


def test_respawn_guard_active_pr_automatic_events_do_not_bypass(kanban_home):
    """Automatic 'reclaimed'/'promoted' events after the PR comment do NOT
    bypass active_pr — only operator-originated kinds do.

    A worker that crashes after opening its PR gets an automatic 'reclaimed'
    from release_stale_claims; treating that as a deliberate requeue would
    respawn the task and open the exact duplicate PR the guard prevents."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        past = int(time.time()) - 120
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'alice', "
            "'PR created: https://github.com/totemx-AI/subsidysmart/pull/42', ?)",
            (t, past),
        )
        for kind in ("reclaimed", "promoted"):
            kb._append_event(conn, t, kind, {})
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "active_pr"


def test_respawn_guard_active_pr_manual_reclaim_bypasses(kanban_home):
    """Operator reclaim records ``reclaimed`` with ``manual: true`` and is a
    deliberate recovery action, so it releases an existing PR guard."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        past = int(time.time()) - 120
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'alice', "
            "'PR created: https://github.com/totemx-AI/subsidysmart/pull/42', ?)",
            (t, past),
        )
        kb._append_event(conn, t, "reclaimed", {"manual": True})
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_active_pr_released_by_manual_promote(kanban_home):
    """`hermes kanban promote` emits 'promoted_manual' — an operator verb —
    which bypasses active_pr like 'status'/'unblocked' do."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        past = int(time.time()) - 120
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'alice', "
            "'PR created: https://github.com/totemx-AI/subsidysmart/pull/42', ?)",
            (t, past),
        )
        kb._append_event(conn, t, "promoted_manual", {"actor": "eric"})
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_active_pr_parent_reopen_status_does_not_bypass(kanban_home):
    """The automatic child 'status' event emitted by parent-reopen dependency
    maintenance (payload reason='parent_reopened') does NOT bypass active_pr —
    same automatic-event class as 'reclaimed'/'promoted'. An operator status
    event (no parent_reopened reason) still does."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        past = int(time.time()) - 120
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'alice', "
            "'PR created: https://github.com/totemx-AI/subsidysmart/pull/42', ?)",
            (t, past),
        )
        kb._append_event(
            conn, t, "status",
            {"status": "todo", "reason": "parent_reopened", "parent": "t_p"},
        )
        assert kb.check_respawn_guard(conn, t) == "active_pr"
        kb._append_event(conn, t, "status", {"status": "ready"})
        assert kb.check_respawn_guard(conn, t) is None


def test_respawn_guard_active_pr_survives_reassignment(kanban_home):
    """A prior worker's PR comment still guards after the task is reassigned:
    own-lane matching includes any profile in task_runs for this task, not
    just the current assignee. (A sibling lane's cross-post still never
    matches — that profile has no run row here.)"""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES (?, 'alice', 'crashed', ?)",
            (t, int(time.time()) - 300),
        )
        kb.add_comment(
            conn, t, "alice",
            "PR created: https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        conn.execute("UPDATE tasks SET assignee = 'bob' WHERE id = ?", (t,))
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "active_pr"


def test_respawn_guard_active_pr_case_insensitive_author_match(kanban_home):
    """Display-cased assignee ("Alice") vs normalized comment author ("alice")
    must still match — both sides canonicalize via _canonical_assignee. A raw
    compare would drop the task's own PR comment and release the guard."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        conn.execute("UPDATE tasks SET assignee = 'Alice' WHERE id = ?", (t,))
        kb.add_comment(
            conn, t, "alice",
            "PR created: https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "active_pr"


def test_respawn_guard_active_pr_blank_author_ignored(kanban_home):
    """A blank/whitespace comment author neither crashes normalization nor
    counts as own-lane."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ' ', "
            "'PR: https://github.com/totemx-AI/subsidysmart/pull/42', ?)",
            (t, int(time.time())),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_old_pr_comment_not_guarded(kanban_home):
    """A GitHub PR URL in a comment older than the PR window does not block."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="old-pr", assignee="alice")
        old_ts = int(time.time()) - kb._RESPAWN_GUARD_PR_WINDOW - 60
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'worker', "
            "'PR: https://github.com/totemx-AI/subsidysmart/pull/10', ?)",
            (t, old_ts),
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_babysit_task_not_stranded_by_own_pr_comment(kanban_home):
    """A babysit/re-verify task anchored to a pre-existing PR is NOT guarded
    by active_pr even when its own comments echo that PR's URL.

    Regression: the active_pr guard exists to stop an IMPLEMENTATION task from
    re-spawning and opening a DUPLICATE PR. A babysit task's whole job is to
    keep working an EXISTING PR, so it posts the PR URL in its comments and
    must re-spawn — the guard previously stranded it for the full 24h PR
    window. The discriminator is a PR reference in the task title/body.
    """
    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="Babysit PR #48 to merge-ready: ruff fix",
            assignee="pr-babysitter",
        )
        kb.add_comment(
            conn, t, "pr-babysitter",
            "PR #48 head pushed: https://github.com/exiao/hermes-agent/pull/48",
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_babysit_task_owner_repo_ref_not_stranded(kanban_home):
    """An owner/repo#N anchor in the title (e.g. cpe-research/research-agent#801)
    also exempts the task from the active_pr comment guard."""
    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="Babysit cpe-research/research-agent#801 to CI-green",
            assignee="pr-babysitter",
        )
        kb.add_comment(
            conn, t, "pr-babysitter",
            "head 67351a4e: https://github.com/cpe-research/research-agent/pull/801",
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason is None


def test_respawn_guard_impl_task_still_guarded_by_pr_comment(kanban_home):
    """An IMPLEMENTATION task (no PR reference in title/body) is STILL guarded
    by active_pr when a PR URL appears in its comments — the duplicate-PR
    protection the guard was built for must remain intact."""
    with kb.connect() as conn:
        t = kb.create_task(
            conn, title="Implement rate limiter", assignee="alice",
        )
        kb.add_comment(
            conn, t, "alice",
            "PR created: https://github.com/totemx-AI/subsidysmart/pull/42",
        )
        reason = kb.check_respawn_guard(conn, t)
    assert reason == "active_pr"


def test_dispatch_respawn_guard_defers_auth_error_without_auto_block(
    kanban_home, all_assignees_spawnable
):
    """dispatch_once defers (does NOT auto-block) a ready task whose last
    error is a blocker_auth.

    The old behaviour auto-blocked on first occurrence, which was too
    aggressive: a transient 429 rate-limit (which typically clears in
    seconds to minutes) would end up requiring manual unblock. The new
    behaviour defers the spawn this tick; the task stays in ``ready``
    and gets another chance next tick. If the auth error genuinely
    persists, the existing ``consecutive_failures`` circuit breaker
    will auto-block via the normal failure-limit path.
    """
    spawned_ids = []

    def fake_spawn(task, workspace):
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="quota-storm", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("rate limit exceeded: 429 Too Many Requests", t),
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    # Critical: task is NOT auto-blocked on first occurrence.
    assert t not in res.auto_blocked, (
        f"blocker_auth should defer, not auto-block on first occurrence; "
        f"got auto_blocked={res.auto_blocked!r}"
    )
    # It IS recorded as respawn_guarded with the reason.
    assert (t, "blocker_auth") in res.respawn_guarded, (
        f"expected (task_id, 'blocker_auth') in respawn_guarded; "
        f"got {res.respawn_guarded!r}"
    )
    # And it's NOT spawned this tick.
    assert t not in spawned_ids
    # Status stays ``ready`` so a future tick (or operator action) can
    # retry without manual unblock.
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"


def test_dispatch_respawn_guard_skips_recent_success(
    kanban_home, all_assignees_spawnable
):
    """dispatch_once skips (but does not block) a task with a recent completed run."""
    spawned_ids = []

    def fake_spawn(task, workspace):
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="recent-winner", assignee="alice")
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, now - 300, now - 60),
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert (t, "recent_success") in res.respawn_guarded
    assert t not in spawned_ids
    assert t not in res.auto_blocked
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"  # not blocked, just skipped


def test_dispatch_respawn_guard_skips_active_pr(
    kanban_home, all_assignees_spawnable
):
    """dispatch_once skips (but does not block) a task with an active PR comment."""
    spawned_ids = []

    def fake_spawn(task, workspace):
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="has-pr", assignee="alice")
        kb.add_comment(
            conn, t, "alice",
            "Opened https://github.com/totemx-AI/subsidysmart/pull/99",
        )
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert (t, "active_pr") in res.respawn_guarded
    assert t not in spawned_ids
    assert t not in res.auto_blocked
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"


def test_dispatch_respawn_guard_dry_run_no_auto_block(
    kanban_home, all_assignees_spawnable
):
    """In dry_run mode, blocker_auth tasks are recorded in respawn_guarded (not auto-blocked)."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="dry-quota", assignee="alice")
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("quota exceeded", t),
        )
        res = kb.dispatch_once(conn, dry_run=True)

    assert (t, "blocker_auth") in res.respawn_guarded
    assert t not in res.auto_blocked
    with kb.connect() as conn:
        assert kb.get_task(conn, t).status == "ready"  # dry_run: no writes


def test_dispatch_respawn_guard_allows_clean_task(
    kanban_home, all_assignees_spawnable
):
    """A task with no guard triggers is spawned normally."""
    spawned_ids = []

    def fake_spawn(task, workspace):
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        t = kb.create_task(conn, title="clean-task", assignee="alice")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert t in spawned_ids
    assert not res.respawn_guarded
    assert t not in res.auto_blocked


def test_dispatch_respawn_guard_emits_event_for_skipped_task(
    kanban_home, all_assignees_spawnable
):
    """dispatch_once emits a respawn_guarded task_event so operators can diagnose stuck-ready tasks."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="event-check", assignee="alice")
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_runs (task_id, status, outcome, started_at, ended_at) "
            "VALUES (?, 'done', 'completed', ?, ?)",
            (t, now - 300, now - 60),
        )
        kb.dispatch_once(conn, spawn_fn=lambda task, ws: None)
        events = kb.list_events(conn, t)

    kinds = [e.kind for e in events]
    assert "respawn_guarded" in kinds
    guarded_evt = next(e for e in events if e.kind == "respawn_guarded")
    # Event.payload is already parsed as a dict by list_events.
    assert isinstance(guarded_evt.payload, dict)
    assert guarded_evt.payload.get("reason") == "recent_success"


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------







def test_worktree_no_path_anchors_on_board_default_workdir(kanban_home, tmp_path):
    """A worktree task created with no explicit path inherits the board's
    default_workdir as its anchor and materializes a per-task linked worktree
    at ``<repo>/.worktrees/<id>`` — NOT the dispatcher's CWD, and NOT the
    shared default_workdir verbatim (which would collapse every task into one
    directory)."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("wt-default-board", default_workdir=str(repo))
    with kb.connect(board="wt-default-board") as conn:
        t = kb.create_task(
            conn, title="ship", workspace_kind="worktree", board="wt-default-board"
        )
        task = kb.get_task(conn, t)
        assert task is not None
        ws = kb.resolve_workspace(task, board="wt-default-board")

    expected = repo / ".worktrees" / t
    assert ws == expected
    assert ws.exists()
    assert ws != repo  # not the shared default verbatim


def test_worktree_no_path_subdir_board_default_persists_repo_root_anchor(
    kanban_home, tmp_path
):
    """A board default_workdir may be a package/subdir inside the repo.

    Persist the resolved repo root, not the raw subdir, so dispatch stays
    path-stable even if the board default changes before the task is claimed.
    """
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    other_repo = tmp_path / "other"
    _init_git_repo(other_repo)
    subdir = repo / "packages" / "core"
    subdir.mkdir(parents=True)
    kb.create_board("wt-subdir-default-board", default_workdir=str(subdir))
    with kb.connect(board="wt-subdir-default-board") as conn:
        tid = kb.create_task(
            conn,
            title="ship",
            workspace_kind="worktree",
            board="wt-subdir-default-board",
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(repo)
    kb.write_board_metadata("wt-subdir-default-board", default_workdir=str(other_repo))
    ws = kb.resolve_workspace(task, board="wt-subdir-default-board")
    assert ws == repo / ".worktrees" / tid
    assert ws != subdir


def test_worktree_no_path_no_board_default_raises(kanban_home, tmp_path, monkeypatch):
    """A worktree task with neither an explicit workspace_path nor a board
    default_workdir is un-spawnable, so create_task must fail LOUDLY at create
    time rather than storing a 'ready' row that burns its retries on
    spawn_failed at dispatch (the silent-zombie bug, t_c7b4b1a6).

    The raise happens inside the write_txn, so NO orphan row is left behind.
    """
    # Park the dispatcher CWD inside a real git repo so the OLD cwd-anchored
    # code would have "succeeded" — proving the new code does NOT use cwd.
    decoy_repo = tmp_path / "decoy"
    _init_git_repo(decoy_repo)
    monkeypatch.chdir(decoy_repo)
    with kb.connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="requires a workspace_path"):
            kb.create_task(conn, title="ship", workspace_kind="worktree")
        # The txn aborted cleanly — no orphan row.
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert after == before


def test_dir_no_path_no_board_default_raises(kanban_home):
    """The same fail-fast guard applies to workspace_kind='dir'."""
    with kb.connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="requires a workspace_path"):
            kb.create_task(conn, title="ship", workspace_kind="dir")
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert after == before


def test_worktree_explicit_path_succeeds_control(kanban_home, tmp_path):
    """Control for the fail-fast guard: the SAME call WITH an explicit
    worktree path succeeds and stores the row."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="ship",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "worktree"
    assert task.workspace_path == str(repo)


# ---------------------------------------------------------------------------
# Auto-derived babysit idempotency key (stop duplicate pr-babysitter tickets)
# ---------------------------------------------------------------------------

def _set_origin_remote(repo: Path, slug: str) -> None:
    """Point ``repo``'s origin remote at a github ``owner/repo`` slug so
    ``_slug_from_git_remote`` can resolve it."""
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin",
         f"https://github.com/{slug}.git"],
        check=True, capture_output=True, text=True,
    )


def test_babysit_same_pr_same_repo_dedups(kanban_home, tmp_path):
    """Two pr-babysitter creates for the same PR #70 in the same repo with NO
    explicit key dedup to ONE row — the second returns the first task's id.

    Fails before the auto-derive change (two distinct rows).
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="Babysit hermes-agent PR #70",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        second = kb.create_task(
            conn,
            title="hermes-agent#70: re-check the flaky CI",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        assert second == first
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE assignee = 'pr-babysitter'"
        ).fetchone()[0]
    assert rows == 1
    with kb.connect() as conn:
        task = kb.get_task(conn, first)
    assert task is not None
    assert task.idempotency_key == "babysit:exiao/hermes-agent#70"


def test_babysit_dedups_via_board_default_workdir(kanban_home, tmp_path):
    """Two pr-babysitter worktree creates for the same PR that supply NO
    explicit ``workspace_path`` and rely on the board ``default_workdir`` to
    anchor the repo still dedup to ONE row.

    This is the common ``workspace_kind='worktree'`` + board-default create
    path. The babysit key must be derived AFTER the board default resolves the
    repo path — otherwise ``workspace_path`` is None at derive time, the key is
    left unset, and the two creates insert two non-idempotent rows.

    Fails before the move (key derived too early → 2 rows).
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    kb.create_board("babysit-default-board", default_workdir=str(repo))
    with kb.connect(board="babysit-default-board") as conn:
        first = kb.create_task(
            conn,
            title="Babysit hermes-agent PR #70",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            board="babysit-default-board",
        )
        second = kb.create_task(
            conn,
            title="hermes-agent#70: re-check the flaky CI",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            board="babysit-default-board",
        )
        assert second == first
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE assignee = 'pr-babysitter'"
        ).fetchone()[0]
        task = kb.get_task(conn, first)
    assert rows == 1
    assert task is not None
    assert task.idempotency_key == "babysit:exiao/hermes-agent#70"


def test_slug_from_git_remote_skips_missing_path(tmp_path):
    """``_slug_from_git_remote`` returns None without reading a remote when the
    path has no existing git-repo ancestor or is not a directory (gemini high
    finding — avoid a guaranteed-to-fail subprocess)."""
    missing = tmp_path / "does-not-exist"
    assert kb._slug_from_git_remote(str(missing)) is None
    a_file = tmp_path / "afile"
    a_file.write_text("x", encoding="utf-8")
    assert kb._slug_from_git_remote(str(a_file)) is None
    assert kb._slug_from_git_remote(None) is None
    assert kb._slug_from_git_remote("") is None


def test_slug_from_git_remote_resolves_pending_worktree_target(tmp_path):
    """A not-yet-created worktree target under a real repo
    (``<repo>/.worktrees/pr70``) still resolves the repo slug by walking to the
    nearest existing git ancestor (Codex P2 — pending worktree forms must not
    bypass dedup)."""
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    pending = repo / ".worktrees" / "pr70"  # does not exist on disk yet
    assert not pending.exists()
    assert kb._slug_from_git_remote(str(pending)) == "exiao/hermes-agent"


def test_slug_from_git_remote_ignores_non_github_remotes(tmp_path):
    """A non-GitHub origin (gitlab/bitbucket/self-hosted) yields no slug — the
    host-less ``babysit:`` key is GitHub-specific, so a gitlab ``owner/repo``
    must not be allowed to cross-dedup a github repo of the same name."""
    for url in (
        "https://gitlab.com/Owner/Repo.git",
        "git@bitbucket.org:owner/repo.git",
        "https://git.example.com/owner/repo.git",
    ):
        repo = tmp_path / url.replace("/", "_").replace(":", "_")
        _init_git_repo(repo)
        subprocess.run(
            ["git", "-C", str(repo), "remote", "add", "origin", url],
            check=True, capture_output=True, text=True,
        )
        assert kb._slug_from_git_remote(str(repo)) is None
    # An scp-style GitHub remote still resolves.
    gh = tmp_path / "gh"
    _init_git_repo(gh)
    subprocess.run(
        ["git", "-C", str(gh), "remote", "add", "origin", "git@github.com:exiao/hermes-agent.git"],
        check=True, capture_output=True, text=True,
    )
    assert kb._slug_from_git_remote(str(gh)) == "exiao/hermes-agent"


def test_slug_from_git_remote_accepts_authenticated_https(tmp_path):
    """An authenticated HTTPS origin carrying userinfo before the host (the
    ``https://x-access-token:TOKEN@github.com/owner/repo.git`` form used by the
    CI/private-repo push path) still resolves the slug — otherwise a
    ``PR #<n>``-only babysit task in that setup never gets its canonical key and
    duplicate tickets slip through.
    """
    for url in (
        "https://x-access-token:ghs_SECRET@github.com/exiao/hermes-agent.git",
        "https://exiao:ghp_TOKEN@github.com/exiao/hermes-agent.git",
        "ssh://git@github.com/exiao/hermes-agent.git",
    ):
        repo = tmp_path / url.replace("/", "_").replace(":", "_").replace("@", "_")
        _init_git_repo(repo)
        subprocess.run(
            ["git", "-C", str(repo), "remote", "add", "origin", url],
            check=True, capture_output=True, text=True,
        )
        assert kb._slug_from_git_remote(str(repo)) == "exiao/hermes-agent"
    # Userinfo must not loosen the host check — an authenticated NON-github
    # remote still yields no slug.
    other = tmp_path / "authed_gitlab"
    _init_git_repo(other)
    subprocess.run(
        ["git", "-C", str(other), "remote", "add", "origin",
         "https://x-access-token:SECRET@gitlab.com/exiao/hermes-agent.git"],
        check=True, capture_output=True, text=True,
    )
    assert kb._slug_from_git_remote(str(other)) is None


def test_babysit_pending_worktree_target_dedups(kanban_home, tmp_path):
    """Two pr-babysitter creates for the same PR whose ``workspace_path`` is a
    pending ``<repo>/.worktrees/<x>`` target (not yet materialized) still dedup
    to one row — the slug resolves from the parent repo's remote."""
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="Babysit hermes-agent PR #70",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo / ".worktrees" / "a"),
        )
        second = kb.create_task(
            conn,
            title="hermes-agent#70: re-check",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo / ".worktrees" / "b"),
        )
        assert second == first
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE assignee = 'pr-babysitter'"
        ).fetchone()[0]
        task = kb.get_task(conn, first)
    assert rows == 1
    assert task is not None
    assert task.idempotency_key == "babysit:exiao/hermes-agent#70"


def test_babysit_blank_idempotency_key_still_dedups(kanban_home, tmp_path):
    """A caller that passes ``idempotency_key=""`` (blank/whitespace) for a
    pr-babysitter task must NOT bypass the auto-derive + dedup: the blank key is
    normalized to None, the key is derived, and two creates for the same PR
    collapse to one row.

    Fails before normalization (blank key skips derive, falsy key skips dedup
    lookup → 2 rows).
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="Babysit hermes-agent PR #70",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo),
            idempotency_key="   ",
        )
        second = kb.create_task(
            conn,
            title="hermes-agent#70: re-check",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo),
            idempotency_key="",
        )
        assert second == first
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE assignee = 'pr-babysitter'"
        ).fetchone()[0]
        task = kb.get_task(conn, first)
    assert rows == 1
    assert task is not None
    assert task.idempotency_key == "babysit:exiao/hermes-agent#70"


def test_babysit_pull_url_number_wins_over_title_ref():
    """When a pull URL is present, its PR number is authoritative for the key —
    a stray ``#<n>`` (e.g. an issue ref) in the title must NOT override it and
    pair the URL's slug with the wrong PR number."""
    # Title carries an issue-style ``#123`` but the body links PR #70; the key
    # must be the URL's owner/repo#70, not owner/repo#123.
    key = kb._derive_babysit_idempotency_key(
        "Babysit fix #123",
        "see https://github.com/org/repo/pull/70 for the change",
        None,
    )
    assert key == "babysit:org/repo#70"
    # No URL → fall back to the title ``#<n>`` (detector convention) using the
    # workspace git remote for the slug. Covered elsewhere; here assert the
    # no-URL/no-title case still returns None.
    assert kb._derive_babysit_idempotency_key("no pr ref", "", None) is None


def test_babysit_pull_url_host_anchored_and_git_suffix_stripped():
    """The pull-URL slug source is GitHub-host-anchored and ``.git``-stripped so
    it (a) can't be spoofed by a look-alike host and (b) canonicalizes to the
    same slug the remote-derived path produces for one repo.
    """
    # Look-alike host must NOT match — no other PR signal, so no key derives.
    assert (
        kb._derive_babysit_idempotency_key(
            "babysit", "see https://notgithub.com/org/repo/pull/70", None
        )
        is None
    )
    assert (
        kb._derive_babysit_idempotency_key(
            "babysit", "see https://evilgithub.com/org/repo/pull/70", None
        )
        is None
    )
    # A github SUBDOMAIN (e.g. GitHub Enterprise ghe.github.com) must NOT match
    # either — it would derive the same key as the public repo and collide.
    assert (
        kb._derive_babysit_idempotency_key(
            "babysit", "see https://ghe.github.com/org/repo/pull/70", None
        )
        is None
    )
    # A real github.com URL still works.
    assert (
        kb._derive_babysit_idempotency_key(
            "babysit", "https://github.com/org/repo/pull/70", None
        )
        == "babysit:org/repo#70"
    )
    # A ``.git``-bearing URL canonicalizes to the SAME slug as the bare form, so
    # it dedups against the remote-derived key for the same repo.
    assert (
        kb._derive_babysit_idempotency_key(
            "babysit", "https://github.com/Org/Repo.git/pull/70", None
        )
        == "babysit:org/repo#70"
    )


def test_babysit_owner_repo_ref_form_derives_key():
    """A bare ``owner/repo#<n>`` reference (the documented babysit anchor form,
    and a scratch ``kanban_create`` handoff's only PR signal) pins both the slug
    and the PR number — no workspace remote or pull URL needed."""
    # Title-only ref, no workspace path (scratch handoff).
    key = kb._derive_babysit_idempotency_key(
        "exiao/hermes-agent#73: fix(kanban): auto-derive babysit key",
        None,
        None,
    )
    assert key == "babysit:exiao/hermes-agent#73"
    # Ref in the body works too.
    key2 = kb._derive_babysit_idempotency_key(
        "Re-verify the babysit PR",
        "anchored to cpe-research/research-agent#801",
        None,
    )
    assert key2 == "babysit:cpe-research/research-agent#801"
    # A TITLE owner/repo#n anchor (the card's own subject) outranks a pull URL
    # that sits only in the BODY — a body link is a secondary/related mention
    # and must not override the card's own PR (else retries wouldn't dedup).
    key3 = kb._derive_babysit_idempotency_key(
        "owner/other#5 mention",
        "related PR https://github.com/org/repo/pull/70",
        None,
    )
    assert key3 == "babysit:owner/other#5"
    # With NO title anchor, a body pull URL is honored.
    key4 = kb._derive_babysit_idempotency_key(
        "Babysit the PR",
        "real PR https://github.com/org/repo/pull/70",
        None,
    )
    assert key4 == "babysit:org/repo#70"


def test_babysit_pr_number_in_body_with_workspace_slug_derives_key(tmp_path):
    """An orchestrator handoff with a short title (``Babysit PR``) and the
    ``PR #<n>`` anchor in the BODY, with the slug supplied by the workspace git
    remote, still derives a key. Without searching the body for ``PR #<n>`` the
    key would stay None and repeated creates would insert duplicate rows.
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    key = kb._derive_babysit_idempotency_key(
        "Babysit PR",
        "Please watch PR #70 and keep it green.",
        str(repo),
    )
    assert key == "babysit:exiao/hermes-agent#70"
    # A bare ``#<n>`` in free-form body prose (no ``PR`` prefix) must NOT be
    # keyed on — too likely an unrelated reference — so the title-only bare
    # fallback leaves the key None when the title has no number.
    none_key = kb._derive_babysit_idempotency_key(
        "Babysit the thing",
        "unrelated note mentioning #999 somewhere",
        str(repo),
    )
    assert none_key is None


def test_babysit_title_workspace_pr_outranks_body_shorthand_ref(tmp_path):
    """A card with a real title (``Babysit PR #70``) + a workspace remote keys
    on ITS OWN PR (#70 in the workspace repo), even when the body mentions an
    unrelated ``owner/repo#5`` shorthand. The body ref must not override the
    card's title/workspace signal, or retries for the real PR would not dedup
    and could collide with an unrelated babysitter card.
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    key = kb._derive_babysit_idempotency_key(
        "Babysit PR #70",
        "context: related to other/project#5 from last week",
        str(repo),
    )
    assert key == "babysit:exiao/hermes-agent#70"
    # A body ``owner/repo#<n>`` ref is still honored as the LAST resort when the
    # card carries no title/workspace PR signal of its own.
    fallback = kb._derive_babysit_idempotency_key(
        "Re-verify the babysit task",
        "anchored to cpe-research/research-agent#801",
        None,
    )
    assert fallback == "babysit:cpe-research/research-agent#801"


def test_babysit_title_pr_anchor_outranks_body_pull_url(tmp_path):
    """A card whose TITLE names its own PR (``Babysit PR #70`` + workspace
    remote, or ``owner/repo#70`` in the title) keys on that PR even when the
    BODY contains a github pull URL for a related/different PR. A body link is a
    secondary mention and must not override the card's own subject, or retries
    for the real PR won't dedup and could return an unrelated existing task.
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    # Title PR #70 + workspace slug, body links a DIFFERENT repo's PR.
    key = kb._derive_babysit_idempotency_key(
        "Babysit PR #70",
        "see also https://github.com/other/project/pull/5 for context",
        str(repo),
    )
    assert key == "babysit:exiao/hermes-agent#70"
    # Title owner/repo#70 anchor, body links a different PR.
    key2 = kb._derive_babysit_idempotency_key(
        "Babysit exiao/hermes-agent#70",
        "related: https://github.com/other/project/pull/5",
        None,
    )
    assert key2 == "babysit:exiao/hermes-agent#70"


def test_babysit_body_pull_url_outranks_bare_title_number(tmp_path):
    """A full github pull URL in the BODY outranks a BARE ``#<n>`` in the title:
    a bare title number may be an ISSUE reference (``Babysit fix #123``), while
    the pull URL unambiguously pins both the slug and the PR. An explicit
    ``PR #<n>`` in the title still wins (it names the card's own PR), but a bare
    ``#<n>`` does not.
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    # Bare title #123 (likely an issue) + a body pull URL for PR #70 → key on
    # the URL's PR, not the ambiguous bare title number.
    key = kb._derive_babysit_idempotency_key(
        "Babysit fix #123",
        "the PR is https://github.com/org/repo/pull/70",
        str(repo),
    )
    assert key == "babysit:org/repo#70"
    # But an explicit title ``PR #<n>`` still outranks a body URL.
    key2 = kb._derive_babysit_idempotency_key(
        "Babysit PR #70",
        "compare against https://github.com/org/repo/pull/5",
        str(repo),
    )
    assert key2 == "babysit:exiao/hermes-agent#70"


def test_babysit_body_pr_number_outranks_bare_title_number(tmp_path):
    """An explicit ``PR #<n>`` in the BODY outranks a BARE ``#<n>`` in the title
    (same reasoning as the body-URL case): the body ``PR #70`` is an explicit PR
    signal, while a bare title ``#123`` may be an issue ref. Key on the body PR.
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    key = kb._derive_babysit_idempotency_key(
        "Babysit fix #123",
        "the actual PR #70 needs watching",
        str(repo),
    )
    assert key == "babysit:exiao/hermes-agent#70"


def test_babysit_body_owner_repo_ref_outranks_bare_title_number(tmp_path):
    """A BODY ``owner/repo#<n>`` ref (explicit, pins both pieces) outranks a
    BARE ``#<n>`` in the title (which may be an issue number): the bare title
    number is the ONLY ambiguous signal, so it is strictly last — every explicit
    signal, title or body, wins over it.
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    key = kb._derive_babysit_idempotency_key(
        "Babysit fix #123",
        "anchored to other/project#70",
        str(repo),
    )
    assert key == "babysit:other/project#70"


def test_babysit_default_assignee_derives_key_and_dedups(kanban_home, tmp_path, monkeypatch):
    """A card created WITHOUT an explicit assignee under
    ``kanban.default_assignee = pr-babysitter`` (the dispatcher applies the
    default later) is still keyed at create time, so two such creates for the
    same PR dedup to one row instead of both inserting keyless and recreating
    the duplicate-ticket path.
    """
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    # Operator default routes unassigned cards to pr-babysitter.
    monkeypatch.setattr(kb, "_default_assignee", lambda: "pr-babysitter")
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="Babysit PR #70",
            assignee=None,
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        second = kb.create_task(
            conn,
            title="re-check PR #70",
            assignee=None,
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        assert second == first
        task = kb.get_task(conn, first)
    assert task is not None
    assert task.idempotency_key == "babysit:exiao/hermes-agent#70"


def test_babysit_scratch_owner_repo_handoff_dedups(kanban_home):
    """Two scratch pr-babysitter creates that name the PR as ``owner/repo#<n>``
    (no workspace_path) dedup to one row — the ref form alone keys them."""
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="exiao/hermes-agent#73: fix babysit key",
            assignee="pr-babysitter",
        )
        second = kb.create_task(
            conn,
            title="re-check exiao/hermes-agent#73",
            assignee="pr-babysitter",
        )
        assert second == first
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE assignee = 'pr-babysitter'"
        ).fetchone()[0]
        task = kb.get_task(conn, first)
    assert rows == 1
    assert task is not None
    assert task.idempotency_key == "babysit:exiao/hermes-agent#73"


def test_babysit_slug_case_insensitive_dedups(kanban_home):
    """Two scratch pr-babysitter creates naming the same PR with different
    owner/repo casing dedup to one row — GitHub slugs are case-insensitive, so
    the key lowercases the slug."""
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="Babysit NousResearch/hermes-agent#70",
            assignee="pr-babysitter",
        )
        second = kb.create_task(
            conn,
            title="re-check nousresearch/hermes-agent#70",
            assignee="pr-babysitter",
        )
        assert second == first
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE assignee = 'pr-babysitter'"
        ).fetchone()[0]
        task = kb.get_task(conn, first)
    assert rows == 1
    assert task is not None
    assert task.idempotency_key == "babysit:nousresearch/hermes-agent#70"


def test_babysit_explicit_mixed_case_key_canonicalized():
    """An explicit ``babysit:<Owner>/<Repo>#<n>`` key (e.g. the detector's,
    built from a mixed-case git remote) is canonicalized to a lowercase slug so
    it matches a later auto-derived key for the same PR."""
    assert (
        kb._canonicalize_babysit_key("babysit:NousResearch/hermes-agent#70")
        == "babysit:nousresearch/hermes-agent#70"
    )
    # Non-babysit keys and unparseable values pass through verbatim.
    assert kb._canonicalize_babysit_key("custom:thing#1") == "custom:thing#1"
    assert kb._canonicalize_babysit_key("babysit:noslug#1") == "babysit:noslug#1"
    assert kb._canonicalize_babysit_key(None) is None


def test_babysit_explicit_and_derived_key_dedup_across_case(kanban_home, tmp_path):
    """A detector row keyed with an explicit mixed-case slug and a later
    tool-derived create for the same PR (lowercased) dedup to one row."""
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    with kb.connect() as conn:
        first = kb.create_task(
            conn,
            title="Babysit hermes-agent PR #70",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo),
            idempotency_key="babysit:Exiao/Hermes-Agent#70",
        )
        # Auto-derived from the git remote → lowercase exiao/hermes-agent#70.
        second = kb.create_task(
            conn,
            title="hermes-agent#70: re-check",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        assert second == first
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE assignee = 'pr-babysitter'"
        ).fetchone()[0]
        task = kb.get_task(conn, first)
    assert rows == 1
    assert task is not None
    assert task.idempotency_key == "babysit:exiao/hermes-agent#70"


def test_legacy_mixed_case_babysit_key_migrated_and_dedups(kanban_home):
    """A row stored BEFORE slug-lowercasing (mixed-case key) is canonicalized
    by the migration pass so a later derived create for the same PR dedups to
    it instead of inserting a duplicate babysitter row.

    Reproduces the pre-existing-row case: the dedup lookup compares
    ``idempotency_key`` byte-for-byte, so without migrating the stored key a
    legacy ``babysit:NousResearch/hermes-agent#70`` row would never match a
    freshly-derived ``babysit:nousresearch/hermes-agent#70``.
    """
    # Seed a legacy row with a mixed-case key directly, bypassing create_task's
    # canonicalization (simulating a row written by an older version).
    with kb.connect() as conn:
        legacy = kb.create_task(
            conn,
            title="Babysit legacy PR",
            assignee="pr-babysitter",
        )
        conn.execute(
            "UPDATE tasks SET idempotency_key = ? WHERE id = ?",
            ("babysit:NousResearch/hermes-agent#70", legacy),
        )
        conn.commit()

    # Re-run the migration pass over the existing DB.
    kb.init_db()

    with kb.connect() as conn:
        migrated = conn.execute(
            "SELECT idempotency_key FROM tasks WHERE id = ?", (legacy,)
        ).fetchone()[0]
        assert migrated == "babysit:nousresearch/hermes-agent#70"

        # A later derived create for the same PR now finds the migrated row.
        second = kb.create_task(
            conn,
            title="re-check nousresearch/hermes-agent#70",
            assignee="pr-babysitter",
        )
        assert second == legacy
        rows = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE assignee = 'pr-babysitter'"
        ).fetchone()[0]
    assert rows == 1


def test_babysit_prefers_pr_ref_over_leading_issue_ref(tmp_path):
    """When a no-URL title names another ref before the PR (e.g. ``Babysit issue
    #123 for PR #70``), the key uses the ``PR #<n>`` number, not the leading
    bare ``#123``."""
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    key = kb._derive_babysit_idempotency_key(
        "Babysit issue #123 for PR #70",
        None,
        str(repo),
    )
    assert key == "babysit:exiao/hermes-agent#70"


def test_babysit_same_pr_different_repos_no_cross_dedup(kanban_home, tmp_path):
    """The SAME PR number #70 in two DIFFERENT repos creates two distinct
    tasks — PR numbers collide across repos, so a repo-less key must never
    cross-dedup."""
    repo_a = tmp_path / "hermes-agent"
    _init_git_repo(repo_a)
    _set_origin_remote(repo_a, "exiao/hermes-agent")
    repo_b = tmp_path / "research-agent"
    _init_git_repo(repo_b)
    _set_origin_remote(repo_b, "cpe-research/research-agent")
    with kb.connect() as conn:
        a = kb.create_task(
            conn,
            title="Babysit hermes-agent PR #70",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo_a),
        )
        b = kb.create_task(
            conn,
            title="Babysit research-agent PR #70",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo_b),
        )
        assert a != b
        ta = kb.get_task(conn, a)
        tb = kb.get_task(conn, b)
    assert ta is not None and tb is not None
    assert ta.idempotency_key == "babysit:exiao/hermes-agent#70"
    assert tb.idempotency_key == "babysit:cpe-research/research-agent#70"


def test_babysit_explicit_key_respected_unchanged(kanban_home, tmp_path):
    """An explicit idempotency_key passed by the caller (e.g. the detector
    cron) is stored verbatim and the auto-derive does NOT override it."""
    repo = tmp_path / "hermes-agent"
    _init_git_repo(repo)
    _set_origin_remote(repo, "exiao/hermes-agent")
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Babysit hermes-agent PR #70",
            assignee="pr-babysitter",
            workspace_kind="worktree",
            workspace_path=str(repo),
            idempotency_key="babysit:explicit/override#999",
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.idempotency_key == "babysit:explicit/override#999"


def test_babysit_unresolvable_leaves_key_none(kanban_home, tmp_path):
    """A pr-babysitter task whose title/body/workspace yield no (slug, pr)
    pair leaves the key None and still creates the task (no regression)."""
    # No PR number in the title and no resolvable repo slug (scratch workspace,
    # no git remote) → key cannot be derived.
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Babysit something with no PR reference",
            assignee="pr-babysitter",
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.idempotency_key is None


def test_worktree_explicit_non_repo_path_raises_at_create(kanban_home, tmp_path, monkeypatch):
    """Fail-fast (t_f19db0e0): a ``worktree`` task whose explicit path is a real
    directory that is NOT a git repo (e.g. an umbrella dir like
    ~/projects/content with no .git) must be rejected at CREATE time, not stored
    and then blocked after two spawn failures + a circuit-breaker trip.

    The path exists but neither it nor any ancestor is a git repo, so
    ``_resolve_worktree_workspace`` would raise at spawn. We surface that error
    here, inside the write_txn, leaving NO orphan row.
    """
    # A real, existing directory with no git anywhere up the tree. Put it under
    # its own isolated root so no ambient repo (e.g. the checkout running the
    # tests) is found by the upward walk.
    isolated_root = tmp_path / "no_git_root"
    not_a_repo = isolated_root / "content"
    not_a_repo.mkdir(parents=True)
    # Park the dispatcher CWD inside a real repo so a cwd-anchored shortcut
    # could not mask the bug.
    decoy_repo = tmp_path / "decoy"
    _init_git_repo(decoy_repo)
    monkeypatch.chdir(decoy_repo)
    with kb.connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="not .*inside a git repo"):
            kb.create_task(
                conn,
                title="ship",
                workspace_kind="worktree",
                workspace_path=str(not_a_repo),
            )
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert after == before  # txn aborted cleanly, no zombie row


def test_worktree_repo_root_path_accepted_at_create(kanban_home, tmp_path):
    """Accept case: an explicit path that IS a git repo root passes the new
    create-time guard and stores the row (the path the dispatcher later anchors
    a per-task ``<repo>/.worktrees/<id>`` on)."""
    repo = tmp_path / "real-repo"
    _init_git_repo(repo)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="ship",
            workspace_kind="worktree",
            workspace_path=str(repo),
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        # The stored row must actually resolve at spawn time (end-to-end check).
        ws = kb.resolve_workspace(task, conn=conn)
    assert task.workspace_path == str(repo)
    assert ws == repo / ".worktrees" / tid


def test_worktree_target_under_repo_accepted_at_create(kanban_home, tmp_path):
    """Accept case: an explicit target path that does not exist yet but whose
    parent lives inside a git repo (the ``<repo>/.worktrees/<id>`` materialize
    semantics) must NOT be rejected by the create-time guard."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "my-task"  # parent is inside the repo
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="ship",
            workspace_kind="worktree",
            workspace_path=str(target),
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_path == str(target)


def test_worktree_existing_repo_subdir_rejected_at_create(kanban_home, tmp_path):
    """Existing non-worktree dirs inside a repo are not valid worktree anchors."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    subdir = repo / "src"
    subdir.mkdir()

    with kb.connect() as conn:
        with pytest.raises(ValueError, match="not .*inside a git repo"):
            kb.create_task(
                conn,
                title="ship",
                workspace_kind="worktree",
                workspace_path=str(subdir),
            )


def test_worktree_existing_linked_worktree_subdir_rejected_at_create(
    kanban_home, tmp_path
):
    """Only the linked worktree checkout root is a valid existing anchor."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-b", "linked", str(linked), "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    subdir = linked / "pkg"
    subdir.mkdir()

    with kb.connect() as conn:
        ok = kb.create_task(
            conn,
            title="linked root",
            workspace_kind="worktree",
            workspace_path=str(linked),
        )
        assert kb.get_task(conn, ok) is not None
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="not .*inside a git repo"):
            kb.create_task(
                conn,
                title="linked subdir",
                workspace_kind="worktree",
                workspace_path=str(subdir),
            )
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert after == before


def test_worktree_missing_target_under_file_ancestor_rejected_at_create(
    kanban_home, tmp_path
):
    """A missing worktree target cannot be materialized below a file path."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / "README.md" / "wt"

    with kb.connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="not .*inside a git repo"):
            kb.create_task(
                conn,
                title="file ancestor",
                workspace_kind="worktree",
                workspace_path=str(target),
            )
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert after == before


def test_worktree_missing_target_skips_direct_git_probe(tmp_path, monkeypatch):
    """A missing target is resolved from ancestors without `git -C <missing>`."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "my-task"
    original_git_toplevel = kb._git_toplevel
    probed: list[Path] = []

    def tracking_git_toplevel(path: Path):
        probed.append(path)
        return original_git_toplevel(path)

    monkeypatch.setattr(kb, "_git_toplevel", tracking_git_toplevel)

    assert kb._worktree_path_resolvable(str(target)) is True
    assert target not in probed



def test_worktree_no_path_non_current_board_default_succeeds(kanban_home, tmp_path, monkeypatch):
    """Regression (Codex P2): create_task(board='target', workspace_kind='worktree')
    on a board with a default_workdir must NOT falsely raise just because the
    *active* board differs from the target board. The board-default lookup must
    consult the explicitly-passed ``board``, not get_current_board()."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("target-board", default_workdir=str(repo))
    # Pin the ACTIVE board to a different board with NO default_workdir, so a
    # get_current_board()-based lookup would resolve nothing and falsely raise.
    kb.create_board("other-active-board")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "other-active-board")
    with kb.connect(board="target-board") as conn:
        tid = kb.create_task(
            conn, title="ship", workspace_kind="worktree", board="target-board"
        )
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.workspace_kind == "worktree"
    # Anchored on the target board's default_workdir, not raised.
    assert task.workspace_path == str(repo)


def test_worktree_blank_path_normalizes_to_none_and_resolves_board_default(
    kanban_home, tmp_path
):
    """Regression (Codex P2a): a JSON/tool caller that sends
    ``workspace_path=""`` (or whitespace) with workspace_kind='worktree' on a
    board that HAS a default_workdir must behave exactly like passing None —
    the blank string is normalized to None at the top of create_task so (1) the
    board-default fallback fires and (2) the fail-fast guard does NOT reject it.
    The stored row inherits the board default and dispatch anchors a per-task
    worktree at ``<repo>/.worktrees/<id>``.
    """
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("blank-wt-board", default_workdir=str(repo))
    with kb.connect(board="blank-wt-board") as conn:
        for blank in ("", "   ", "\t\n"):
            tid = kb.create_task(
                conn,
                title="ship",
                workspace_kind="worktree",
                workspace_path=blank,
                board="blank-wt-board",
            )
            task = kb.get_task(conn, tid)
            assert task is not None
            # Blank normalized to None -> inherited the board default.
            assert task.workspace_path == str(repo)
            ws = kb.resolve_workspace(task, board="blank-wt-board")
            assert ws == repo / ".worktrees" / tid


def test_worktree_blank_path_no_board_default_still_raises(kanban_home, tmp_path, monkeypatch):
    """Control for P2a: a blank-string worktree path with NO board default is
    still genuinely unresolvable, so the guard must raise (the blank string is
    normalized to None, then the guard fires exactly as for an explicit None).
    """
    decoy_repo = tmp_path / "decoy"
    _init_git_repo(decoy_repo)
    monkeypatch.chdir(decoy_repo)
    with kb.connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="requires a workspace_path"):
            kb.create_task(
                conn, title="ship", workspace_kind="worktree", workspace_path="   "
            )
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert after == before


def test_worktree_workspace_explicit_target_materializes_linked_worktree(kanban_home, tmp_path):
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    target = repo / ".worktrees" / "custom-task"
    branch = "wt/custom-task"
    with kb.connect() as conn:
        t = kb.create_task(
            conn,
            title="ship",
            workspace_kind="worktree",
            workspace_path=str(target),
            branch_name=branch,
        )
        task = kb.get_task(conn, t)
        assert task is not None
        ws = kb.resolve_workspace(task)

    assert ws == target
    assert ws.exists()
    repo_common = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ws_common = subprocess.run(
        ["git", "-C", str(ws), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert ws_common == repo_common
    listed = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert f"worktree {target}" in listed
    assert f"branch refs/heads/{branch}" in listed


# ---------------------------------------------------------------------------
# Scratch cleanup containment (#28818)
# ---------------------------------------------------------------------------



def test_complete_task_persists_scratch_artifacts_before_cleanup(kanban_home):
    """Completion artifacts from scratch workspaces survive workspace cleanup."""
    with kb.connect() as conn:
        t = kb.create_task(conn, title="render chart")
        task = kb.get_task(conn, t)
        ws = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, t, ws)
        artifact = ws / "chart.png"
        artifact.write_bytes(b"png-bytes")

        assert kb.complete_task(
            conn,
            t,
            result="ok",
            metadata={"artifacts": [str(artifact)]},
        )

        completed = [e for e in kb.list_events(conn, t) if e.kind == "completed"][-1]
        persisted = Path(completed.payload["artifacts"][0])
        run = kb.latest_run(conn, t)

    assert not ws.exists(), "scratch workspace should still be cleaned up"
    assert persisted.exists(), "artifact copy should survive scratch cleanup"
    assert persisted.parent == kb.task_attachments_dir(t)
    assert persisted.name == "chart.png"
    assert persisted.read_bytes() == b"png-bytes"
    assert str(persisted) != str(artifact)
    assert run is not None
    assert run.metadata["artifacts"] == [str(persisted)]
    with kb.connect() as conn:
        attachments = kb.list_attachments(conn, t)
    assert [(a.filename, a.stored_path) for a in attachments] == [
        ("chart.png", str(persisted.resolve()))
    ]




# ---------------------------------------------------------------------------
# Deferred scratch cleanup for parent/child handoff (#33774)
# ---------------------------------------------------------------------------




def test_dir_child_completion_unblocks_deferred_scratch_parent(kanban_home, tmp_path):
    """A non-scratch ('dir') child completing must still sweep its scratch parent.

    Regression for the gap where ``_cleanup_workspace`` returned early for a
    non-scratch task and never ran the parent sweep — leaking the parent's
    deferred scratch dir forever.
    """
    child_dir = tmp_path / "persistent-child"
    child_dir.mkdir()
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="scratch parent")
        child = kb.create_task(
            conn, title="dir child", workspace_kind="dir",
            workspace_path=str(child_dir),
        )
        kb.link_tasks(conn, parent, child)
        p_task = kb.get_task(conn, parent)
        parent_ws = kb.resolve_workspace(p_task)
        kb.set_workspace_path(conn, parent, parent_ws)

        kb.complete_task(conn, parent, result="handoff")
        assert parent_ws.exists(), "deferred while dir child active"

        kb.complete_task(conn, child, result="built")

    assert not parent_ws.exists(), (
        "A 'dir' child completing must trigger the parent scratch sweep"
    )
    assert child_dir.exists(), "Non-scratch 'dir' child workspace is never deleted"




def test_is_managed_scratch_path_rejects_kanban_metadata_subtrees(kanban_home):
    """Hermes' own DB/metadata/log subtrees under ``<kanban_home>/kanban`` are NOT managed.

    Regression guard for the Copilot finding on #28819: a scratch task whose
    ``workspace_path`` was mis-set to the kanban home, the logs dir, or a
    board's metadata dir (i.e. the board root itself, not its ``workspaces/``
    child) must be refused. Without this, the containment check would happily
    ``shutil.rmtree`` Hermes' DB/metadata/logs on task completion.
    """
    kanban_root = kanban_home / "kanban"
    kanban_root.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(kanban_root)

    logs_dir = kanban_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(logs_dir)

    board_root = kanban_root / "boards" / "my-board"
    board_root.mkdir(parents=True, exist_ok=True)
    # The board root itself is NOT a managed scratch dir — only the
    # ``workspaces/`` child (and its descendants) are.
    assert not kb._is_managed_scratch_path(board_root)

    # Sibling subtrees of ``workspaces/`` under a board (e.g. its kanban.db
    # or board.json living next to ``workspaces/``) are also not managed.
    board_logs = board_root / "logs"
    board_logs.mkdir(parents=True, exist_ok=True)
    assert not kb._is_managed_scratch_path(board_logs)

    # Now create the board's workspaces dir and a task scratch dir under it —
    # the latter is the only thing the guard should allow.
    board_workspaces = board_root / "workspaces"
    board_workspaces.mkdir(parents=True, exist_ok=True)
    # The workspaces root itself is also NOT managed — deleting it would
    # wipe every task's scratch dir at once.
    assert not kb._is_managed_scratch_path(board_workspaces)
    task_dir = board_workspaces / "task-42"
    task_dir.mkdir(parents=True, exist_ok=True)
    assert kb._is_managed_scratch_path(task_dir)


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------









# ---------------------------------------------------------------------------
# Originating session id (ACP propagation)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Shared-board path resolution (issue #19348)
#
# The kanban board is a cross-profile coordination primitive: a worker
# spawned with `hermes -p <profile>` must read/write the same kanban.db
# as the dispatcher that claimed the task. These tests exercise the
# path-resolution layer directly and would have caught the regression
# where `kanban_db_path()` resolved to the active profile's HERMES_HOME.
# ---------------------------------------------------------------------------

class TestSharedBoardPaths:
    """`kanban_home`/`kanban_db_path`/`workspaces_root`/`worker_log_path`
    must anchor at the **shared root**, not the active profile's HERMES_HOME."""

    def _set_home(self, monkeypatch, tmp_path, hermes_home):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)


    def test_profile_worker_resolves_to_shared_root(
        self, tmp_path, monkeypatch
    ):
        # Reproduces the bug: dispatcher uses ~/.hermes/kanban.db,
        # worker spawned with -p <profile> previously resolved to
        # ~/.hermes/profiles/<profile>/kanban.db. After the fix both
        # converge on ~/.hermes/kanban.db.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)
        self._set_home(monkeypatch, tmp_path, profile_home)

        # All four resolvers must anchor at the shared root, not the
        # profile-local HERMES_HOME.
        assert kb.kanban_home() == default_home
        assert kb.kanban_db_path() == default_home / "kanban.db"
        assert kb.workspaces_root() == default_home / "kanban" / "workspaces"
        assert (
            kb.worker_log_path("t_0d214f19")
            == default_home / "kanban" / "logs" / "t_0d214f19.log"
        )

        # Sanity: the profile-local path that used to be returned is
        # explicitly NOT what we resolve to anymore.
        assert kb.kanban_db_path() != profile_home / "kanban.db"






    def test_dispatcher_and_worker_share_a_real_database(
        self, tmp_path, monkeypatch
    ):
        # Belt-and-suspenders: round-trip a task across the two
        # HERMES_HOME perspectives via a real SQLite file. Without the
        # fix the worker would open a different file and see no rows.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        profile_home = default_home / "profiles" / "nehemiahkanban"
        profile_home.mkdir(parents=True)

        # Dispatcher creates the board and a task.
        self._set_home(monkeypatch, tmp_path, default_home)
        kb.init_db()
        with kb.connect() as conn:
            task_id = kb.create_task(conn, title="cross-profile")

        # Worker switches to the profile HERMES_HOME and reads.
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        with kb.connect() as conn:
            task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.title == "cross-profile"




    def test_dispatcher_spawn_injects_kanban_paths_without_stale_session(
        self, tmp_path, monkeypatch
    ):
        # The dispatcher must pin board paths while stripping any unrelated
        # HERMES_SESSION_* identity inherited from the long-lived gateway.
        # The one exception is HERMES_SESSION_SOURCE, which the dispatcher
        # re-sets to its own `kanban` tag AFTER the strip — a value it owns,
        # never one inherited from whatever the gateway last routed.
        default_home = tmp_path / ".hermes"
        default_home.mkdir()
        self._set_home(monkeypatch, tmp_path, default_home)

        from gateway import session_context as sc

        # A dispatcher can launch before the gateway binds its first session.
        monkeypatch.setattr(sc, "_session_context_engaged", False)
        sc.reset_session_vars()
        for key in sc._VAR_MAP:
            monkeypatch.setenv(key, "stale-routing-value")

        captured = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["env"] = kwargs.get("env", {})
                self.pid = 4242

        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        task = kb.Task(
            id="t_dispatch_env",
            title="x",
            body=None,
            assignee="coder",
            status="ready",
            priority=0,
            created_by=None,
            created_at=0,
            started_at=None,
            completed_at=None,
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "ws"),
            claim_lock=None,
            claim_expires=None,
            tenant=None,
            branch_name="wt/t_dispatch_env",
        )
        kb._default_spawn(task, str(tmp_path / "ws"))

        env = captured["env"]
        assert env["HERMES_KANBAN_DB"] == str(default_home / "kanban.db")
        assert env["HERMES_KANBAN_WORKSPACES_ROOT"] == str(
            default_home / "kanban" / "workspaces"
        )
        assert env["HERMES_KANBAN_TASK"] == "t_dispatch_env"
        assert env["HERMES_KANBAN_BRANCH"] == "wt/t_dispatch_env"
        for key in sc._VAR_MAP:
            if key == "HERMES_SESSION_SOURCE":
                # Re-set by the dispatcher, so what matters is that it carries
                # the worker's own tag rather than the inherited routing value.
                assert env[key] == "kanban"
                continue
            assert key not in env


# ---------------------------------------------------------------------------
# latest_summary / latest_summaries — surface task_runs.summary handoffs
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# NFS / network-filesystem fallback (see hermes_state.apply_wal_with_fallback)
# ---------------------------------------------------------------------------

def test_connect_falls_back_to_delete_on_locking_protocol(tmp_path, monkeypatch, caplog):
    """kanban_db.connect() must handle ``locking protocol`` on NFS/SMB.

    Without this fallback, the gateway's kanban dispatcher crashes every
    60s and the kanban migration (``consecutive_failures`` ADD COLUMN) is
    retried forever — which is what the real-world user report shows
    (see hermes-agent issue #22032).

    NOTE: We do NOT use the ``kanban_home`` fixture here because that
    fixture pre-initializes the DB via ``kb.init_db()`` — putting the
    file in WAL on disk. The Bug D safety guard now refuses to downgrade
    to DELETE when the on-disk header is already WAL, so testing the
    NFS-fallback path requires a truly-fresh DB file (NFS scenario in
    production: first connection of the first process ever to touch the
    file, where downgrading is safe because nobody else has WAL state
    yet).
    """
    import sqlite3 as _sqlite3
    from unittest.mock import patch as _patch

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # These tests exercise the WAL-attempt path; assume a fixed SQLite so the
    # WAL-reset vulnerability gate doesn't short-circuit before the pragma.
    import hermes_state as _hermes_state
    monkeypatch.setattr(
        _hermes_state, "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )
    _hermes_state._wal_fallback_warned_paths.clear()

    # Clear module cache so a fresh connect() is attempted
    kb._INITIALIZED_PATHS.clear()
    hermes_state._wal_fallback_warned_paths.clear()

    real_connect = _sqlite3.connect

    class _WalBlockingConnection(_sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                raise _sqlite3.OperationalError("locking protocol")
            return super().execute(sql, *args, **kwargs)

    def wal_blocking_connect(*args, **kwargs):
        # connect_tracked passes a tracking-augmented factory; drop it and
        # substitute the double, which connect_tracked re-applies to the
        # returned instance.
        kwargs.pop("factory", None)
        return real_connect(
            *args, factory=_WalBlockingConnection, **kwargs
        )

    with _patch("hermes_cli.kanban_db.sqlite3.connect", side_effect=wal_blocking_connect):
        with caplog.at_level("ERROR", logger="hermes_state"):
            conn = kb.connect()

    # One fallback error, naming kanban.db
    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "kanban.db" in r.getMessage()
    ]
    assert len(errors) >= 1, (
        f"Expected a kanban.db ERROR, got: {[r.getMessage() for r in caplog.records]}"
    )

    # DB still usable end-to-end — create + list a task
    t = kb.create_task(conn, title="post-fallback task")
    tasks = kb.list_tasks(conn)
    assert any(row.id == t for row in tasks)
    conn.close()


def test_connect_works_when_wal_is_silently_refused(tmp_path, monkeypatch, caplog):
    """kanban_db.connect() must stay usable when WAL silently no-ops to DELETE."""
    import sqlite3 as _sqlite3
    from unittest.mock import patch as _patch

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    hermes_state._wal_fallback_warned_paths.clear()
    # Assume a fixed SQLite so the WAL-reset gate doesn't short-circuit.
    monkeypatch.setattr(
        hermes_state, "is_sqlite_wal_reset_vulnerable",
        lambda version_info=None: False,
    )

    real_connect = _sqlite3.connect

    class _WalSilentNoOpConnection(_sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if "journal_mode=wal" in sql.lower().replace(" ", ""):
                return super().execute("PRAGMA journal_mode=delete", *args, **kwargs)
            return super().execute(sql, *args, **kwargs)

    def wal_silent_noop_connect(*args, **kwargs):
        kwargs.pop("factory", None)
        return real_connect(
            *args, factory=_WalSilentNoOpConnection, **kwargs
        )

    with _patch(
        "hermes_cli.kanban_db.sqlite3.connect",
        side_effect=wal_silent_noop_connect,
    ):
        with caplog.at_level("ERROR", logger="hermes_state"):
            conn = kb.connect()

    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    t = kb.create_task(conn, title="post-silent-fallback task")
    tasks = kb.list_tasks(conn)
    assert any(row.id == t for row in tasks)
    conn.close()

    errors = [
        r
        for r in caplog.records
        if r.levelname == "ERROR" and "kanban.db" in r.getMessage()
    ]
    assert len(errors) >= 1, (
        f"Expected a kanban.db ERROR, got: {[r.getMessage() for r in caplog.records]}"
    )


def test_sqlite_connect_closes_tracked_conn_on_setup_failure(tmp_path, monkeypatch):
    """A PRAGMA failure after connect must not abandon a tracked kanban fd."""
    from hermes_cli import sqlite_safe_read

    db_path = tmp_path / "kanban.db"
    real_connect = sqlite3.connect
    opened = []

    class _BusyTimeoutFailure(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):  # type: ignore[override]
            if str(sql).startswith("PRAGMA busy_timeout="):
                raise sqlite3.OperationalError("simulated setup failure")
            return super().execute(sql, *args, **kwargs)

    def failing_connect(*args, **kwargs):
        kwargs.pop("factory", None)
        conn = real_connect(*args, factory=_BusyTimeoutFailure, **kwargs)
        opened.append(conn)
        return conn

    key = sqlite_safe_read._key(db_path)
    with sqlite_safe_read._live_lock:
        before = sqlite_safe_read._live_connections.get(key, 0)
    monkeypatch.setattr(kb.sqlite3, "connect", failing_connect)

    with pytest.raises(sqlite3.OperationalError, match="simulated setup failure"):
        kb._sqlite_connect(db_path)

    with sqlite_safe_read._live_lock:
        after = sqlite_safe_read._live_connections.get(key, 0)
    assert after == before


def test_unlink_tasks_triggers_recompute_ready(kanban_home):
    """Regression test for issue #22459.

    Removing a dependency via unlink_tasks must immediately promote the child
    to ready when all remaining parents are done — same contract as
    complete_task and unblock_task.

    Before the fix, child stayed 'todo' indefinitely after unlink; only the
    next dispatcher tick or a manual 'hermes kanban recompute' would promote it.
    """
    with kb.connect() as conn:
        # A is done.
        a = kb.create_task(conn, title="parent-done")
        kb.complete_task(conn, a)

        # C is running (not done) — blocks child B.
        c = kb.create_task(conn, title="parent-running")
        kb.claim_task(conn, c, claimer="worker:1")

        # B depends on both A (done) and C (running) → stays todo.
        b = kb.create_task(conn, title="child", parents=[a, c])
        assert kb.get_task(conn, b).status == "todo"

        # Remove the blocking dependency C → B.
        removed = kb.unlink_tasks(conn, c, b)
        assert removed is True

        # B's only remaining parent is A (done) → must be ready immediately.
        assert kb.get_task(conn, b).status == "ready", (
            "child should promote to ready immediately after unlink_tasks "
            "removes its last blocking dependency"
        )




def test_archive_stamps_completed_at_when_not_done(kanban_home):
    """Archiving a never-done task stamps completed_at so the dashboard's
    terminal-column windowing (ordered by completed_at DESC) places it by
    archive time, not its original created_at. An already-done task keeps
    its original completion timestamp."""
    with kb.connect() as conn:
        # never-done -> gets a completed_at on archive
        t = kb.create_task(conn, title="never done")
        assert kb.get_task(conn, t).completed_at is None
        assert kb.archive_task(conn, t)
        assert kb.get_task(conn, t).completed_at is not None

        # already-done -> completed_at preserved (not overwritten on archive)
        d = kb.create_task(conn, title="was done")
        kb.complete_task(conn, d)
        original = kb.get_task(conn, d).completed_at
        assert original is not None
        assert kb.archive_task(conn, d)
        assert kb.get_task(conn, d).completed_at == original


def test_list_tasks_exclude_statuses(kanban_home):
    """exclude_statuses drops the named statuses in SQL so the live-board
    pass never materializes the (potentially huge) done/archived history."""
    with kb.connect() as conn:
        ready = kb.create_task(conn, title="ready one")
        done = kb.create_task(conn, title="done one")
        kb.complete_task(conn, done)

        live = kb.list_tasks(conn, exclude_statuses={"done", "archived"})
        ids = {t.id for t in live}
        assert ready in ids
        assert done not in ids

        with pytest.raises(ValueError):
            kb.list_tasks(conn, exclude_statuses={"not-a-status"})

# ---------------------------------------------------------------------------
# _add_column_if_missing / _migrate_add_optional_columns idempotency (#21708)
# ---------------------------------------------------------------------------

def test_add_column_if_missing_is_idempotent_on_race(kanban_home):
    """``_add_column_if_missing`` must swallow 'duplicate column name' errors.

    Regression for #21708: the kanban dispatcher opens the DB twice per tick
    (once via _tick_once_for_board, once via init_db's discard-and-reconnect
    path).  A second concurrent connection runs _migrate_add_optional_columns
    before the first one commits, so ALTER TABLE raises OperationalError with
    'duplicate column name: consecutive_failures'.  Without the idempotency
    guard that crashes the dispatcher on the first tick after every restart.
    """
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT NOT NULL)"
    )

    # First call adds the column — returns True.
    added = kb._add_column_if_missing(conn, "tasks", "extra_col", "extra_col TEXT")
    assert added is True
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "extra_col" in cols

    # Second call on same connection — column already exists — must return
    # False without raising, simulating the race the dispatcher hits.
    added_again = kb._add_column_if_missing(
        conn, "tasks", "extra_col", "extra_col TEXT"
    )
    assert added_again is False

    conn.close()


def test_migrate_add_optional_columns_tolerates_concurrent_migration(kanban_home):
    """Full _migrate_add_optional_columns must not raise when columns already
    exist (issue #21708 race window — two connections migrate concurrently)."""
    import sqlite3

    # Schema already in fully-migrated state (all optional columns present).
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            branch_name TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_failure_error TEXT,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER,
            workflow_template_id TEXT,
            current_step_key TEXT,
            skills TEXT,
            max_retries INTEGER,
            session_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE task_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id    TEXT NOT NULL DEFAULT '',
            run_id     INTEGER,
            kind       TEXT NOT NULL DEFAULT '',
            payload    TEXT,
            created_at INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    # Running migration on an already-migrated schema must not raise.
    kb._migrate_add_optional_columns(conn)
    conn.close()


# ---------------------------------------------------------------------------
# Dispatcher spawn invocation — _resolve_hermes_argv()
#
# Workers spawned by the dispatcher must use a `hermes` invocation that does
# not depend on PATH being set up correctly. cron jobs, systemd User= services,
# launchd jobs, and other detached processes routinely run with a stripped
# $PATH that doesn't include the venv's bin/, so a bare `["hermes", ...]`
# spawn fails with FileNotFoundError and the task gets stuck. The resolver
# prefers the PATH shim (familiar `ps` output) but falls back to the module
# form so the spawn keeps working when PATH is missing the shim.
# ---------------------------------------------------------------------------


def test_resolve_hermes_argv_falls_back_to_module_form_when_no_path_shim(monkeypatch):
    """When the shim is not on PATH, fall back to `python -m hermes_cli.main`.

    Pins the correct module name (NOT `hermes` — there is no top-level
    `hermes` package). Regression for #23198: the original PR shipped
    `python -m hermes` which fails with `No module named hermes` on every
    invocation.
    """
    import shutil
    import sys
    import hermes_cli.kanban_db as kb

    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    argv = kb._resolve_hermes_argv()
    assert argv == [sys.executable, "-m", "hermes_cli.main"]


def test_resolve_hermes_argv_module_actually_runs():
    """The fallback module name must be importable + runnable.

    A unit test that pins the literal string is necessary but not
    sufficient — if `hermes_cli.main` ever loses `if __name__ == "__main__"`
    handling or its argparse setup, `python -m hermes_cli.main --version`
    would fail and so would every dispatcher spawn that hits the fallback.
    Run it as a real subprocess to catch that regression.
    """
    import subprocess
    import hermes_cli.kanban_db as kb
    import shutil
    import unittest.mock as mock

    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("HERMES_BIN", None)
        with mock.patch.object(shutil, "which", return_value=None):
            argv = kb._resolve_hermes_argv()
    r = subprocess.run(argv + ["--version"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (
        f"`{' '.join(argv)} --version` failed (rc={r.returncode}); "
        f"stderr={r.stderr[:200]!r}"
    )
    assert "Hermes Agent" in r.stdout, f"unexpected output: {r.stdout[:200]!r}"


# ---------------------------------------------------------------------------
# task_age — guard against corrupt timestamp values
#
# The Task dataclass declares ``created_at: int`` but rows come from sqlite
# without coercion at the boundary. A row that ever held a non-int (e.g. an
# unsubstituted ``'%s'`` from a logged format string, ``None``, an arbitrary
# string, or a float-as-string) used to crash ``task_age`` with ``ValueError``
# and turn ``GET /api/plugins/kanban/board`` into a 500 because the dashboard
# calls ``task_age`` unguarded for every task in the response.
#
# After the fix, ``_safe_int`` returns ``None`` on bad input and ``task_age``
# degrades gracefully (per-field ``None`` rather than a hard crash).
# ---------------------------------------------------------------------------


def _make_task(**overrides) -> "kb.Task":
    """Minimal Task with all required fields filled in. Override anything."""
    defaults = dict(
        id="t_age",
        title="x",
        body=None,
        assignee=None,
        status="ready",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    defaults.update(overrides)
    return kb.Task(**defaults)












# ---------------------------------------------------------------------------
# Board-level default_workdir
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# dispatch_once — max_in_progress
# ---------------------------------------------------------------------------


def test_dispatch_max_in_progress_blocks_review_when_at_limit(
    kanban_home, all_assignees_spawnable,
):
    """Review-only backlog must still respect max_in_progress."""
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        kb.claim_task(conn, running)
        review = kb.create_task(conn, title="review", assignee="bob")
        _set_task_status(conn, review, "review")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_in_progress=1)
        review_task = kb.get_task(conn, review)

    assert not res.spawned
    assert not spawns
    assert review_task is not None
    assert review_task.status == "review"

# Review column dispatch
# ---------------------------------------------------------------------------


def _set_task_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    """Test helper: set a task's status directly."""
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))








# Stale detection — detect_stale_running
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Corruption guard (issue #30687)
# ---------------------------------------------------------------------------

def _write_corrupt_db(path: Path) -> bytes:
    """Write a kanban DB with a VALID SQLite header but malformed page content.

    This is the corruption shape the integrity guard specifically targets
    (e.g. issue #29507 follow-up reports where the file's first 16 bytes
    pass the header byte check but ``PRAGMA integrity_check`` then fails
    because the internal pages are damaged). It's what main's header-only
    validator was letting through, and what this PR adds the full guard
    for.
    """
    # 100-byte SQLite header (magic + minimal valid-looking fields) so the
    # cheap header check passes, then deliberate garbage so sqlite refuses
    # to read the file past the header.
    header = b"SQLite format 3\x00" + b"\x10\x00\x02\x02\x00\x40\x20\x20"
    header += b"\x00\x00\x00\x0c\x00\x00\x23\x46\x00\x00\x00\x00"
    header = header.ljust(100, b"\x00")
    payload = b"definitely not a valid sqlite page \x00\x01\x02\x03" * 64
    blob = header + payload
    path.write_bytes(blob)
    return blob




def test_repeated_corrupt_open_reuses_single_backup(tmp_path):
    """Repeated quarantines of the same corrupt bytes must not amplify disk usage.

    Regression for the gateway dispatcher's 5-min retry loop on shared kanban
    DBs across multi-profile fleets: each retry on an unchanged corrupt file
    used to create a fresh ``.corrupt.<timestamp>.bak`` until disk filled. The
    content-addressed backup name is deterministic in the DB's sha256, so
    N retries of the same bytes share one backup.
    """
    db_path = tmp_path / "kanban.db"
    original = _write_corrupt_db(db_path)

    backups: set[Path] = set()
    for _ in range(10):
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
            kb.connect(db_path=db_path)
        assert excinfo.value.backup_path is not None
        backups.add(excinfo.value.backup_path)

    assert len(backups) == 1, f"expected 1 deterministic backup, got {len(backups)}"
    (backup,) = backups
    assert backup.exists()
    assert backup.read_bytes() == original

    # Mutate the corrupt bytes — fingerprint changes, separate backup preserved.
    with db_path.open("r+b") as f:
        f.seek(4096)
        f.write(b"\xAB" * 64)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with pytest.raises(kb.KanbanDbCorruptError) as excinfo2:
        kb.connect(db_path=db_path)
    second_backup = excinfo2.value.backup_path
    assert second_backup is not None
    assert second_backup != backup
    assert second_backup.exists()


def test_locked_healthy_db_does_not_classify_as_corrupt(tmp_path, monkeypatch):
    """A transient lock during the probe must not produce a .corrupt backup
    and must not be reported as :class:`KanbanDbCorruptError`. Raw sqlite
    ``OperationalError`` (lock/busy) is acceptable and expected."""
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    real_connect = sqlite3.connect

    def flaky_connect(*args, **kwargs):
        # First call is the integrity probe — simulate a lock.
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kb.sqlite3, "connect", flaky_connect)

    with pytest.raises(sqlite3.OperationalError):
        kb.connect(db_path=db_path)

    # No .corrupt backup may be produced for a healthy-but-locked DB.
    backups = list(tmp_path.glob("*.corrupt.*"))
    assert backups == [], f"unexpected corrupt backups: {backups}"

    # And once the lock clears, normal access still works.
    monkeypatch.setattr(kb.sqlite3, "connect", real_connect)
    with kb.connect(db_path=db_path) as conn:
        kb.create_task(conn, title="still here")
        titles = [t.title for t in kb.list_tasks(conn)]
    assert "still here" in titles




# ---------------------------------------------------------------------------
# Clone-storm cap + index-only self-heal (2026-07-13 incident)
# ---------------------------------------------------------------------------


def _corrupt_an_index_page(db_path: Path) -> bool:
    """Blank a secondary/auto index's b-tree entries while keeping the page
    structurally valid, so ``PRAGMA integrity_check`` reports the *index* has
    the wrong entry count (the recoverable index-only class) rather than a
    page-level "database disk image is malformed".

    Technique: set the index root page's "number of cells" field (2-byte big-
    endian at page offset 3) to zero and clear the cell-pointer array. SQLite
    then sees an index b-tree with fewer entries than the base table → "wrong #
    of entries in index". The base-table b-trees are untouched, so every row is
    still readable and ``iterdump`` recovers them fully.

    Returns True if an index page was corrupted, False if the DB has no
    corruptible secondary index (caller should skip). Assumes rollback journal
    mode with no live WAL sidecars.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        rows = conn.execute(
            "SELECT name, rootpage FROM sqlite_master "
            "WHERE type='index' AND rootpage > 0 ORDER BY rootpage"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return False
    _name, rootpage = rows[-1]
    offset = (rootpage - 1) * page_size
    with db_path.open("r+b") as fh:
        fh.seek(offset)
        page = bytearray(fh.read(page_size))
        page_type = page[0]  # 0x0a leaf index / 0x02 interior index — keep it
        # Zero the "number of cells" field (offset 3-4) and the cell-pointer
        # array so the index b-tree reports zero entries while staying parseable.
        page[3] = 0x00
        page[4] = 0x00
        header_len = 8 if page_type in (0x0a, 0x0d) else 12
        for i in range(header_len, page_size):
            page[i] = 0x00
        fh.seek(offset)
        fh.write(bytes(page))
    return True


def _make_index_corrupt_kanban_db(db_path: Path) -> list[str]:
    """Build a healthy kanban DB with data, then corrupt one index page.

    Returns the task titles that must survive a repair. Skips (pytest.skip) if
    the schema has no corruptible secondary index on this SQLite build.
    """
    with kb.connect_closing(db_path=db_path) as conn:
        kb.create_task(conn, title="alpha")
        kb.create_task(conn, title="beta")
        kb.create_task(conn, title="gamma")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    # Collapse WAL into the main file and switch to rollback journaling so the
    # main-file page layout is stable before we corrupt a page by byte offset.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.commit()
    finally:
        conn.close()
    for sidecar in ("-wal", "-shm"):
        s = db_path.with_name(db_path.name + sidecar)
        if s.exists():
            s.unlink()
    if not _corrupt_an_index_page(db_path):
        pytest.skip("no corruptible secondary index in kanban schema")
    return ["alpha", "beta", "gamma"]


def test_is_index_only_corruption_classifier():
    assert kb._is_index_only_corruption(
        ["wrong # of entries in index idx_notify_task"]
    )
    assert kb._is_index_only_corruption(
        [
            "row missing from index sqlite_autoindex_kanban_notify_subs_1",
            "wrong # of entries in index idx_notify_task",
        ]
    )
    # Any non-index problem disqualifies the whole set — never a partial repair.
    assert not kb._is_index_only_corruption(
        ["wrong # of entries in index idx_x", "Page 4 is never used"]
    )
    assert not kb._is_index_only_corruption(["database disk image is malformed"])
    assert not kb._is_index_only_corruption([])


def test_repairable_index_names_accepts_sqlite_grouped_diagnostics():
    """SQLite may return index diagnostics as one newline-delimited row."""
    messages = [
        "*** in database main ***\n"
        "Fragmentation of 19 bytes reported as 0 on page 30\n"
        "wrong # of entries in index idx_notify_task"
    ]

    assert kb._repairable_index_names(messages) == ["idx_notify_task"]


def test_index_only_corruption_self_heals_and_preserves_rows(tmp_path):
    """The 2026-07-13 board corruption class (stale indexes, base tables intact)
    must self-heal on open via REINDEX / .dump+reload with zero row loss —
    instead of refusing and quarantining a clone."""
    db_path = tmp_path / "kanban.db"
    expected = _make_index_corrupt_kanban_db(db_path)

    # Precondition: the DB is genuinely index-corrupt right now.
    problems = kb._integrity_problems(db_path)
    assert problems is not None, "expected the injected index corruption to register"
    assert kb._is_index_only_corruption(problems), problems

    # Clear the one-shot repair claim so the guard actually attempts repair.
    kb._REPAIR_ATTEMPTED_PATHS.discard(str(db_path.resolve()))
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    # Opening the DB must now recover it in place, not raise.
    with kb.connect(db_path=db_path) as conn:
        titles = sorted(t.title for t in kb.list_tasks(conn))
    assert titles == sorted(expected), "row data must survive the repair"

    # DB is clean afterwards and no forensic clone was kept for a recovered DB.
    assert kb._integrity_problems(db_path) is None
    # A recovered board keeps the pre-repair forensic copy by design.
    assert len(list(tmp_path.glob("*.corrupt.*.bak"))) == 1


def test_successful_repair_clears_repair_claim(tmp_path):
    """After a successful in-place repair the path is discarded from
    _REPAIR_ATTEMPTED_PATHS, so a genuinely fresh corruption on the same path
    later can be retried (honors the one-shot-claim contract)."""
    db_path = tmp_path / "kanban.db"
    _make_index_corrupt_kanban_db(db_path)
    resolved = str(db_path.resolve())

    kb._REPAIR_ATTEMPTED_PATHS.discard(resolved)
    kb._INITIALIZED_PATHS.discard(resolved)

    with kb.connect(db_path=db_path) as conn:
        kb.list_tasks(conn)

    # Repair succeeded, so the one-shot claim must have been released.
    with kb._REPAIR_ATTEMPT_LOCK:
        assert resolved not in kb._REPAIR_ATTEMPTED_PATHS


def test_reindex_lock_error_propagates_not_swallowed(tmp_path, monkeypatch):
    """A `database is locked` OperationalError during REINDEX is transient
    contention, not corruption. It subclasses DatabaseError but must re-raise
    (not fall through to the destructive dump+reload swap)."""
    db_path = tmp_path / "kanban.db"
    _make_index_corrupt_kanban_db(db_path)

    real_connect = kb._sqlite_connect

    class _LockOnReindex:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *a, **kw):
            if sql.strip().upper().startswith("REINDEX"):
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def _wrapped(path, *a, **kw):
        return _LockOnReindex(real_connect(path, *a, **kw))

    monkeypatch.setattr(kb, "_sqlite_connect", _wrapped)

    with pytest.raises(sqlite3.OperationalError):
        kb._attempt_index_only_repair(db_path)

    # It must NOT have fallen through to dump+reload (no rebuild temp file, and
    # the original corrupt DB is left in place untouched by a destructive swap).
    assert list(tmp_path.glob("*.rebuild.tmp")) == []


def test_transient_lock_during_repair_releases_claim(tmp_path, monkeypatch):
    """When the guard's self-heal hits a transient `database is locked`, it
    re-raises AND releases the one-shot repair claim, so the next connect (after
    the lock clears) re-attempts the heal instead of skipping to backup+refuse
    and leaving a recoverable board down until restart."""
    db_path = tmp_path / "kanban.db"
    _make_index_corrupt_kanban_db(db_path)
    resolved = str(db_path.resolve())
    kb._REPAIR_ATTEMPTED_PATHS.discard(resolved)
    kb._INITIALIZED_PATHS.discard(resolved)

    def _boom(_path, _index_names):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(kb, "_attempt_index_reindex_repair", _boom)

    with pytest.raises(sqlite3.OperationalError):
        kb._guard_existing_db_is_healthy(db_path)

    # Claim released so a retry can re-attempt the repair.
    with kb._REPAIR_ATTEMPT_LOCK:
        assert resolved not in kb._REPAIR_ATTEMPTED_PATHS


def test_dump_reload_lock_error_propagates(tmp_path, monkeypatch):
    """A transient `database is locked` during the dump+reload fallback must
    re-raise (not be swallowed by the broad DatabaseError->None handler that
    would trigger backup+refuse of a recoverable board)."""
    db_path = tmp_path / "kanban.db"
    _make_index_corrupt_kanban_db(db_path)

    real_connect = kb._sqlite_connect

    class _LockOnIterdump:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *a, **kw):
            # Force REINDEX to fail structurally so we fall into dump+reload...
            if sql.strip().upper().startswith("REINDEX"):
                raise sqlite3.DatabaseError("database disk image is malformed")
            return self._conn.execute(sql, *a, **kw)

        def iterdump(self):
            # ...then have the dump raise a transient lock.
            raise sqlite3.OperationalError("database is locked")

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def _wrapped(path, *a, **kw):
        return _LockOnIterdump(real_connect(path, *a, **kw))

    monkeypatch.setattr(kb, "_sqlite_connect", _wrapped)

    with pytest.raises(sqlite3.OperationalError):
        kb._attempt_index_only_repair(db_path)

    # Rebuild temp must be cleaned up on the transient re-raise.
    assert list(tmp_path.glob("*.rebuild.tmp")) == []


def test_repair_quiesces_writers_before_swap(tmp_path, monkeypatch):
    """The dump+reload repair must acquire an exclusive lock before swapping.
    A concurrent connection holding a write lock makes the repair re-raise a
    transient OperationalError (retry later) rather than publishing a rebuilt
    DB that could drop the other writer's committed rows."""
    db_path = tmp_path / "kanban.db"
    _make_index_corrupt_kanban_db(db_path)

    # Keep the repair's BEGIN EXCLUSIVE from waiting the full default 120s on the
    # blocker — fail fast so the test is quick and deterministic.
    monkeypatch.setenv("HERMES_KANBAN_BUSY_TIMEOUT_MS", "200")

    # Hold an exclusive write transaction open on the board from another
    # connection for the duration of the repair attempt.
    blocker = sqlite3.connect(str(db_path), isolation_level=None, timeout=0.2)
    blocker.execute("PRAGMA busy_timeout=200")
    try:
        blocker.execute("BEGIN EXCLUSIVE")
        with pytest.raises(sqlite3.OperationalError):
            kb._attempt_index_only_repair(db_path)
        # Nothing was published: no rebuild temp lingering, original untouched.
        assert list(tmp_path.glob("*.rebuild.tmp")) == []
    finally:
        blocker.rollback()
        blocker.close()


def test_repair_preserves_inode_so_idle_handle_writes_survive(tmp_path):
    """The in-place REINDEX repair keeps pre-existing handles attached."""
    db_path = tmp_path / "kanban.db"
    survivors = _make_index_corrupt_kanban_db(db_path)
    inode_before = db_path.stat().st_ino

    # Bypass the connection guard to hold the pre-repair SQLite handle.
    idle = kb._sqlite_connect(db_path)
    try:
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        kb._guard_existing_db_is_healthy(db_path)

        assert db_path.stat().st_ino == inode_before
        assert kb._integrity_problems(db_path) is None

        titles = {row[0] for row in idle.execute("SELECT title FROM tasks")}
        assert set(survivors).issubset(titles)
        kb.create_task(idle, title="written-through-idle-handle")
        idle.commit()

        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with kb.connect_closing(db_path=db_path) as fresh:
            titles_after = {row[0] for row in fresh.execute("SELECT title FROM tasks")}
        assert "written-through-idle-handle" in titles_after
    finally:
        idle.close()


def test_reused_backup_fills_in_missing_sidecars(tmp_path):
    """When the same corrupt main-file hash is quarantined again after a WAL
    sidecar appears, reusing the existing backup must still copy the sidecar so
    a WAL-only committed transaction is recoverable from the forensic copy."""
    db_path = tmp_path / "kanban.db"
    _write_corrupt_db(db_path)

    # First quarantine: no sidecars yet.
    first = kb._backup_corrupt_db(db_path)
    assert first is not None and first.exists()
    assert not (tmp_path / (first.name + "-wal")).exists()

    # A WAL sidecar now appears alongside the (byte-identical) corrupt main file.
    wal = db_path.with_name(db_path.name + "-wal")
    wal.write_bytes(b"fake-wal-committed-rows")

    # Re-quarantine: same hash → reuses `first`, but must now copy the sidecar.
    second = kb._backup_corrupt_db(db_path)
    assert second == first, "byte-identical corruption should reuse the backup"
    assert (tmp_path / (first.name + "-wal")).exists(), (
        "reused backup must gain the newly-present WAL sidecar"
    )


def test_corrupt_clone_count_is_capped(tmp_path):
    """A live WAL DB mutates on every rw probe, so N concurrent opens fingerprint
    different bytes and each used to write a fresh multi-MB clone (the storm).
    The per-path clone cap bounds this: the number of .corrupt.*.bak files must
    never exceed _MAX_CORRUPT_CLONES no matter how many times the bytes change."""
    db_path = tmp_path / "kanban.db"
    _write_corrupt_db(db_path)

    for i in range(12):
        # Mutate the corrupt bytes each iteration so content-addressing alone
        # would mint a brand-new clone (this is what the WAL churn did live).
        with db_path.open("r+b") as fh:
            fh.seek(4096)
            fh.write(bytes([i % 256]) * 128)
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kb.KanbanDbCorruptError):
            kb.connect(db_path=db_path)

    clones = list(tmp_path.glob("kanban.db.corrupt.*.bak"))
    assert len(clones) <= kb._MAX_CORRUPT_CLONES, (
        f"clone-storm not bounded: {len(clones)} clones "
        f"(cap {kb._MAX_CORRUPT_CLONES}): {[c.name for c in clones]}"
    )


def test_capped_backup_still_returns_a_forensic_path(tmp_path):
    """Even at the cap, the raised error carries a real, existing backup path so
    forensics (and the error message) are never left dangling."""
    db_path = tmp_path / "kanban.db"
    _write_corrupt_db(db_path)
    seen: list[Path] = []
    for i in range(6):
        with db_path.open("r+b") as fh:
            fh.seek(2048)
            fh.write(bytes([i]) * 64)
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kb.KanbanDbCorruptError) as excinfo:
            kb.connect(db_path=db_path)
        bp = excinfo.value.backup_path
        assert bp is not None and bp.exists()
        seen.append(bp)
    # At most the cap number of distinct forensic clones exist on disk.
    assert len({p.resolve() for p in seen}) <= kb._MAX_CORRUPT_CLONES


def test_capped_backup_reuse_still_fills_missing_sidecars(tmp_path):
    """At the clone cap, reusing an existing backup must still copy a WAL/SHM
    sidecar that appeared after that backup was made.

    In WAL mode committed rows can live only in kanban.db-wal until a
    checkpoint, so a main-file-only forensic copy can't reproduce/recover the
    board. The byte-identical reuse path already fills missing sidecars; the
    cap-reached reuse path used to early-return the newest clone WITHOUT calling
    _copy_missing_sidecars, so its backup_path could lack the WAL state.
    """
    db_path = tmp_path / "kanban.db"
    _write_corrupt_db(db_path)

    # Mint clones up to the cap by mutating the corrupt bytes each open.
    for i in range(kb._MAX_CORRUPT_CLONES + 3):
        with db_path.open("r+b") as fh:
            fh.seek(4096)
            fh.write(bytes([i % 256]) * 96)
        kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
        with pytest.raises(kb.KanbanDbCorruptError):
            kb.connect(db_path=db_path)

    clones = list(tmp_path.glob("kanban.db.corrupt.*.bak"))
    assert len(clones) == kb._MAX_CORRUPT_CLONES  # cap reached

    # A WAL sidecar now appears next to the live corrupt DB. The next quarantine
    # is over cap → it reuses the newest clone, but must copy the sidecar too.
    wal = db_path.with_name(db_path.name + "-wal")
    wal.write_bytes(b"wal-only-committed-rows")

    with db_path.open("r+b") as fh:
        fh.seek(4096)
        fh.write(b"\xab" * 96)  # mutate again so it's still over-cap reuse
    reused = kb._backup_corrupt_db(db_path)
    assert reused is not None and reused.exists()
    assert (tmp_path / (reused.name + "-wal")).exists(), (
        "capped-reuse backup must gain the newly-present WAL sidecar"
    )



# ---------------------------------------------------------------------------
# First-use tip for scratch workspaces
# ---------------------------------------------------------------------------

def test_maybe_emit_scratch_tip_fires_once_per_install(kanban_home, caplog):
    """First scratch workspace materialization warns + emits an event.

    Subsequent scratch workspaces on the SAME install stay silent — the
    sentinel file under kanban_home() flips after the first emit.
    """
    import logging

    with kb.connect() as conn:
        t1 = kb.create_task(conn, title="first scratch")
        t2 = kb.create_task(conn, title="second scratch")

    # Sentinel must not exist yet on a fresh install.
    assert not kb._scratch_tip_shown()

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with kb.connect() as conn:
            kb._maybe_emit_scratch_tip(conn, t1, "scratch")

    # Sentinel is now set.
    assert kb._scratch_tip_shown()
    assert kb._scratch_tip_sentinel_path().exists()

    # Warning was logged exactly once.
    tip_records = [
        r for r in caplog.records
        if "scratch workspaces are ephemeral" in r.getMessage()
    ]
    assert len(tip_records) == 1, (
        f"Expected exactly one tip warning, got {len(tip_records)}: "
        f"{[r.getMessage() for r in tip_records]!r}"
    )

    # An event row was appended on the first task.
    with kb.connect() as conn:
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (t1,),
        ).fetchall()
    kinds = [e["kind"] for e in events]
    assert "tip_scratch_workspace" in kinds, (
        f"Expected tip_scratch_workspace event on first scratch task; "
        f"got {kinds!r}"
    )

    # Second scratch materialization on the same install stays silent.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        with kb.connect() as conn:
            kb._maybe_emit_scratch_tip(conn, t2, "scratch")
    tip_records2 = [
        r for r in caplog.records
        if "scratch workspaces are ephemeral" in r.getMessage()
    ]
    assert tip_records2 == [], (
        f"Tip should not re-fire after sentinel is set; got "
        f"{[r.getMessage() for r in tip_records2]!r}"
    )
    with kb.connect() as conn:
        events2 = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (t2,),
        ).fetchall()
    assert "tip_scratch_workspace" not in [e["kind"] for e in events2], (
        "Tip event should not be appended for subsequent scratch tasks."
    )




# ---------------------------------------------------------------------------
# Connection pragmas (secure_delete, cell_size_check, synchronous=FULL)
# ---------------------------------------------------------------------------


def test_connect_sets_secure_delete_on(tmp_path):
    """secure_delete=ON must be active on every new connection."""
    db_path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path=db_path) as conn:
        row = conn.execute("PRAGMA secure_delete").fetchone()
    assert row[0] == 1, f"expected secure_delete=1, got {row[0]}"





# write_txn — rollback handler must not mask the original exception
# ---------------------------------------------------------------------------


def test_write_txn_preserves_original_exception_when_rollback_fails(kanban_home):
    """When a write inside write_txn raises an OperationalError that SQLite
    has already auto-rolled-back (e.g. ``disk I/O error``,
    ``database is locked``, ``database disk image is malformed``), the
    explicit ROLLBACK in ``write_txn.__exit__`` itself raises
    ``cannot rollback - no transaction is active``. The original cause
    must NOT be masked by the secondary rollback failure — operators rely
    on the original cause to diagnose the underlying issue.
    """

    class FailingConnWrapper:
        """Delegate to a real connection, simulating an EIO during an INSERT
        that SQLite has already auto-rolled-back."""

        def __init__(self, real):
            self._real = real
            self._fail_armed = True

        def execute(self, sql, *args, **kwargs):
            if (
                self._fail_armed
                and sql.lstrip().upper().startswith("INSERT")
                and "task_events" in sql.lower()
            ):
                self._fail_armed = False  # one-shot
                # Simulate SQLite auto-rolling back the transaction by
                # issuing a real ROLLBACK now. After this, BEGIN IMMEDIATE
                # is no longer active and an explicit ROLLBACK would error.
                try:
                    self._real.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                raise sqlite3.OperationalError("disk I/O error")
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    with kb.connect() as conn:
        wrapper = FailingConnWrapper(conn)
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            with kb.write_txn(wrapper):
                kb._append_event(wrapper, "t_bogus", "promoted", None)

    msg = str(excinfo.value)
    assert "disk I/O error" in msg, (
        f"write_txn masked the original exception with rollback failure; "
        f"got {msg!r} (expected to contain 'disk I/O error')"
    )
    assert "cannot rollback" not in msg, (
        f"write_txn surfaced the rollback failure instead of the original "
        f"OperationalError; got {msg!r}"
    )


def test_write_txn_check_reads_correct_header_fields(tmp_path):
    """A genuinely truncated DB is never reported as passing the invariant.

    The check no longer opens the database file to read header bytes (that
    open/close would cancel this process's POSIX advisory locks — the
    corruption route in sqlite.org/howtocorrupt.html §2.2). It asks SQLite for
    ``page_count`` instead. On a truncated file SQLite refuses that pragma, so
    the helper reports "not healthy" rather than a page-count mismatch; either
    way the file must never come back clean.
    """
    import struct
    from hermes_cli.kanban_db import connect
    from hermes_cli.sqlite_safe_read import file_length_matches_header

    db = tmp_path / "synthetic.db"
    conn = connect(db_path=db)
    conn.execute("PRAGMA journal_mode=DELETE")
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    conn.close()

    with open(db, "rb") as f:
        data = bytearray(f.read())
    real_page_count = struct.unpack(">I", data[28:32])[0]
    if real_page_count < 2:
        pytest.skip("DB too small for synthetic truncation test")
    truncated = bytes(data[: (real_page_count - 1) * page_size])
    with open(db, "wb") as f:
        f.write(truncated)

    raw_conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        assert file_length_matches_header(raw_conn) is not True
    finally:
        raw_conn.close()


# ---------------------------------------------------------------------------
# reap_worker_zombies() tests
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# connect_closing(): context manager that actually closes the FD
# Regression coverage for #33159 (kanban.db FD leak — gateway crashes after
# ~4 days). sqlite3.Connection's built-in __exit__ commits/rollbacks but
# does NOT close, so `with kb.connect() as conn:` leaks the FD in
# long-lived processes (gateway run_slash, dashboard decompose handler).
# `connect_closing()` is the leak-safe replacement.
# ---------------------------------------------------------------------------




def test_bare_connect_does_not_close_on_context_exit(tmp_path):
    """Document the leak that connect_closing exists to prevent.

    sqlite3.Connection's __exit__ commits/rollbacks but doesn't close.
    This is the upstream behaviour we cannot change; the regression
    guard is to make sure connect_closing() does the right thing.
    """
    db_path = tmp_path / "kanban.db"
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path=db_path) as conn:
        pass
    # Still usable after with-block exit (the leak).
    conn.execute("SELECT 1").fetchone()
    conn.close()  # explicit close to avoid leaking THIS test
