"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _init_git_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "kanban@example.com"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Kanban Test"], check=True, capture_output=True, text=True)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True, capture_output=True, text=True)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kb.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_returns_none_when_task_missing(kanban_home):
    with kb.connect() as conn:
        result = kb.decompose_triage_task(
            conn,
            "nonexistent",
            root_assignee="orch",
            children=[{"title": "x"}],
            author="me",
        )
    assert result is None


def test_decompose_returns_none_when_task_not_in_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="already a real task")  # not triage
        result = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "x"}],
            author="me",
        )
    assert result is None


def test_decompose_empty_children_returns_none(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        result = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[],
            author="me",
        )
    assert result is None


def test_decompose_rejects_self_parent(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="cannot list itself"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "x", "parents": [0]}],
                author="me",
            )


def test_decompose_rejects_out_of_range_parent(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="not a valid index"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[{"title": "x", "parents": [5]}],
                author="me",
            )


def test_decompose_rejects_cyclic_parents(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        with pytest.raises(ValueError, match="cyclic dependency"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orch",
                children=[
                    {"title": "A", "parents": [1]},
                    {"title": "B", "parents": [0]},
                ],
                author="me",
            )


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn)
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kb.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


def test_decompose_children_inherit_dir_workspace(kanban_home):
    """Fan-out children inherit the root's dir workspace, not scratch."""
    proj = "/home/teknium/myproject"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="codegen root", assignee="worker",
            workspace_kind="dir", workspace_path=proj, triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "part A"}, {"title": "part B", "parents": [0]}],
            author="decomposer",
        )
    assert child_ids and len(child_ids) == 2
    with kb.connect() as conn:
        for cid in child_ids:
            t = kb.get_task(conn, cid)
            assert t.workspace_kind == "dir"
            assert t.workspace_path == proj


def test_decompose_children_stay_scratch_when_root_scratch(kanban_home):
    """No regression: a scratch root still fans out into scratch children."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="scratch root", assignee="worker",
            workspace_kind="scratch", triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "s1"}], author="decomposer",
        )
    with kb.connect() as conn:
        t = kb.get_task(conn, child_ids[0])
    assert t.workspace_kind == "scratch"
    assert t.workspace_path is None


def test_decompose_per_child_workspace_override(kanban_home):
    """An explicit per-child workspace beats inheritance."""
    proj = "/home/teknium/myproject"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="root", assignee="worker",
            workspace_kind="dir", workspace_path=proj, triage=True,
        )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[
                {"title": "override", "workspace_kind": "dir",
                 "workspace_path": "/other/repo"},
                {"title": "inherit"},
            ],
            author="decomposer",
        )
    with kb.connect() as conn:
        over = kb.get_task(conn, child_ids[0])
        inh = kb.get_task(conn, child_ids[1])
    assert over.workspace_path == "/other/repo"
    assert inh.workspace_path == proj


def test_decompose_strips_scheme_prefix_from_child_override(kanban_home, tmp_path):
    """A child workspace override carrying a '<scheme>:<path>' prefix must be
    self-healed before the direct INSERT, just like create_task does. Codex P2:
    decompose_triage_task bypassed the create-path guard, so a child passing
    workspace_path='worktree:/repo' (or inheriting a prefixed legacy root)
    landed as scratch + 'worktree:/repo' — a newly malformed row."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="root", assignee="worker", triage=True)
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[
                {"title": "wt-prefix", "workspace_path": f"worktree:{repo}"},
                {"title": "dir-prefix", "workspace_path": "dir:/abs/dir"},
            ],
            author="decomposer",
        )
    assert child_ids is not None
    with kb.connect() as conn:
        wt = kb.get_task(conn, child_ids[0])
        dr = kb.get_task(conn, child_ids[1])
    assert wt is not None
    assert dr is not None
    # Scheme promoted to kind, path stored bare (no prefix).
    assert wt.workspace_kind == "worktree"
    assert wt.workspace_path == str(repo)
    assert dr.workspace_kind == "dir"
    assert dr.workspace_path == "/abs/dir"


def test_decompose_strips_prefix_from_inherited_legacy_root(kanban_home, tmp_path):
    """When the root itself carries a legacy prefixed path, an inheriting child
    (no explicit override) must not copy the malformed value verbatim."""
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="root", assignee="worker", triage=True)
        # Force a malformed persisted root path (bypass create guard).
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'scratch', "
                "workspace_path = ? WHERE id = ?",
                (f"worktree:{repo}", tid),
            )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "inherit"}],
            author="decomposer",
        )
    assert child_ids is not None
    with kb.connect() as conn:
        inh = kb.get_task(conn, child_ids[0])
    assert inh is not None
    assert inh.workspace_kind == "worktree"
    assert inh.workspace_path == str(repo)


def test_decompose_rejects_unresolvable_worktree_child_path(kanban_home, tmp_path):
    """A direct child INSERT must not bypass the create-time worktree guard."""
    not_repo = tmp_path / "not-repo"
    not_repo.mkdir()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="root", assignee="worker", triage=True)
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="not inside a git repo"):
            kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="orchestrator",
                children=[
                    {
                        "title": "bad worktree child",
                        "workspace_kind": "worktree",
                        "workspace_path": str(not_repo),
                    }
                ],
                author="decomposer",
            )
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        root = kb.get_task(conn, tid)
    assert after == before
    assert root is not None
    assert root.status == "triage"


def test_decompose_child_kind_mismatch_no_path_raises_and_rolls_back(kanban_home):
    """A child that inherits a persistent kind but ends with no resolvable
    path is an un-spawnable zombie — same defect as create_task's silent
    NULL-path bug. Here the root is 'dir' (with a path) and a child overrides
    workspace_kind='worktree' WITHOUT a path: kinds mismatch, so the child
    can't inherit the root's dir path and ends with child_ws_path=None.

    The guard must raise inside the write_txn, rolling back the WHOLE
    decomposition: no children created, root stays in triage.
    """
    proj = "/home/teknium/myproject"
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="codegen root", assignee="worker",
            workspace_kind="dir", workspace_path=proj, triage=True,
        )
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="requires a workspace_path"):
            kb.decompose_triage_task(
                conn, tid, root_assignee="orchestrator",
                children=[
                    {"title": "ok part"},
                    {"title": "bad part", "workspace_kind": "worktree"},
                ],
                author="decomposer",
            )
        # Whole decomposition rolled back: no children, count unchanged.
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert after == before
        # Root untouched — still in triage, not flipped to todo.
        root = kb.get_task(conn, tid)
        assert root.status == "triage"


def test_decompose_legacy_worktree_root_null_path_resolves_board_default(kanban_home, monkeypatch, tmp_path):
    """Regression (Codex P2): a pre-existing triage root with
    workspace_kind='worktree' and a NULL workspace_path is NOT necessarily
    un-spawnable — resolve_workspace can anchor a worktree on the board's
    default_workdir at dispatch. The decompose guard must therefore resolve
    the board default before raising, so upgrading does not strand legitimate
    legacy triage cards. The worktree child is created (not rolled back), but
    keeps its workspace_path NULL so dispatch anchors a per-task worktree at
    ``<repo>/.worktrees/<id>`` rather than running in the shared default dir
    (see Codex P2b below).
    """
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    kb.create_board("legacy-wt-board", default_workdir=str(repo))
    # The dispatcher pins the worker's board via HERMES_KANBAN_BOARD; decompose
    # runs on that active board, so get_current_board() resolves it.
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "legacy-wt-board")
    with kb.connect(board="legacy-wt-board") as conn:
        tid = kb.create_task(
            conn, title="root", assignee="worker", triage=True,
            workspace_kind="scratch", board="legacy-wt-board",
        )
        # Simulate a pre-fix legacy root: worktree kind, NULL path.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'worktree', "
                "workspace_path = NULL WHERE id = ?",
                (tid,),
            )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "inherit"}],
            author="decomposer",
        )
    assert child_ids is not None
    with kb.connect(board="legacy-wt-board") as conn:
        inh = kb.get_task(conn, child_ids[0])
    assert inh.workspace_kind == "worktree"
    # Resolved (not rolled back), but path stays NULL so dispatch anchors a
    # per-task worktree under the board-default repo.
    assert inh.workspace_path is None
    with kb.connect(board="legacy-wt-board") as conn:
        ws = kb.resolve_workspace(inh, board="legacy-wt-board")
    assert ws == repo / ".worktrees" / inh.id


def test_decompose_worktree_child_subdir_default_keeps_null_path(kanban_home, monkeypatch, tmp_path):
    """Regression (Codex P2b): when a legacy triage root is worktree kind +
    NULL path and the board default_workdir points at a SUBDIR inside the repo,
    the decompose fallback must NOT persist that subdir as the child's explicit
    workspace_path. If it did, dispatch's _resolve_worktree_workspace would
    treat the subdir as the requested worktree target and run the child in the
    SHARED checkout subdir instead of an isolated ``<repo>/.worktrees/<id>``.
    The path must stay NULL so dispatch anchors per-task at the repo root.
    """
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    subdir = repo / "packages" / "core"
    subdir.mkdir(parents=True)
    kb.create_board("subdir-wt-board", default_workdir=str(subdir))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "subdir-wt-board")
    with kb.connect(board="subdir-wt-board") as conn:
        tid = kb.create_task(
            conn, title="root", assignee="worker", triage=True,
            workspace_kind="scratch", board="subdir-wt-board",
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'worktree', "
                "workspace_path = NULL WHERE id = ?",
                (tid,),
            )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "inherit"}],
            author="decomposer",
        )
    assert child_ids is not None
    with kb.connect(board="subdir-wt-board") as conn:
        inh = kb.get_task(conn, child_ids[0])
        assert inh.workspace_kind == "worktree"
        # NOT the raw subdir default — NULL, so dispatch anchors per-task.
        assert inh.workspace_path is None
        ws = kb.resolve_workspace(inh, board="subdir-wt-board")
    # Isolated per-task worktree under the repo ROOT, not the shared subdir.
    assert ws == repo / ".worktrees" / inh.id
    assert ws != subdir


def test_decompose_dir_child_subdir_default_persists_path(kanban_home, monkeypatch, tmp_path):
    """Control for P2b: a 'dir' child (not worktree) with no path legitimately
    inherits the board default_workdir verbatim — dir tasks run in-place, so
    persisting the subdir is correct. Only the worktree branch keeps NULL.
    """
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    subdir = repo / "packages" / "core"
    subdir.mkdir(parents=True)
    kb.create_board("subdir-dir-board", default_workdir=str(subdir))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "subdir-dir-board")
    with kb.connect(board="subdir-dir-board") as conn:
        tid = kb.create_task(
            conn, title="root", assignee="worker", triage=True,
            workspace_kind="scratch", board="subdir-dir-board",
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'dir', "
                "workspace_path = NULL WHERE id = ?",
                (tid,),
            )
        child_ids = kb.decompose_triage_task(
            conn, tid, root_assignee="orchestrator",
            children=[{"title": "inherit"}],
            author="decomposer",
        )
    assert child_ids is not None
    with kb.connect(board="subdir-dir-board") as conn:
        inh = kb.get_task(conn, child_ids[0])
    assert inh.workspace_kind == "dir"
    assert inh.workspace_path == str(subdir)

