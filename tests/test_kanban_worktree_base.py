"""Worktree base-ref behavior for kanban task workspaces.

A spawned task's worktree must branch from ``origin/<default>``, never from the
anchor checkout's HEAD. On 2026-07-21 a live board's anchor checkout was parked
on a stacked review branch, so every spawned worktree silently baked two
unmerged PRs into its diff. These tests pin the fix (branch off the remote
default; fall back to HEAD only when no remote exists).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli.kanban_db import _ensure_git_worktree, _worktree_base_ref


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture()
def origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """A bare 'origin' with main history, and a clone acting as the anchor checkout."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   capture_output=True, check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(origin), str(seed)], capture_output=True, check=True)
    _git(seed, "config", "user.email", "t@t")
    _git(seed, "config", "user.name", "t")
    (seed / "base.txt").write_text("base\n")
    _git(seed, "add", "base.txt")
    _git(seed, "commit", "-m", "base commit")
    _git(seed, "push", "origin", "main")
    clone = tmp_path / "anchor"
    subprocess.run(["git", "clone", str(origin), str(clone)], capture_output=True, check=True)
    _git(clone, "config", "user.email", "t@t")
    _git(clone, "config", "user.name", "t")
    return origin, clone


def test_base_ref_prefers_origin_default(origin_and_clone) -> None:
    _, clone = origin_and_clone
    assert _worktree_base_ref(clone) == "origin/main"


def test_base_ref_refreshes_only_default_branch(origin_and_clone, tmp_path: Path) -> None:
    """Refreshing the base must not update every tracked remote branch."""
    origin, clone = origin_and_clone
    writer = tmp_path / "writer"
    subprocess.run(["git", "clone", str(origin), str(writer)], capture_output=True, check=True)
    _git(writer, "config", "user.email", "t@t")
    _git(writer, "config", "user.name", "t")

    _git(writer, "checkout", "-b", "unrelated")
    (writer / "unrelated.txt").write_text("first\n")
    _git(writer, "add", "unrelated.txt")
    _git(writer, "commit", "-m", "first unrelated commit")
    _git(writer, "push", "origin", "unrelated")
    _git(clone, "fetch", "origin", "unrelated:refs/remotes/origin/unrelated")
    unrelated_before = _git(clone, "rev-parse", "origin/unrelated")

    _git(writer, "checkout", "main")
    (writer / "base.txt").write_text("fresh default\n")
    _git(writer, "commit", "-am", "fresh default commit")
    _git(writer, "push", "origin", "main")
    _git(writer, "checkout", "unrelated")
    (writer / "unrelated.txt").write_text("second\n")
    _git(writer, "commit", "-am", "second unrelated commit")
    _git(writer, "push", "origin", "unrelated")

    assert _worktree_base_ref(clone) == "origin/main"
    assert _git(clone, "rev-parse", "origin/main") == _git(writer, "rev-parse", "origin/main")
    assert _git(clone, "rev-parse", "origin/unrelated") == unrelated_before


def test_base_ref_discovers_non_main_remote_default_without_origin_head(tmp_path: Path) -> None:
    """A deleted local origin/HEAD must not lose a remote trunk default."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-b", "trunk", str(origin)],
                   capture_output=True, check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(origin), str(seed)], capture_output=True, check=True)
    _git(seed, "config", "user.email", "t@t")
    _git(seed, "config", "user.name", "t")
    (seed / "base.txt").write_text("base\n")
    _git(seed, "add", "base.txt")
    _git(seed, "commit", "-m", "base commit")
    _git(seed, "push", "origin", "trunk")

    clone = tmp_path / "anchor"
    subprocess.run(["git", "clone", str(origin), str(clone)], capture_output=True, check=True)
    _git(clone, "remote", "set-head", "origin", "-d")

    assert _worktree_base_ref(clone) == "origin/trunk"


def test_base_ref_refreshes_a_stale_origin_head_default(origin_and_clone, tmp_path: Path) -> None:
    """The remote's current default wins when local origin/HEAD is stale."""
    origin, clone = origin_and_clone
    writer = tmp_path / "writer"
    subprocess.run(["git", "clone", str(origin), str(writer)], capture_output=True, check=True)
    _git(writer, "config", "user.email", "t@t")
    _git(writer, "config", "user.name", "t")
    _git(writer, "checkout", "-b", "trunk")
    (writer / "trunk.txt").write_text("trunk\n")
    _git(writer, "add", "trunk.txt")
    _git(writer, "commit", "-m", "trunk default commit")
    _git(writer, "push", "origin", "trunk")
    _git(origin, "symbolic-ref", "HEAD", "refs/heads/trunk")

    assert _git(clone, "symbolic-ref", "refs/remotes/origin/HEAD") == "refs/remotes/origin/main"
    assert _worktree_base_ref(clone) == "origin/trunk"


def test_base_ref_force_updates_a_rewritten_default(origin_and_clone, tmp_path: Path) -> None:
    """A force-pushed remote default must replace its cached tracking ref."""
    origin, clone = origin_and_clone
    base_sha = _git(clone, "rev-parse", "origin/main")
    writer = tmp_path / "writer"
    subprocess.run(["git", "clone", str(origin), str(writer)], capture_output=True, check=True)
    _git(writer, "config", "user.email", "t@t")
    _git(writer, "config", "user.name", "t")
    (writer / "base.txt").write_text("discarded\n")
    _git(writer, "commit", "-am", "discarded default commit")
    _git(writer, "push", "origin", "main")
    assert _worktree_base_ref(clone) == "origin/main"

    _git(writer, "reset", "--hard", base_sha)
    (writer / "replacement.txt").write_text("replacement\n")
    _git(writer, "add", "replacement.txt")
    _git(writer, "commit", "-m", "replacement default commit")
    _git(writer, "push", "--force", "origin", "main")

    assert _worktree_base_ref(clone) == "origin/main"
    assert _git(clone, "rev-parse", "origin/main") == _git(writer, "rev-parse", "main")


def test_base_ref_falls_back_to_head_without_remote(tmp_path: Path) -> None:
    repo = tmp_path / "local-only"
    subprocess.run(["git", "init", "-b", "main", str(repo)], capture_output=True, check=True)
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("x\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-m", "only commit")
    assert _worktree_base_ref(repo) == "HEAD"


def test_worktree_ignores_parked_anchor_branch(origin_and_clone, tmp_path: Path) -> None:
    """Anchor parked on a stacked feature branch must NOT leak into the worktree."""
    _, clone = origin_and_clone
    # Park the anchor checkout on a local branch with a foreign commit
    # (simulates a local PR-review checkout left behind).
    _git(clone, "checkout", "-b", "stacked-review")
    (clone / "foreign.txt").write_text("should never appear in task worktrees\n")
    _git(clone, "add", "foreign.txt")
    _git(clone, "commit", "-m", "foreign stacked commit")

    target = clone / ".worktrees" / "t_test1234"
    _ensure_git_worktree(clone, target, "wt/t_test1234")

    assert (target / "base.txt").exists()
    assert not (target / "foreign.txt").exists(), (
        "worktree branched from the anchor's parked HEAD instead of origin/<default>"
    )
    # And the new branch's tip is exactly the remote default tip.
    assert _git(target, "rev-parse", "HEAD") == _git(clone, "rev-parse", "origin/main")


def test_worktree_reuses_existing_branch_unchanged(origin_and_clone) -> None:
    """A pre-existing task branch is checked out as-is (resume semantics)."""
    _, clone = origin_and_clone
    _git(clone, "branch", "wt/resume", "origin/main")
    target = clone / ".worktrees" / "t_resume"
    _ensure_git_worktree(clone, target, "wt/resume")
    assert _git(target, "rev-parse", "--abbrev-ref", "HEAD") == "wt/resume"
