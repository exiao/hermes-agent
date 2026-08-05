"""Backup/scratch dot-directories inside ``skills/`` must not be indexed.

Regression context: `.archive` was fixed in eda1d516d ("fix(skills): exclude
.archive from skill index walk") after archived skills surfaced in
<available_skills> under a fake '.archive' category. The fix added one literal
to a hardcoded denylist, so the SAME bug returned the moment another dot-dir
convention appeared:

  * ``.curator_backups/`` — written by the skills curator before each run
  * ``.restore-backups/`` — written by core itself, tools/skills_sync.py:364

Both surfaced deleted skills (a June ``hyperframes`` snapshot, a July
``trl-fine-tuning``) as live, addressable entries. That is worse than wasted
prompt bytes: ``skill_view("hyperframes")`` could resolve to a stale backup
copy instead of the current skill.

These tests encode the RULE (no dot-directory is a skill category) rather than
the literals, so the next backup convention cannot re-open the hole.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.skill_utils import (
    is_excluded_skill_path,
    iter_skill_index_files,
)
from tools.skills_tool import skill_view

# Known writers of dot-dirs inside skills/, and the shape of what they store.
BACKUP_DIRS = (
    ".curator_backups/hyperframes-pre-v0.7.12-20260626/hyperframes",
    ".restore-backups/official-optional-20260730-163323/mlops/trl-fine-tuning",
    ".archive/some-retired-skill",
)


@pytest.mark.parametrize("rel", BACKUP_DIRS)
def test_backup_dirs_are_excluded(tmp_path, rel):
    """A SKILL.md inside a backup dot-dir is not a skill."""
    p = tmp_path / rel / "SKILL.md"
    p.parent.mkdir(parents=True)
    p.write_text("---\nname: stale\ndescription: from a backup\n---\n")
    assert is_excluded_skill_path(p, root=tmp_path), f"{rel} leaked into the index"


def test_no_dot_directory_is_ever_a_skill_category(tmp_path):
    """The RULE, not the literals.

    This is the test that actually prevents regression: any future dot-dir
    convention (.foo-backups, .snapshots, .trash) is excluded without anyone
    remembering to update a denylist.
    """
    for name in (".curator_backups", ".restore-backups", ".worktrees",
                 ".snapshots", ".trash", ".some-future-convention"):
        p = tmp_path / name / "pkg" / "SKILL.md"
        p.parent.mkdir(parents=True)
        p.write_text("---\nname: x\ndescription: y\n---\n")
        assert is_excluded_skill_path(p, root=tmp_path), (
            f"{name} would be indexed as a live skill category"
        )


def test_real_skills_still_indexed(tmp_path):
    """The guard must not over-match: normal categories keep working.

    Guards against a fix that excludes too much (e.g. matching any name
    containing a dot, which would kill a 'v1.2-tools' category).
    """
    for rel in ("coding/systematic-debugging", "memory/recall",
                "v1.2-tools/some-skill", "app-store/aso/keyword-research"):
        p = tmp_path / rel / "SKILL.md"
        p.parent.mkdir(parents=True)
        p.write_text("---\nname: real\ndescription: live skill\n---\n")
        assert not is_excluded_skill_path(p, root=tmp_path), f"{rel} wrongly excluded"


def test_index_walk_skips_backup_dirs_end_to_end(tmp_path):
    """Exercise the real os.walk path, not just the predicate.

    prompt_builder and skills_tool filter with `d not in EXCLUDED_SKILL_DIRS`
    during the walk; the predicate passing does not prove the walk prunes.
    """
    live = tmp_path / "coding" / "real-skill"
    live.mkdir(parents=True)
    (live / "SKILL.md").write_text("---\nname: real-skill\ndescription: live\n---\n")

    for backup in (".curator_backups/old/hyperframes", ".restore-backups/x/mlops/trl"):
        d = tmp_path / backup
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: stale\ndescription: backup\n---\n")

    # A DESCRIPTION.md in a backup dir must not become a category either —
    # that is exactly how ".curator_backups/... (6 skills)" appeared.
    desc = tmp_path / ".curator_backups" / "old"
    (desc / "DESCRIPTION.md").write_text("---\ndescription: stale category\n---\n")

    found = [str(f) for f in iter_skill_index_files(tmp_path, "SKILL.md")]
    found += [str(f) for f in iter_skill_index_files(tmp_path, "DESCRIPTION.md")]
    assert any("real-skill" in f for f in found), "live skill missing from index"
    assert not [f for f in found if ".curator_backups" in f or ".restore-backups" in f], (
        f"backup files leaked into the index: {found}"
    )


@pytest.mark.parametrize(
    "name",
    (".git", ".github", ".hub", ".archive", "__pycache__", "node_modules"),
)
def test_historical_exclusions_remain_behavioral(tmp_path, name):
    """Whatever the mechanism, the previously-fixed cases stay fixed."""
    candidate = tmp_path / name / "old" / "SKILL.md"
    assert is_excluded_skill_path(candidate, root=tmp_path)


def test_unrooted_absolute_scan_excludes_skill_backup(tmp_path):
    """Documented rootless calls still reject backups below a skills root."""
    candidate = (
        tmp_path
        / ".hermes"
        / "skills"
        / ".restore-backups"
        / "old"
        / "nested"
        / "skills"
        / "ghost"
        / "SKILL.md"
    )
    assert is_excluded_skill_path(candidate)

    live = tmp_path / ".hermes" / "skills" / "coding" / "real" / "SKILL.md"
    assert not is_excluded_skill_path(live)


def test_unrooted_external_scan_uses_configured_root(tmp_path):
    """External skill roots need not use the conventional 'skills' name."""
    external_root = tmp_path / ".team-library"
    candidate = external_root / ".snapshots" / "old" / "ghost" / "SKILL.md"

    with patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[external_root]
    ):
        assert is_excluded_skill_path(candidate)


def test_index_walk_rejects_visible_symlink_into_backup(tmp_path):
    """A visible category alias must not revive an in-tree hidden backup."""
    hidden = tmp_path / ".restore-backups" / "old" / "ghost"
    hidden.mkdir(parents=True)
    (hidden / "SKILL.md").write_text(
        "---\nname: ghost\ndescription: stale backup\n---\n"
    )
    try:
        (tmp_path / "visible-alias").symlink_to(
            tmp_path / ".restore-backups", target_is_directory=True
        )
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    assert list(iter_skill_index_files(tmp_path, "SKILL.md")) == []


def test_skill_view_skips_legacy_flat_files_in_backup_dirs(tmp_path):
    """Legacy <name>.md lookup must obey the same active-tree boundary."""
    skills_dir = tmp_path / "skills"
    hidden = skills_dir / ".restore-backups" / "old"
    hidden.mkdir(parents=True)
    (hidden / "ghost.md").write_text("# stale flat skill\n")

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        result = json.loads(skill_view("ghost"))

    assert result["success"] is False
    assert result["error"] == "Skill 'ghost' not found."


def test_skill_view_rejects_direct_symlink_alias_into_backup(tmp_path):
    """Explicit directory, flat, and categorized paths obey the boundary."""
    skills_dir = tmp_path / "skills"
    hidden = skills_dir / ".restore-backups" / "old"
    skill = hidden / "ghost"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: ghost\ndescription: stale backup\n---\nbackup\n"
    )
    (hidden / "flat.md").write_text("# stale flat skill\n")
    try:
        (skills_dir / "visible").symlink_to(
            skills_dir / ".restore-backups", target_is_directory=True
        )
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")

    with patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        for name in ("visible/old/ghost", "visible/old/flat", "visible:old/ghost"):
            result = json.loads(skill_view(name))
            assert result["success"] is False, f"{name} loaded a backup skill"
            assert result["error"] == f"Skill '{name}' not found."
