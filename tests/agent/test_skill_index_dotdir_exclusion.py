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
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS,
    is_excluded_skill_path,
    iter_skill_index_files,
)

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


def test_denylist_still_covers_historical_entries():
    """Whatever the mechanism, the previously-fixed cases stay fixed."""
    for name in (".git", ".github", ".hub", ".archive", "__pycache__", "node_modules"):
        assert name in EXCLUDED_SKILL_DIRS
