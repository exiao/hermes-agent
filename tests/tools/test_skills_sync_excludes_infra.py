"""Skills sync ships skill content, not the tree's local infrastructure."""

from pathlib import Path

import pytest

from tools.credential_files import _walk_skill_tree


def _skill_tree(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    (root / "coding" / "debug").mkdir(parents=True)
    (root / "coding" / "debug" / "SKILL.md").write_text("# debug\n")
    (root / "coding" / "debug" / "references").mkdir()
    (root / "coding" / "debug" / "references" / "api.md").write_text("api\n")
    return root


def _remote_paths(root: Path) -> set[str]:
    return {e["container_path"] for e in _walk_skill_tree(root, "/root/.hermes/skills")}


def test_real_skill_files_are_still_synced(tmp_path):
    root = _skill_tree(tmp_path)
    paths = _remote_paths(root)
    assert "/root/.hermes/skills/coding/debug/SKILL.md" in paths
    assert "/root/.hermes/skills/coding/debug/references/api.md" in paths


def test_registry_index_cache_is_not_synced(tmp_path):
    root = _skill_tree(tmp_path)
    hub = root / ".hub" / "index-cache"
    hub.mkdir(parents=True)
    (hub / "hermes-index.json").write_text("{}" * 100)
    assert not any(".hub" in p for p in _remote_paths(root))


def test_vendored_dependencies_are_not_synced(tmp_path):
    root = _skill_tree(tmp_path)
    nm = root / "coding" / "debug" / "scripts" / "node_modules" / "left-pad"
    nm.mkdir(parents=True)
    (nm / "index.js").write_text("module.exports = 1\n")
    assert not any("node_modules" in p for p in _remote_paths(root))


@pytest.mark.parametrize(
    "name",
    [
        ".archive",
        ".curator_backups",
        ".git",
        ".github",
        ".hub",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".restore-backups",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".worktrees",
        "__pycache__",
        "node_modules",
        "site-packages",
        "venv",
    ],
)
def test_each_excluded_dir_is_skipped(tmp_path, name):
    root = _skill_tree(tmp_path)
    junk = root / name
    junk.mkdir()
    (junk / "payload.bin").write_text("x" * 64)
    assert not any(f"/{name}/" in p for p in _remote_paths(root))


def test_a_skill_file_sharing_an_excluded_name_is_still_synced(tmp_path):
    # The guard matches directories only: a file named ".git" or a skill
    # documenting node_modules must not be dropped.
    root = _skill_tree(tmp_path)
    (root / "coding" / "debug" / "node_modules").write_text("notes\n")
    paths = _remote_paths(root)
    assert "/root/.hermes/skills/coding/debug/node_modules" in paths
