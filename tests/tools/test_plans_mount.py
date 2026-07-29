"""Regression tests: `~/.hermes/plans` must be mirrored into remote sandboxes.

Constitution rule 2f makes a plan file path mandatory in every dev card body, and
the kanban worker lanes (`dev`, `qa-modal`, `code-reviewer`, `pr-babysitter`) run
in Modal cloud VMs that cannot see the host filesystem. Before `iter_plans_files()`
existed, a card told the worker to read `~/.hermes/plans/<task>.md` and the worker
blocked because no such file was present (kanban card t_8b08b46a, 2026-07-29).

These tests exercise the real function against a temp HERMES_HOME rather than
mocking, per AGENTS.md ("E2E validation, not just green unit mocks").
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture()
def plans_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with a representative plans/ tree."""
    home = tmp_path / "hermes_home"
    plans = home / "plans"
    (plans / "hermes-patches").mkdir(parents=True)
    (plans / "archive").mkdir()
    (plans / "wt-reaper-rescued" / "t_dead" / "raw").mkdir(parents=True)

    (plans / "t_live.md").write_text("# live plan\n")
    (plans / "diagram.svg").write_text("<svg/>")
    (plans / "hermes-patches" / "some-patch.md").write_text("# patch note\n")
    (plans / "archive" / "t_old.md").write_text("# superseded\n")
    (plans / "wt-reaper-rescued" / "t_dead" / "raw" / "corpus.md").write_text("junk\n")
    (plans / "binary.bin").write_bytes(b"\x00\x01\x02")

    monkeypatch.setenv("HERMES_HOME", str(home))
    import tools.credential_files as cf

    importlib.reload(cf)
    yield plans, cf
    importlib.reload(cf)


class TestIterPlansFiles:
    def test_live_plan_is_mounted_at_the_path_a_card_would_reference(self, plans_home):
        """A card linking ~/.hermes/plans/<name>.md must resolve in-container."""
        _, cf = plans_home
        entries = cf.iter_plans_files()
        container = {e["container_path"] for e in entries}
        assert "/root/.hermes/plans/t_live.md" in container

    def test_nested_structure_is_preserved(self, plans_home):
        _, cf = plans_home
        container = {e["container_path"] for e in cf.iter_plans_files()}
        assert "/root/.hermes/plans/hermes-patches/some-patch.md" in container

    def test_diagrams_are_included(self, plans_home):
        """Plans reference architecture SVGs; they travel with the plan."""
        _, cf = plans_home
        container = {e["container_path"] for e in cf.iter_plans_files()}
        assert "/root/.hermes/plans/diagram.svg" in container

    @pytest.mark.parametrize("skipped", ["archive", "wt-reaper-rescued"])
    def test_dead_subtrees_are_not_uploaded(self, plans_home, skipped):
        """Archived plans and rescued run artifacts would add ~28MB per sandbox."""
        _, cf = plans_home
        container = {e["container_path"] for e in cf.iter_plans_files()}
        assert not any(f"/plans/{skipped}/" in c for c in container)

    def test_non_text_artifacts_are_excluded(self, plans_home):
        """A stray binary in plans/ must not bloat every sandbox start."""
        _, cf = plans_home
        container = {e["container_path"] for e in cf.iter_plans_files()}
        assert not any(c.endswith(".bin") for c in container)

    def test_symlinks_are_skipped(self, plans_home):
        plans, cf = plans_home
        target = plans / "t_live.md"
        link = plans / "t_link.md"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform")
        container = {e["container_path"] for e in cf.iter_plans_files()}
        assert "/root/.hermes/plans/t_link.md" not in container

    def test_host_paths_exist_and_are_absolute(self, plans_home):
        _, cf = plans_home
        for entry in cf.iter_plans_files():
            host = Path(entry["host_path"])
            assert host.is_absolute()
            assert host.is_file()

    def test_missing_plans_dir_is_not_an_error(self, tmp_path, monkeypatch):
        """A fresh HERMES_HOME with no plans/ must not break sandbox creation."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty_home"))
        import tools.credential_files as cf

        importlib.reload(cf)
        try:
            assert cf.iter_plans_files() == []
        finally:
            importlib.reload(cf)


class TestModalWiring:
    def test_modal_backend_mounts_plans(self):
        """The Modal sandbox path must actually call iter_plans_files()."""
        source = Path(__file__).resolve().parents[2] / "tools/environments/modal.py"
        text = source.read_text()
        assert "iter_plans_files" in text, (
            "modal.py must import and iterate iter_plans_files(); without the "
            "wiring the helper exists but no plan reaches the sandbox."
        )
