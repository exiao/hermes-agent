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

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def plans_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with a representative plans/ tree."""
    home = tmp_path / "hermes_home"
    plans = home / "plans"
    (plans / "hermes-patches").mkdir(parents=True)
    (plans / "archive").mkdir()
    (plans / "wt-reaper-rescued" / "t_dead" / "raw").mkdir(parents=True)

    (plans / "t_live.md").write_text("# live plan\n", encoding="utf-8")
    (plans / "diagram.svg").write_text("<svg/>", encoding="utf-8")
    (plans / "hermes-patches" / "some-patch.md").write_text(
        "# patch note\n", encoding="utf-8"
    )
    (plans / "archive" / "t_old.md").write_text("# superseded\n", encoding="utf-8")
    (plans / "wt-reaper-rescued" / "t_dead" / "raw" / "corpus.md").write_text(
        "junk\n", encoding="utf-8"
    )
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
    def test_modal_backend_mounts_plan_files(self, monkeypatch):
        """Plans are passed through to Modal's sandbox mounts at creation."""
        import tools.credential_files as cf
        from tools.environments import modal as modal_env

        mounts = []
        create_calls = []

        class FakeMount:
            @staticmethod
            def from_local_file(host_path, *, remote_path):
                mounts.append((host_path, remote_path))
                return (host_path, remote_path)

        async def lookup_app(*_args, **_kwargs):
            return object()

        async def create_sandbox(*args, **kwargs):
            create_calls.append((args, kwargs))
            return object()

        fake_modal = SimpleNamespace(
            App=SimpleNamespace(lookup=SimpleNamespace(aio=lookup_app)),
            Mount=FakeMount,
            Sandbox=SimpleNamespace(create=SimpleNamespace(aio=create_sandbox)),
        )

        class ImmediateWorker:
            def start(self):
                pass

            def stop(self):
                pass

            def run_coroutine(self, coro, timeout=600):
                return asyncio.run(coro)

        class NoopSyncManager:
            def __init__(self, **_kwargs):
                pass

            def sync(self, force=False):
                pass

        monkeypatch.setitem(sys.modules, "modal", fake_modal)
        monkeypatch.setattr(modal_env, "_ensure_modal_sdk", lambda: None)
        monkeypatch.setattr(modal_env, "_resolve_modal_image", lambda image: image)
        monkeypatch.setattr(modal_env, "_AsyncWorker", ImmediateWorker)
        monkeypatch.setattr(modal_env, "FileSyncManager", NoopSyncManager)
        monkeypatch.setattr(
            modal_env.ModalEnvironment, "init_session", lambda self: None
        )
        monkeypatch.setattr(cf, "get_credential_file_mounts", lambda: [])
        monkeypatch.setattr(cf, "iter_skills_files", lambda: [])
        monkeypatch.setattr(cf, "iter_cache_files", lambda: [])
        monkeypatch.setattr(
            cf,
            "iter_plans_files",
            lambda: [
                {
                    "host_path": "/host/plan.md",
                    "container_path": "/root/.hermes/plans/plan.md",
                }
            ],
        )

        modal_env.ModalEnvironment("test-image", persistent_filesystem=False)

        assert mounts == [("/host/plan.md", "/root/.hermes/plans/plan.md")]
        assert create_calls[0][1]["mounts"] == mounts


def test_managed_modal_stays_usable_when_plans_exist(monkeypatch):
    """Plans existing on the host must NOT make managed Modal unconstructible.

    Plans accumulate as a side effect of normal work, unlike credential mounts
    which a user explicitly opts into. Raising on their mere presence bricked
    every managed sandbox -- including ordinary sessions and cards referencing
    no plan -- and left auto-mode users without direct Modal credentials with
    no working backend at all.
    """
    import tools.credential_files as cf
    from tools.environments.managed_modal import ManagedModalEnvironment

    monkeypatch.setattr(cf, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(
        cf,
        "iter_plans_files",
        lambda *a, **k: [
            {
                "host_path": "/host/plan.md",
                "container_path": "/root/.hermes/plans/plan.md",
            }
        ],
    )

    env = object.__new__(ManagedModalEnvironment)
    env._guard_unsupported_credential_passthrough()


def test_managed_modal_still_refuses_credential_passthrough(monkeypatch):
    """The credential guard is opt-in and must keep failing loudly."""
    import tools.credential_files as cf
    from tools.environments.managed_modal import ManagedModalEnvironment

    monkeypatch.setattr(
        cf,
        "get_credential_file_mounts",
        lambda: [{"host_path": "/host/tok", "container_path": "/root/.hermes/tok"}],
    )

    env = object.__new__(ManagedModalEnvironment)
    with pytest.raises(ValueError, match="credential-file passthrough"):
        env._guard_unsupported_credential_passthrough()


def test_plans_are_in_the_recurring_sync_path(monkeypatch):
    """A plan edited after sandbox construction must still reach the worker.

    Creation-time mounts alone go stale for the life of a cached environment.
    """
    import tools.credential_files as cf

    monkeypatch.setattr(cf, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(cf, "iter_skills_files", lambda **k: [])
    monkeypatch.setattr(cf, "iter_cache_files", lambda **k: [])
    monkeypatch.setattr(
        cf,
        "iter_plans_files",
        lambda **k: [
            {
                "host_path": "/host/plan.md",
                "container_path": "/root/.hermes/plans/plan.md",
            }
        ],
    )

    from tools.environments.file_sync import iter_sync_files

    assert ("/host/plan.md", "/root/.hermes/plans/plan.md") in iter_sync_files(
        "/root/.hermes"
    )


def test_managed_modal_transports_plans_over_exec(monkeypatch, tmp_path):
    """Managed sandboxes must carry plans themselves, not reroute the backend.

    The gateway exposes no mount or upload primitive, so plans travel as a
    base64 payload on an exec. Rerouting auto-mode to direct instead would
    silently move ordinary sessions onto the user's own Modal credentials.
    """
    import base64
    import tools.credential_files as cf
    from tools.environments import managed_modal as mm

    plan = tmp_path / "task.md"
    plan.write_text("# the plan\nstep one\n")

    monkeypatch.setattr(cf, "get_credential_file_mounts", lambda: [])
    monkeypatch.setattr(
        cf,
        "iter_plans_files",
        lambda *a, **k: [
            {
                "host_path": str(plan),
                "container_path": "/root/.hermes/plans/task.md",
            }
        ],
    )

    calls = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"id": "sb-1"}

    def fake_request(self, method, path, **kwargs):
        calls.append((method, path, kwargs.get("json")))
        return _Resp()

    monkeypatch.setattr(mm.ManagedModalEnvironment, "_request", fake_request)
    monkeypatch.setattr(
        mm,
        "resolve_managed_tool_gateway",
        lambda _n: type("G", (), {"gateway_origin": "https://gw", "nous_user_token": "t"})(),
    )

    mm.ManagedModalEnvironment(image="img")

    exec_calls = [c for c in calls if c[1].endswith("/execs")]
    assert exec_calls, "no exec issued to transport plans"
    command = exec_calls[0][2]["command"]
    script = command[-1]
    assert "/root/.hermes/plans/task.md" in script
    # The plan's real bytes must be in the payload, not just its path.
    encoded = base64.b64encode(plan.read_bytes()).decode("ascii")
    assert encoded in script


def test_plans_do_not_reroute_auto_modal_selection(monkeypatch, tmp_path):
    """Unrelated plans must not flip an auto-mode session onto direct Modal.

    Doing so would spend the user's own Modal credentials and change execution
    behavior for sessions that reference no plan at all.
    """
    from tools.tool_backend_helpers import resolve_modal_backend_state

    state = resolve_modal_backend_state(
        "auto", has_direct=True, managed_ready=True, managed_enabled=True
    )
    assert state["selected_backend"] == "managed"
