import json
import os
import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"


def _load_module(module_name: str, path: Path):
    spec = spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _reset_modules(prefixes: tuple[str, ...]):
    for name in list(sys.modules):
        if name.startswith(prefixes):
            sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _restore_tool_modules():
    original_hermes_home = os.environ.get("HERMES_HOME")
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "tools"
        or name.startswith("tools.")
        or name == "hermes_cli"
        or name.startswith("hermes_cli.")
        or name == "modal"
        or name.startswith("modal.")
    }
    try:
        yield
    finally:
        if original_hermes_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = original_hermes_home
        _reset_modules(("tools", "hermes_cli", "modal"))
        sys.modules.update(original_modules)


def _install_modal_test_modules(
    tmp_path: Path,
    *,
    fail_on_snapshot_ids: set[str] | None = None,
    snapshot_id: str = "im-fresh",
    purge_exit_code: int = 0,
    purge_raises: bool = False,
    credential_mounts: list[dict[str, str]] | None = None,
    sync_mounts: list[dict[str, str]] | None = None,
):
    _reset_modules(("tools", "hermes_cli", "modal"))

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    sys.modules["hermes_cli"] = hermes_cli
    hermes_home = tmp_path / "hermes-home"
    os.environ["HERMES_HOME"] = str(hermes_home)
    sys.modules["hermes_cli.config"] = types.SimpleNamespace(
        get_hermes_home=lambda: hermes_home,
    )

    tools_package = types.ModuleType("tools")
    tools_package.__path__ = [str(TOOLS_DIR)]  # type: ignore[attr-defined]
    sys.modules["tools"] = tools_package

    env_package = types.ModuleType("tools.environments")
    env_package.__path__ = [str(TOOLS_DIR / "environments")]  # type: ignore[attr-defined]
    sys.modules["tools.environments"] = env_package

    class _DummyBaseEnvironment:
        def __init__(self, cwd: str, timeout: int, env=None):
            self.cwd = cwd
            self.timeout = timeout
            self.env = env or {}

        def _prepare_command(self, command: str):
            return command, None

        def init_session(self):
            pass

    # Stub _ThreadedProcessHandle: modal.py imports it but only uses it at
    # runtime inside _run_bash; the snapshot-isolation tests never call _run_bash,
    # so a class placeholder is sufficient.
    class _DummyThreadedProcessHandle:
        def __init__(self, exec_fn, cancel_fn=None):
            pass

    def _load_json_store(path):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {}

    def _save_json_store(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def _file_mtime_key(host_path):
        try:
            st = Path(host_path).stat()
            return (st.st_mtime, st.st_size)
        except OSError:
            return None

    sys.modules["tools.environments.base"] = types.SimpleNamespace(
        BaseEnvironment=_DummyBaseEnvironment,
        _ThreadedProcessHandle=_DummyThreadedProcessHandle,
        _load_json_store=_load_json_store,
        _save_json_store=_save_json_store,
        _file_mtime_key=_file_mtime_key,
    )
    sys.modules["tools.interrupt"] = types.SimpleNamespace(is_interrupted=lambda: False)
    sys.modules["tools.credential_files"] = types.SimpleNamespace(
        get_credential_file_mounts=lambda: credential_mounts or [],
        iter_skills_files=lambda **kw: sync_mounts or [],
        iter_plans_files=lambda **kw: [],
        iter_cache_files=lambda **kw: [],
        _CACHE_DIRS=[("cache/documents", "document_cache")],
    )

    from_id_calls: list[str] = []
    registry_calls: list[tuple[str, list[str] | None]] = []
    create_calls: list[dict] = []
    exec_calls: list[tuple] = []
    sandbox_instances: list = []

    class _FakeImage:
        @staticmethod
        def from_id(image_id: str):
            from_id_calls.append(image_id)
            return {"kind": "snapshot", "image_id": image_id}

        @staticmethod
        def from_registry(image: str, setup_dockerfile_commands=None):
            registry_calls.append((image, setup_dockerfile_commands))
            return {"kind": "registry", "image": image}

    async def _lookup_aio(_name: str, create_if_missing: bool = False):
        return types.SimpleNamespace(name="hermes-agent", create_if_missing=create_if_missing)

    class _FakeSandboxInstance:
        def __init__(self, image):
            self.image = image
            self.terminated = False

            async def _snapshot_aio():
                return types.SimpleNamespace(object_id=snapshot_id)

            async def _terminate_aio():
                self.terminated = True
                return None

            async def _exec_aio(*cmd):
                exec_calls.append(cmd)
                is_purge = any("rm -rf" in str(a) for a in cmd)
                if is_purge and purge_raises:
                    raise RuntimeError("exec transport failed")

                async def _wait_aio():
                    return purge_exit_code if is_purge else 0

                return types.SimpleNamespace(wait=types.SimpleNamespace(aio=_wait_aio))

            self.exec = types.SimpleNamespace(aio=_exec_aio)
            self.snapshot_filesystem = types.SimpleNamespace(aio=_snapshot_aio)
            self.terminate = types.SimpleNamespace(aio=_terminate_aio)
            sandbox_instances.append(self)

    async def _create_aio(*_args, image=None, app=None, timeout=None, **kwargs):
        create_calls.append({
            "image": image,
            "app": app,
            "timeout": timeout,
            **kwargs,
        })
        image_id = image.get("image_id") if isinstance(image, dict) else None
        if fail_on_snapshot_ids and image_id in fail_on_snapshot_ids:
            raise RuntimeError(f"cannot restore {image_id}")
        return _FakeSandboxInstance(image)

    class _FakeMount:
        @staticmethod
        def from_local_file(host_path: str, remote_path: str):
            return {"host_path": host_path, "remote_path": remote_path}

    class _FakeApp:
        lookup = types.SimpleNamespace(aio=_lookup_aio)

    class _FakeSandbox:
        create = types.SimpleNamespace(aio=_create_aio)

    sys.modules["modal"] = types.SimpleNamespace(
        Image=_FakeImage,
        App=_FakeApp,
        Sandbox=_FakeSandbox,
        Mount=_FakeMount,
    )

    return {
        "snapshot_store": hermes_home / "modal_snapshots.json",
        "create_calls": create_calls,
        "from_id_calls": from_id_calls,
        "registry_calls": registry_calls,
        "exec_calls": exec_calls,
        "sandbox_instances": sandbox_instances,
    }


def test_modal_environment_migrates_legacy_snapshot_key_and_uses_snapshot_id(tmp_path):
    state = _install_modal_test_modules(tmp_path)
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"task-legacy": "im-legacy123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-legacy")

    try:
        assert state["from_id_calls"] == ["im-legacy123"]
        assert state["create_calls"][0]["image"] == {"kind": "snapshot", "image_id": "im-legacy123"}
        assert json.loads(snapshot_store.read_text()) == {"direct:task-legacy": "im-legacy123"}
    finally:
        env.cleanup()


def test_modal_environment_prunes_stale_direct_snapshot_and_retries_base_image(tmp_path):
    state = _install_modal_test_modules(tmp_path, fail_on_snapshot_ids={"im-stale123"})
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"direct:task-stale": "im-stale123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-stale")

    try:
        assert [call["image"] for call in state["create_calls"]] == [
            {"kind": "snapshot", "image_id": "im-stale123"},
            {"kind": "registry", "image": "python:3.11"},
        ]
        assert json.loads(snapshot_store.read_text()) == {}
    finally:
        env.cleanup()


def test_modal_environment_cleanup_writes_namespaced_snapshot_key(tmp_path):
    state = _install_modal_test_modules(tmp_path, snapshot_id="im-cleanup456")
    snapshot_store = state["snapshot_store"]

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-cleanup")
    env.cleanup()

    assert json.loads(snapshot_store.read_text()) == {"direct:task-cleanup": "im-cleanup456"}


def test_resolve_modal_image_uses_snapshot_ids_and_registry_images(tmp_path):
    state = _install_modal_test_modules(tmp_path)
    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")

    snapshot_image = modal_module._resolve_modal_image("im-snapshot123")
    registry_image = modal_module._resolve_modal_image("python:3.11")

    assert snapshot_image == {"kind": "snapshot", "image_id": "im-snapshot123"}
    assert registry_image == {"kind": "registry", "image": "python:3.11"}
    assert state["from_id_calls"] == ["im-snapshot123"]
    assert state["registry_calls"][0][0] == "python:3.11"
    assert "ensurepip" in state["registry_calls"][0][1][0]


def test_restored_sandbox_purges_sync_owned_subtrees_before_first_sync(tmp_path):
    """A fresh FileSyncManager issues no deletes for the prior instance's uploads.

    Without the purge, a plan/skill/cache file removed from the host survives in
    the restored sandbox for that sandbox's whole lifetime.
    """
    state = _install_modal_test_modules(tmp_path)
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"direct:task-restore": "im-restore123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-restore")

    try:
        purges = [c for c in state["exec_calls"] if any("rm -rf" in str(a) for a in c)]
        assert len(purges) == 1, state["exec_calls"]
        cmd = purges[0][-1]
        for root in ("/root/.hermes/plans", "/root/.hermes/skills",
                     "/root/.hermes/external_skills", "/root/.hermes/cache/documents"):
            assert root in cmd
        assert "rm -rf /root/.hermes " not in cmd
    finally:
        env.cleanup()


def test_restored_sandbox_keeps_credentials_but_not_sync_mounts(tmp_path):
    """Restored snapshots keep credential mounts while sync paths stay writable."""
    state = _install_modal_test_modules(
        tmp_path,
        credential_mounts=[
            {"host_path": "/host/token.json", "container_path": "/root/.hermes/token.json"}
        ],
        sync_mounts=[
            {"host_path": "/host/skill.md", "container_path": "/root/.hermes/skills/skill.md"}
        ],
    )
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"direct:task-mounted": "im-restore123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-mounted")

    try:
        assert state["create_calls"][0]["mounts"] == [
            {"host_path": "/host/token.json", "remote_path": "/root/.hermes/token.json"}
        ]
    finally:
        env.cleanup()


def test_fresh_sandbox_does_not_purge(tmp_path):
    """No snapshot means nothing stale; purging would be wasted work."""
    state = _install_modal_test_modules(tmp_path)

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-fresh")

    try:
        assert not [c for c in state["exec_calls"] if any("rm -rf" in str(a) for a in c)]
    finally:
        env.cleanup()


def test_failed_restore_falls_back_to_base_image_without_purge(tmp_path):
    """The retry sandbox is a clean base image, so there is nothing to reconcile."""
    state = _install_modal_test_modules(tmp_path, fail_on_snapshot_ids={"im-stale123"})
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"direct:task-stale2": "im-stale123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-stale2")

    try:
        assert not [c for c in state["exec_calls"] if any("rm -rf" in str(a) for a in c)]
    finally:
        env.cleanup()


def test_failed_purge_aborts_instead_of_running_on_stale_state(tmp_path):
    """A nonzero purge must not be swallowed.

    `sync(force=True)` only uploads files still on the host; it can never
    discover remote paths the previous instance left behind. So if the purge
    silently fails, the sandbox keeps serving exactly the deleted plans and
    skills this PR exists to remove -- and nothing downstream can detect it.
    Failing loudly is the only safe outcome.
    """
    state = _install_modal_test_modules(tmp_path, purge_exit_code=1)
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"direct:task-badpurge": "im-restore123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")

    with pytest.raises(RuntimeError, match="purge"):
        modal_module.ModalEnvironment(image="python:3.11", task_id="task-badpurge")

    # The half-built sandbox must not be leaked when construction aborts.
    assert state["sandbox_instances"], "no sandbox was created"
    assert all(s.terminated for s in state["sandbox_instances"])


def test_purge_transport_error_does_not_leak_the_sandbox(tmp_path):
    """An exception from exec.aio escapes outside the constructor's cleanup.

    The purge runs after the create try/except that stops the worker, so a
    transport failure would otherwise leave a live sandbox and a running
    worker thread behind for the sandbox's full timeout.
    """
    state = _install_modal_test_modules(tmp_path, purge_raises=True)
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"direct:task-purgeboom": "im-restore123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")

    with pytest.raises(RuntimeError):
        modal_module.ModalEnvironment(image="python:3.11", task_id="task-purgeboom")

    assert state["sandbox_instances"], "no sandbox was created"
    assert all(s.terminated for s in state["sandbox_instances"])


def test_legacy_snapshot_store_error_does_not_leak_the_sandbox(tmp_path, monkeypatch):
    """Legacy-key migration errors must clean up the already-created sandbox."""
    state = _install_modal_test_modules(tmp_path)
    snapshot_store = state["snapshot_store"]
    snapshot_store.parent.mkdir(parents=True, exist_ok=True)
    snapshot_store.write_text(json.dumps({"task-legacy-store": "im-restore123"}))

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")

    def fail_store(*_args):
        raise RuntimeError("snapshot store failed")

    monkeypatch.setattr(modal_module, "_store_direct_snapshot", fail_store)
    with pytest.raises(RuntimeError, match="snapshot store failed"):
        modal_module.ModalEnvironment(image="python:3.11", task_id="task-legacy-store")

    assert state["sandbox_instances"], "no sandbox was created"
    assert all(s.terminated for s in state["sandbox_instances"])


def test_late_synced_credentials_are_removed_before_snapshot(tmp_path):
    """Credentials registered after construction must not enter snapshots."""
    late_mounts: list[dict[str, str]] = []
    state = _install_modal_test_modules(tmp_path, credential_mounts=late_mounts)
    late_host_path = tmp_path / "late-token.json"
    late_host_path.write_text("secret")

    modal_module = _load_module("tools.environments.modal", TOOLS_DIR / "environments" / "modal.py")
    env = modal_module.ModalEnvironment(image="python:3.11", task_id="task-late-credential")

    late_mounts.append({
        "host_path": str(late_host_path),
        "container_path": "/root/.hermes/late-token.json",
    })
    env._before_execute()
    env._sync_manager.sync_back = lambda: None
    env.cleanup()

    removals = [
        call for call in state["exec_calls"]
        if any("rm -f" in str(arg) for arg in call)
    ]
    assert len(removals) == 1, state["exec_calls"]
    assert removals[0][-1] == "rm -f /root/.hermes/late-token.json"
