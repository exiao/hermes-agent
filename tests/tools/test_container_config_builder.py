"""Every caller of ``_create_environment`` must pass the SAME container config.

The four call sites used to hand-copy their own subset of keys, and the subsets
drifted: ``file_tools`` and ``code_execution_tool`` never passed ``modal_mode``,
``docker_shm_size``, ``docker_extra_args`` or ``docker_persist_across_processes``.
All four share one environment cache, so the same config.yaml produced a
different container depending on which tool created the environment first.

These are behavioral tests: each caller is driven for real and the
``container_config`` it hands to ``_create_environment`` is captured. A
source-shape test would pass on a dict that is built correctly but never
reaches the callee.
"""

import pytest

from tools import terminal_tool
from tools.terminal_tool import (
    CONTAINER_ENV_TYPES,
    _CONTAINER_CONFIG_DEFAULTS,
    build_container_config,
)

# Keys the smaller hand-copied dicts dropped. Regression cover for the drift.
PREVIOUSLY_DROPPED = (
    "modal_mode",
    "docker_shm_size",
    "docker_extra_args",
    "docker_persist_across_processes",
)


@pytest.fixture
def captured_config(monkeypatch):
    """Capture the container_config each caller passes to _create_environment.

    Callers import ``_create_environment`` lazily inside the function body, so
    patching the attribute on the module reaches every one of them.
    """
    seen = {}

    class _StubEnv:
        def __init__(self, *a, **kw):
            pass

        def execute(self, *a, **kw):
            return {"output": "", "exit_code": 0}

        def cleanup(self):
            pass

    def _fake_create(**kwargs):
        seen["container_config"] = kwargs.get("container_config")
        seen["env_type"] = kwargs.get("env_type")
        return _StubEnv()

    monkeypatch.setattr(terminal_tool, "_create_environment", _fake_create)
    # docker: the only backend for which _get_env_config reads the docker_*
    # env vars at all, so it exercises the widest set of keys.
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_CONTAINER_IDLE_TIMEOUT", "900")
    monkeypatch.setenv("TERMINAL_DOCKER_SHM_SIZE", "4g")
    # Each caller keys off its own cache; clear so creation actually runs.
    terminal_tool._active_environments.clear()
    yield seen
    terminal_tool._active_environments.clear()


def _drive_file_tools(task_id):
    from tools.file_tools import _get_file_ops
    _get_file_ops(task_id)


def _drive_code_execution(task_id):
    from tools.code_execution_tool import _get_or_create_env
    _get_or_create_env(task_id)


CALLERS = {
    "file_tools": _drive_file_tools,
    "code_execution_tool": _drive_code_execution,
}


class TestCallersAgree:
    @pytest.mark.parametrize("name", sorted(CALLERS))
    def test_caller_passes_the_full_container_config(self, name, captured_config):
        """The captured dict must match the builder exactly, key for key."""
        CALLERS[name](f"t_{name}")
        cc = captured_config["container_config"]
        assert cc is not None, f"{name} passed no container_config"
        assert cc == build_container_config(captured_config["env_type"], cc)

    @pytest.mark.parametrize("name", sorted(CALLERS))
    def test_caller_carries_the_keys_the_drift_dropped(self, name, captured_config):
        CALLERS[name](f"t_drift_{name}")
        cc = captured_config["container_config"]
        missing = [k for k in PREVIOUSLY_DROPPED if k not in cc]
        assert not missing, f"{name} dropped: {missing}"

    @pytest.mark.parametrize("name", sorted(CALLERS))
    def test_user_config_reaches_the_environment(self, name, captured_config):
        """A configured value must survive the whole hop, not just the default."""
        CALLERS[name](f"t_cfg_{name}")
        cc = captured_config["container_config"]
        assert cc["container_idle_timeout"] == 900
        assert cc["docker_shm_size"] == "4g"

    def test_all_callers_produce_identical_config(self, captured_config):
        """The actual bug: which tool creates the env first must not matter."""
        results = {}
        for name, drive in sorted(CALLERS.items()):
            terminal_tool._active_environments.clear()
            drive(f"t_same_{name}")
            results[name] = captured_config["container_config"]
        first, *rest = results.values()
        for other in rest:
            assert other == first


class TestBuilder:
    @pytest.mark.parametrize("env_type", sorted(CONTAINER_ENV_TYPES))
    def test_every_container_backend_gets_every_key(self, env_type):
        assert set(build_container_config(env_type, {})) == set(
            _CONTAINER_CONFIG_DEFAULTS
        )

    @pytest.mark.parametrize("env_type", ["local", "ssh", ""])
    def test_non_container_backends_get_none(self, env_type):
        assert build_container_config(env_type, {}) is None

    def test_user_config_wins_over_defaults(self):
        cfg = build_container_config("modal", {"container_idle_timeout": 900})
        assert cfg["container_idle_timeout"] == 900
        assert cfg["modal_mode"] == _CONTAINER_CONFIG_DEFAULTS["modal_mode"]

    def test_result_is_not_shared_state(self):
        """A mutable default must not leak between environments."""
        first = build_container_config("docker", {})
        first["docker_volumes"].append("/tmp:/tmp")
        assert build_container_config("docker", {})["docker_volumes"] == []

    def test_docker_network_accompanies_run_as_host_user(self):
        """Lockdown invariant from issue #46358.

        A config carrying ``docker_run_as_host_user`` without ``docker_network``
        silently falls back to networked containers on that path while the
        terminal path honors the lockdown.
        """
        cfg = build_container_config("docker", {})
        assert "docker_run_as_host_user" in cfg
        assert "docker_network" in cfg
