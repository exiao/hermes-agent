"""Modal sandbox ``idle_timeout`` plumbing.

Regression cover for leaked sandboxes: a kanban worker exits (often via
``os._exit``, skipping cleanup), the sandbox keeps running ``sleep infinity``,
and nothing reaps it until the hard ``timeout`` lifetime expires -- up to an
hour of billed idle time per leaked worker.

``timeout`` is a MAXIMUM LIFETIME and kills mid-command, so it cannot be
lowered to fix this. ``idle_timeout`` is the inactivity reaper. These tests
assert the config value actually reaches ``Sandbox.create`` and that the two
knobs stay distinct.
"""

import inspect

import pytest

from tools import terminal_tool


def _container_config(**overrides):
    cfg = {
        "container_cpu": 1,
        "container_memory": 1024,
        "container_disk": 0,
        "container_persistent": False,
        "modal_mode": "direct",
    }
    cfg.update(overrides)
    return cfg


def _modal_sandbox_kwargs(monkeypatch, container_config):
    """Return the sandbox_kwargs _create_environment builds for modal."""
    captured = {}

    class _StubModalEnv:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        terminal_tool, "_get_modal_backend_state",
        lambda _mode: {
            "selected_backend": "direct",
            "mode": "direct",
            "managed_mode_blocked": False,
        },
    )
    monkeypatch.setattr(terminal_tool, "_ModalEnvironment", _StubModalEnv, raising=False)

    terminal_tool._create_environment(
        env_type="modal",
        image="debian",
        cwd="/root",
        timeout=120,
        ssh_config={},
        container_config=container_config,
        local_config={},
        task_id="t_idle",
        host_cwd="/root",
    )
    return captured.get("modal_sandbox_kwargs", {})


class TestIdleTimeoutPlumbing:
    def test_idle_timeout_forwarded_when_configured(self, monkeypatch):
        kwargs = _modal_sandbox_kwargs(
            monkeypatch, _container_config(container_idle_timeout=300)
        )
        assert kwargs.get("idle_timeout") == 300

    def test_absent_when_disabled(self, monkeypatch):
        """0 must preserve the pre-change behavior: no kwarg at all."""
        kwargs = _modal_sandbox_kwargs(
            monkeypatch, _container_config(container_idle_timeout=0)
        )
        assert "idle_timeout" not in kwargs

    def test_absent_when_unset(self, monkeypatch):
        kwargs = _modal_sandbox_kwargs(monkeypatch, _container_config())
        assert "idle_timeout" not in kwargs

    def test_non_numeric_is_ignored_not_fatal(self, monkeypatch):
        """A bad config value must not break sandbox creation."""
        kwargs = _modal_sandbox_kwargs(
            monkeypatch, _container_config(container_idle_timeout="soon")
        )
        assert "idle_timeout" not in kwargs


class TestMalformedEnvIsNonFatal:
    """A bad value must disable the feature, never block the terminal.

    Exercised through the real ``_get_env_config()`` rather than by hand-building
    container_config: the original bug was that parsing raised *inside* that
    function, so a test that skipped it reported success while
    TERMINAL_ENV=modal + a typo made every environment creation fail.
    """

    @pytest.mark.parametrize("bad", ["soon", "5m", "", "1.5", "-30"])
    def test_bad_value_disables_instead_of_raising(self, monkeypatch, bad):
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.setenv("TERMINAL_CONTAINER_IDLE_TIMEOUT", bad)
        cfg = terminal_tool._get_env_config()
        assert cfg["container_idle_timeout"] == 0

    def test_valid_value_still_parsed(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.setenv("TERMINAL_CONTAINER_IDLE_TIMEOUT", "300")
        assert terminal_tool._get_env_config()["container_idle_timeout"] == 300

    def test_unset_defaults_to_disabled(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.delenv("TERMINAL_CONTAINER_IDLE_TIMEOUT", raising=False)
        assert terminal_tool._get_env_config()["container_idle_timeout"] == 0


class TestGatewayEnvMapParity:
    """The gateway keeps its OWN terminal env map; drift silently disables keys.

    The gateway is how kanban workers run, so a key missing there disables the
    setting for exactly the workload that leaks sandboxes -- with config.yaml
    still showing it as configured. Asserted as a parity invariant rather than
    a fixed key list so the next added key is caught too.
    """

    def _gateway_map(self):
        import re
        from pathlib import Path
        import gateway.run as gw

        src = Path(gw.__file__).read_text()
        block = re.search(
            r'_terminal_env_map\s*=\s*\{(.*?)\n\s*\}', src, re.S
        )
        assert block, "could not locate _terminal_env_map in gateway/run.py"
        return set(re.findall(r'"(\w+)":\s*"TERMINAL_', block.group(1)))

    def test_idle_timeout_is_bridged(self):
        assert "container_idle_timeout" in self._gateway_map()

    def test_no_container_key_is_dropped(self):
        """Every container_* key the CLI bridges must exist in the gateway too."""
        from hermes_cli import config as config_mod

        cli_maps = [
            v for v in vars(config_mod).values()
            if isinstance(v, dict) and "container_persistent" in v
        ]
        assert cli_maps
        cli_keys = {
            k for m in cli_maps for k in m if k.startswith("container_")
        }
        missing = cli_keys - self._gateway_map()
        assert not missing, f"gateway env map is missing: {sorted(missing)}"


class TestIdleTimeoutOutlivesLocalReaper:
    """The clamp is what removes the need for eviction machinery.

    Hermes caches a ModalEnvironment and reaps it locally after
    ``terminal.lifetime_seconds`` of inactivity. If Modal's idle timeout were
    SHORTER, the provider could delete a sandbox that Hermes still holds in
    ``_active_environments``, and the next command would reuse a dead sandbox.
    Keeping the provider window strictly larger makes that state unreachable,
    so no eviction/recovery path has to exist.
    """

    def test_short_value_is_raised_above_the_local_reaper(self, monkeypatch):
        kwargs = _modal_sandbox_kwargs(
            monkeypatch,
            _container_config(container_idle_timeout=60, lifetime_seconds=300),
        )
        assert kwargs["idle_timeout"] >= 600

    def test_generous_value_is_left_alone(self, monkeypatch):
        kwargs = _modal_sandbox_kwargs(
            monkeypatch,
            _container_config(container_idle_timeout=1800, lifetime_seconds=300),
        )
        assert kwargs["idle_timeout"] == 1800

    def test_clamp_reads_the_real_config_default(self, monkeypatch):
        """lifetime_seconds must reach container_config or the clamp no-ops."""
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.setenv("TERMINAL_CONTAINER_IDLE_TIMEOUT", "60")
        cfg = terminal_tool._get_env_config()
        assert cfg.get("lifetime_seconds", 0) > 0


class TestConfigSurface:
    def test_default_is_registered_and_disabled(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        terminal_defaults = DEFAULT_CONFIG["terminal"]
        assert "container_idle_timeout" in terminal_defaults
        assert terminal_defaults["container_idle_timeout"] == 0

    def test_env_var_bridge_exists(self):
        """Without the bridge the config.yaml key never reaches the tool."""
        from hermes_cli import config as config_mod

        found = [
            v for k, v in vars(config_mod).items()
            if isinstance(v, dict) and "container_persistent" in v
        ]
        assert found, "expected a terminal env-var mapping dict"
        assert any(
            m.get("container_idle_timeout") == "TERMINAL_CONTAINER_IDLE_TIMEOUT"
            for m in found
        )


class TestSemanticsAreDistinct:
    def test_timeout_and_idle_timeout_are_separate_sdk_params(self):
        """Guards the misreading that caused the original bug.

        ``timeout`` is a hard lifetime; ``idle_timeout`` is inactivity. If a
        future SDK collapses them, capping idle at the lifetime (below) and the
        whole fix would need rethinking.
        """
        modal = pytest.importorskip("modal")
        params = inspect.signature(modal.Sandbox.create).parameters
        assert "timeout" in params
        assert "idle_timeout" in params
        assert params["idle_timeout"].default is None

    def test_idle_never_exceeds_hard_lifetime(self):
        """An idle window longer than the lifetime is meaningless; clamp it."""
        from tools.environments.modal import _clamp_idle_timeout

        assert _clamp_idle_timeout(9999, 100) == 100
        assert _clamp_idle_timeout(60, 3600) == 60
        assert _clamp_idle_timeout(0, 3600) == 0
        assert _clamp_idle_timeout(None, 3600) == 0
