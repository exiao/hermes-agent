"""Modal sandbox ``idle_timeout`` plumbing.

A kanban worker exits (often via ``os._exit``, skipping cleanup), the sandbox
keeps running ``sleep infinity``, and nothing reaps it until the hard
``timeout`` lifetime expires -- up to an hour of billed idle time per leak.

``timeout`` is a MAXIMUM LIFETIME and kills mid-command, so it cannot be
lowered to fix this. ``idle_timeout`` is the inactivity reaper.
"""

import pytest

from tools import terminal_tool


def _sandbox_kwargs(monkeypatch, **cfg):
    """Return the modal sandbox_kwargs _create_environment builds."""
    captured = {}

    class _Stub:
        def __init__(self, *a, **kw):
            captured.update(kw)

    monkeypatch.setattr(
        terminal_tool, "_get_modal_backend_state",
        lambda _m: {"selected_backend": "direct", "mode": "direct",
                    "managed_mode_blocked": False},
    )
    monkeypatch.setattr(terminal_tool, "_ModalEnvironment", _Stub, raising=False)

    container_config = {
        "container_cpu": 1, "container_memory": 1024, "container_disk": 0,
        "container_persistent": False, "modal_mode": "direct", **cfg,
    }
    terminal_tool._create_environment(
        env_type="modal", image="debian", cwd="/root", timeout=3600,
        ssh_config={}, container_config=container_config, local_config={},
        task_id="t_idle", host_cwd="/root",
    )
    return captured.get("modal_sandbox_kwargs", {})


class TestIdleTimeoutPlumbing:
    def test_forwarded_when_configured(self, monkeypatch):
        kwargs = _sandbox_kwargs(monkeypatch, container_idle_timeout=900)
        assert kwargs.get("idle_timeout") == 900

    @pytest.mark.parametrize("value", [0, None, "soon"])
    def test_absent_unless_valid(self, monkeypatch, value):
        """0/unset/garbage must all preserve pre-change behavior: no kwarg."""
        kwargs = _sandbox_kwargs(monkeypatch, container_idle_timeout=value)
        assert "idle_timeout" not in kwargs

    def test_raised_above_local_reaper(self, monkeypatch):
        """Modal must not delete a sandbox Hermes still holds in its cache."""
        kwargs = _sandbox_kwargs(
            monkeypatch, container_idle_timeout=60, lifetime_seconds=300
        )
        assert kwargs["idle_timeout"] == 600

    def test_never_exceeds_hard_lifetime(self, monkeypatch):
        kwargs = _sandbox_kwargs(monkeypatch, container_idle_timeout=99999)
        assert kwargs["idle_timeout"] == 3600


class TestMalformedEnvIsNonFatal:
    """A bad value must disable the feature, never block the terminal.

    Exercised through the real ``_get_env_config()``: the failure mode is a
    raise *inside* that function, which breaks every environment creation.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("soon", 0), ("5m", 0), ("", 0), ("1.5", 0), ("-30", 0), ("300", 300),
    ])
    def test_parsing(self, monkeypatch, raw, expected):
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.setenv("TERMINAL_CONTAINER_IDLE_TIMEOUT", raw)
        assert terminal_tool._get_env_config()["container_idle_timeout"] == expected

    def test_unset_defaults_to_disabled(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_ENV", "modal")
        monkeypatch.delenv("TERMINAL_CONTAINER_IDLE_TIMEOUT", raising=False)
        cfg = terminal_tool._get_env_config()
        assert cfg["container_idle_timeout"] == 0
        # The clamp silently no-ops if lifetime_seconds never reaches the tool.
        assert cfg.get("lifetime_seconds", 0) > 0


class TestConfigSurface:
    """Every hop the value must survive: default -> env bridge -> gateway."""

    def _cli_maps(self):
        from hermes_cli import config as config_mod
        maps = [v for v in vars(config_mod).values()
                if isinstance(v, dict) and "container_persistent" in v]
        assert maps, "expected a terminal env-var mapping dict"
        return maps

    def test_default_is_registered_and_disabled(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["terminal"]["container_idle_timeout"] == 0

    def test_env_var_bridge_exists(self):
        """Without the bridge the config.yaml key never reaches the tool."""
        assert any(
            m.get("container_idle_timeout") == "TERMINAL_CONTAINER_IDLE_TIMEOUT"
            for m in self._cli_maps()
        )

    def test_no_container_key_is_dropped_by_the_gateway(self):
        """The gateway keeps its OWN map; drift silently disables keys there.

        Kanban workers run through the gateway, so a key missing here disables
        the setting for exactly the workload that leaks. Asserted as a parity
        invariant so the next added key is caught too.
        """
        import re
        from pathlib import Path
        import gateway.run as gw

        block = re.search(r"_terminal_env_map\s*=\s*\{(.*?)\n\s*\}",
                          Path(gw.__file__).read_text(), re.S)
        assert block, "could not locate _terminal_env_map in gateway/run.py"
        gateway_keys = set(re.findall(r'"(\w+)":\s*"TERMINAL_', block.group(1)))

        cli_keys = {k for m in self._cli_maps() for k in m
                    if k.startswith("container_")}
        assert "container_idle_timeout" in gateway_keys
        assert not (cli_keys - gateway_keys), \
            f"gateway env map is missing: {sorted(cli_keys - gateway_keys)}"


def test_timeout_and_idle_timeout_are_separate_sdk_params():
    """Guards the misreading that caused the original bug."""
    import inspect
    modal = pytest.importorskip("modal")
    params = inspect.signature(modal.Sandbox.create).parameters
    assert "timeout" in params and "idle_timeout" in params
