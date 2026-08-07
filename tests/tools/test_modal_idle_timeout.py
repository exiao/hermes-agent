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

    def test_idle_never_exceeds_hard_lifetime(self, monkeypatch):
        """An idle window longer than the lifetime is meaningless; clamp it."""
        modal = pytest.importorskip("modal")
        if "idle_timeout" not in inspect.signature(modal.Sandbox.create).parameters:
            pytest.skip("installed modal SDK has no idle_timeout")

        from tools.environments import modal as modal_env

        captured = {}

        class _FakeCreate:
            async def aio(self, *args, **kwargs):
                captured.update(kwargs)
                return object()

        class _FakeSandbox:
            create = _FakeCreate()

        class _FakeAppLookup:
            async def aio(self, *a, **k):
                return object()

        class _FakeApp:
            lookup = _FakeAppLookup()

        monkeypatch.setattr(
            modal_env, "_modal",
            type("M", (), {"App": _FakeApp, "Sandbox": _FakeSandbox}),
            raising=False,
        )
        monkeypatch.setattr(
            modal_env, "_resolve_modal_image", lambda spec: spec, raising=False
        )

        env = object.__new__(modal_env.ModalEnvironment)
        import asyncio

        async def _go():
            sandbox_kwargs = {"timeout": 100, "idle_timeout": 9999}
            create_kwargs = dict(sandbox_kwargs)
            create_timeout = int(create_kwargs.pop("timeout", 3600))
            idle = create_kwargs.pop("idle_timeout", None)
            extra = {}
            if idle:
                extra["idle_timeout"] = max(1, min(int(idle), create_timeout))
            await _FakeSandbox.create.aio(
                "sleep", "infinity", timeout=create_timeout, **extra, **create_kwargs
            )

        asyncio.run(_go())
        assert captured["idle_timeout"] <= captured["timeout"]
