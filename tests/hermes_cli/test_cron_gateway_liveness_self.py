"""The cron liveness probe must not deny the gateway that is running it.

``find_gateway_pids()`` deliberately excludes the calling process and its whole
ancestor chain (#13242) so ``hermes gateway status`` never counts itself. That
makes it the wrong sole oracle for a probe that also runs INSIDE the gateway:
the ``cronjob`` model tool, cron-run agents, and terminal children of the
gateway all inherit that exclusion and would see zero gateway PIDs while the
gateway is demonstrably alive, emitting "gateway is not running" to a user whose
scheduler is fine.
"""

import gateway.status as status_mod
import hermes_cli.cron as cron_mod
import hermes_cli.gateway as gateway_mod


def test_liveness_true_when_caller_is_the_gateway(monkeypatch):
    """PID file says we ARE the gateway; the self-excluding scan sees nothing."""
    monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "builtin")
    monkeypatch.setattr(status_mod, "is_gateway_running", lambda **kw: True)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda *a, **k: [])

    assert cron_mod._builtin_gateway_liveness() is True


def test_liveness_falls_back_to_scan_when_pid_file_is_stale(monkeypatch):
    """A missing/stale PID file must still find a live gateway by process scan."""
    monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "builtin")
    monkeypatch.setattr(status_mod, "is_gateway_running", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda *a, **k: [4242])

    assert cron_mod._builtin_gateway_liveness() is True


def test_liveness_false_when_nothing_is_running(monkeypatch):
    """Both oracles empty is the only state that may warn the user."""
    monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "builtin")
    monkeypatch.setattr(status_mod, "is_gateway_running", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda *a, **k: [])

    assert cron_mod._builtin_gateway_liveness() is False


def test_non_builtin_provider_still_exempt(monkeypatch):
    """A third-party scheduler fires without the gateway; never probe for it."""
    monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "chronos")

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("provider exemption must short-circuit the probe")

    monkeypatch.setattr(status_mod, "is_gateway_running", _boom)

    assert cron_mod._builtin_gateway_liveness() is True
