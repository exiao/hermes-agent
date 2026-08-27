"""The cron liveness probe must not deny the gateway that is running it."""

import pytest

import gateway.status as status_mod
import hermes_cli.cron as cron_mod
import hermes_cli.gateway as gateway_mod


@pytest.mark.parametrize(
    "name, pid_file_says, scan_says, expected",
    [
        # The regression: probe runs INSIDE the gateway, so find_gateway_pids()
        # excludes its own ancestor chain (#13242) and sees nothing.
        ("caller is the gateway", True, [], True),
        ("pid file stale, scan finds it", False, [4242], True),
        ("genuinely down", False, [], False),
    ],
)
def test_builtin_liveness(monkeypatch, name, pid_file_says, scan_says, expected):
    monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "builtin")
    monkeypatch.setattr(status_mod, "is_gateway_running", lambda **kw: pid_file_says)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda *a, **k: scan_says)

    assert cron_mod._builtin_gateway_liveness() is expected


def test_non_builtin_provider_never_probes(monkeypatch):
    """A third-party scheduler fires without the gateway; probing it is wrong."""
    monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "chronos")
    monkeypatch.setattr(
        status_mod, "is_gateway_running", lambda **kw: pytest.fail("probed")
    )

    assert cron_mod._builtin_gateway_liveness() is True
