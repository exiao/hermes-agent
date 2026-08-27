"""The cron liveness probe must not deny the gateway that is running it.

``find_gateway_pids()`` deliberately excludes the calling process and its whole
ancestor chain (#13242) so ``hermes gateway status`` never counts itself. That
makes it the wrong sole oracle for a probe that also runs INSIDE the gateway:
the ``cronjob`` model tool, cron-run agents, and terminal children of the
gateway all inherit that exclusion and would see zero gateway PIDs while the
gateway is demonstrably alive, emitting "gateway is not running" to a user whose
scheduler is fine.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import gateway.status as status_mod
import hermes_cli.cron as cron_mod
import hermes_cli.gateway as gateway_mod


def test_liveness_true_when_gateway_process_holds_runtime_lock(tmp_path, monkeypatch):
    """A live gateway-like process is recognized through the real status path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cron_mod, "_active_cron_provider_name", lambda: "builtin")

    gateway_script = tmp_path / "hermes-gateway"
    gateway_script.write_text(
        "import os, sys\n"
        f"os.environ['HERMES_HOME'] = {str(tmp_path)!r}\n"
        "sys.argv = ['hermes-gateway']\n"
        "from gateway import status\n"
        "status.write_pid_file()\n"
        "if not status.acquire_gateway_runtime_lock():\n"
        "    raise SystemExit('could not acquire gateway lock')\n"
        "print('READY', flush=True)\n"
        "sys.stdin.read()\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(gateway_script)],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    ready = []
    reader = threading.Thread(target=lambda: ready.append(process.stdout.readline()))
    reader.start()
    reader.join(timeout=5)

    try:
        assert not reader.is_alive(), "gateway-like subprocess did not become ready"
        assert ready == ["READY\n"]
        assert process.poll() is None
        assert (tmp_path / "gateway.pid").exists()
        assert (tmp_path / "gateway.lock").exists()

        monkeypatch.setattr(
            gateway_mod,
            "find_gateway_pids",
            lambda *a, **k: pytest.fail("the process scan must not be needed"),
        )

        assert cron_mod._builtin_gateway_liveness() is True
        assert status_mod.is_gateway_running(cleanup_stale=False) is True
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()

    # cleanup_stale=False is intentional: the dead process leaves its PID
    # record for diagnostics instead of deleting it during this probe.
    assert status_mod.is_gateway_running(cleanup_stale=False) is False
    assert (tmp_path / "gateway.pid").exists()


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
