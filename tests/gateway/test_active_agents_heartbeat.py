"""Tests for the active_agents heartbeat (false-SIGKILL fix).

Symptom: the signal watchdog force-restarted a healthy gateway 4x on
2026-07-18. gateway_state.json:active_agents is the watchdog's manual-mode
liveness proxy, but it is only persisted at CHAT turn boundaries. Cron jobs
and api-server runs feed _active_work_count() with NO persist hook of their
own, so after cron activity the on-disk count froze at a stale non-zero value
(and its mtime stopped advancing) between chat turns -- exactly the watchdog's
"active_agents>0 + stale mtime" HUNG signature.

Fix: _heartbeat_active_agents() re-persists the live count from the gateway
housekeeping loop every tick. These tests exercise the real function against a
real GatewayRunner double writing to a temp HERMES_HOME gateway_state.json.
"""

import json
import os
import time
from pathlib import Path

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.fixture(autouse=True)
def _reset_cron_running_set():
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    yield
    sched._running_job_ids.clear()


@pytest.fixture
def temp_hermes_home(tmp_path, monkeypatch):
    """Point gateway.status at a temp HERMES_HOME so writes are isolated."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # gateway.status caches nothing across calls; it re-resolves the path each
    # write from the env, so setting HERMES_HOME is sufficient.
    return tmp_path


def _state_path(home: Path) -> Path:
    return home / "gateway_state.json"


def _read_active_agents(home: Path):
    data = json.loads(_state_path(home).read_text())
    return data["active_agents"]


class TestHeartbeatPersistsLiveCount:
    def test_heartbeat_writes_current_count_when_idle(self, temp_hermes_home):
        """With no work in flight the heartbeat writes active_agents=0."""
        import gateway.run as run

        runner, _adapter = make_restart_runner()
        with run._install_runner_ref(runner):
            run._heartbeat_active_agents()

        assert _read_active_agents(temp_hermes_home) == 0

    def test_heartbeat_corrects_stale_count_after_cron_completes(self, temp_hermes_home):
        """THE REGRESSION. Simulate the exact leak: a stale on-disk count of 1
        left behind by cron activity, with no chat turn to refresh it. The
        heartbeat must pull it back down to the true live count (0)."""
        import gateway.run as run
        import cron.scheduler as sched
        from gateway.status import write_runtime_status

        runner, _adapter = make_restart_runner()

        # 1. A cron job is running -> live count is 1.
        sched._running_job_ids.add("job-1")
        with run._install_runner_ref(runner):
            run._heartbeat_active_agents()
        assert _read_active_agents(temp_hermes_home) == 1, "count should reflect the in-flight cron job"

        # 2. The cron job finishes. Its scheduler finally discards the id, but
        #    (pre-fix) NOTHING re-persists the count -- the file stays at 1.
        sched._running_job_ids.discard("job-1")
        assert _read_active_agents(temp_hermes_home) == 1, "file is now stale (this is the bug)"

        # 3. The housekeeping heartbeat fires on its next tick and corrects it.
        with run._install_runner_ref(runner):
            run._heartbeat_active_agents()
        assert _read_active_agents(temp_hermes_home) == 0, (
            "heartbeat must re-persist the lowered count after cron work "
            "completes, so the watchdog no longer sees a phantom active agent"
        )

    def test_heartbeat_advances_mtime(self, temp_hermes_home):
        """mtime must move on every heartbeat so it is a true liveness signal,
        not a chat-turn-boundary artifact. This is the second half of the
        watchdog's kill condition."""
        import gateway.run as run

        runner, _adapter = make_restart_runner()

        with run._install_runner_ref(runner):
            run._heartbeat_active_agents()
        first_mtime = _state_path(temp_hermes_home).stat().st_mtime

        time.sleep(0.02)
        with run._install_runner_ref(runner):
            run._heartbeat_active_agents()
        second_mtime = _state_path(temp_hermes_home).stat().st_mtime

        assert second_mtime > first_mtime, "each heartbeat must advance the state-file mtime"

    def test_heartbeat_counts_chat_and_cron_together(self, temp_hermes_home):
        """The persisted count is the aggregate of all work sources."""
        import gateway.run as run
        import cron.scheduler as sched
        from unittest.mock import MagicMock

        runner, _adapter = make_restart_runner()
        runner._running_agents = {"session-1": MagicMock()}  # 1 chat turn
        sched._running_job_ids.add("job-1")                  # + 1 cron job

        with run._install_runner_ref(runner):
            run._heartbeat_active_agents()

        assert _read_active_agents(temp_hermes_home) == 2


class TestHeartbeatFailOpen:
    def test_no_runner_ref_is_silent_noop(self, temp_hermes_home):
        """Before the runner is constructed (or after GC) the weakref derefs to
        None -- the heartbeat must no-op, not crash the housekeeping thread."""
        import gateway.run as run

        # Ensure the ref is empty.
        run._gateway_runner_ref = lambda: None
        run._heartbeat_active_agents()  # must not raise

        assert not _state_path(temp_hermes_home).exists()

    def test_persist_exception_is_swallowed(self, temp_hermes_home, monkeypatch):
        """A failed status write must never disrupt housekeeping."""
        import gateway.run as run

        runner, _adapter = make_restart_runner()

        def _boom():
            raise RuntimeError("disk full")

        monkeypatch.setattr(runner, "_persist_active_agents", _boom)
        with run._install_runner_ref(runner):
            run._heartbeat_active_agents()  # must not raise
