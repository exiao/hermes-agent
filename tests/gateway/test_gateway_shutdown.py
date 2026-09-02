import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import HomeChannel, Platform
from gateway.platforms.base import MessageEvent
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
async def test_cancel_background_tasks_cancels_inflight_message_processing():
    _runner, adapter = make_restart_runner()
    release = asyncio.Event()

    async def block_forever(_event):
        await release.wait()
        return None

    adapter.set_message_handler(block_forever)
    event = MessageEvent(text="work", source=make_restart_source(), message_id="1")

    await adapter.handle_message(event)
    await asyncio.sleep(0)

    session_key = build_session_key(event.source)
    assert session_key in adapter._active_sessions
    assert adapter._background_tasks

    await adapter.cancel_background_tasks()

    assert adapter._background_tasks == set()
    assert adapter._active_sessions == {}
    assert adapter._pending_messages == {}


def test_cleanup_agent_resources_reaps_stale_aux_clients():
    runner, _adapter = make_restart_runner()
    agent = MagicMock()

    with patch("agent.auxiliary_client.cleanup_stale_async_clients") as cleanup_mock:
        runner._cleanup_agent_resources(agent)

    agent.shutdown_memory_provider.assert_called_once()
    agent.close.assert_called_once()
    cleanup_mock.assert_called_once()


def test_cron_provider_stop_cannot_override_gateway_exit_code(caplog):
    provider = MagicMock()
    provider.stop.side_effect = SystemExit(GATEWAY_SERVICE_RESTART_EXIT_CODE)

    gateway_run._stop_cron_provider(provider)

    provider.stop.assert_called_once_with()
    assert f"attempted to exit the gateway with code {GATEWAY_SERVICE_RESTART_EXIT_CODE}; ignoring" in caplog.text


@pytest.mark.asyncio
async def test_gateway_stop_interrupts_running_agents_and_cancels_adapter_tasks():
    runner, adapter = make_restart_runner()
    runner._pending_messages = {"session": "pending text"}
    runner._pending_approvals = {"session": {"command": "rm -rf /tmp/x"}}
    runner._restart_drain_timeout = 0.0

    release = asyncio.Event()

    async def block_forever(_event):
        await release.wait()
        return None

    adapter.set_message_handler(block_forever)
    event = MessageEvent(text="work", source=make_restart_source(), message_id="1")
    await adapter.handle_message(event)
    await asyncio.sleep(0)

    disconnect_mock = AsyncMock()
    adapter.disconnect = disconnect_mock

    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents = {session_key: running_agent}
    # Simulate the agent exiting once interrupted so stop()'s 5s
    # interrupt-deadline poll loop returns immediately.
    running_agent.interrupt.side_effect = lambda *a, **k: runner._running_agents.clear()

    with (
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
        patch("agent.auxiliary_client.shutdown_cached_clients") as shutdown_cached_clients,
    ):
        await runner.stop()

    running_agent.interrupt.assert_called_once_with("Gateway shutting down")
    disconnect_mock.assert_awaited_once()
    shutdown_cached_clients.assert_called_once()
    assert runner.adapters == {}
    assert runner._running_agents == {}
    assert runner._pending_messages == {}
    assert runner._pending_approvals == {}
    assert runner._shutdown_event.is_set() is True


@pytest.mark.asyncio
async def test_gateway_stop_settles_completion_batch_before_adapter_disconnect():
    runner, adapter = make_restart_runner()
    runner._completion_notification_batch_window = 3600
    event = {
        "session_id": "shutdown-batch",
        "started_at": 1.0,
        "session_key": "telegram:dm:123456:u1",
        "platform": "telegram",
        "chat_type": "dm",
        "chat_id": "123456",
        "user_id": "u1",
        "exit_code": 0,
        "output": "done",
    }
    call_order: list[str] = []
    original_cancel = runner._cancel_process_completion_batch_tasks

    async def _tracked_cancel():
        call_order.append("batch_cancel_start")
        await original_cancel()
        call_order.append("batch_cancel_done")

    async def _disconnect():
        call_order.append("disconnect")

    runner._cancel_process_completion_batch_tasks = _tracked_cancel
    adapter.disconnect = _disconnect
    pending = asyncio.create_task(
        runner._enqueue_process_completion_notification("completion", event)
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert runner._completion_notification_batch_flush_tasks

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    assert await asyncio.wait_for(pending, timeout=1.0) is False
    assert call_order == ["batch_cancel_start", "batch_cancel_done", "disconnect"]
    assert runner._completion_notification_batch_flush_tasks == set()


@pytest.mark.asyncio
async def test_planned_service_exit_issues_no_restart_of_its_own(monkeypatch):
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._restart_requested = True
    runner._restart_via_service = True
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail(
            f"planned service exit must not spawn a restart helper: {args}"
        ),
    )

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    assert runner._exit_code == GATEWAY_SERVICE_RESTART_EXIT_CODE


@pytest.mark.asyncio
async def test_unexpected_signal_starts_teardown_after_bounded_interrupt_grace():
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.0
    runner._signal_initiated_shutdown = True
    runner._signal_interrupt_grace_timeout = 0.01
    runner._running_agents = {"session": MagicMock()}

    disconnect_started = asyncio.Event()

    async def disconnect():
        disconnect_started.set()

    adapter.disconnect = disconnect

    with patch("gateway.status.remove_pid_file"), patch(
        "gateway.status.write_runtime_status"
    ):
        stop_task = asyncio.create_task(runner.stop())
        await asyncio.wait_for(disconnect_started.wait(), timeout=0.75)
        await stop_task

    assert runner._shutdown_event.is_set() is True


@pytest.mark.parametrize(
    ("signal_initiated", "restart_requested", "expected"),
    [
        (True, False, 0.25),
        (False, False, 5.0),
        (True, True, 5.0),
    ],
)
def test_post_interrupt_grace_only_shortens_unexpected_signal_shutdown(
    signal_initiated, restart_requested, expected
):
    runner, _adapter = make_restart_runner()
    runner._signal_initiated_shutdown = signal_initiated
    runner._restart_requested = restart_requested
    runner._signal_interrupt_grace_timeout = 0.25

    assert runner._post_interrupt_grace_timeout() == expected


def test_post_interrupt_grace_tolerates_duck_typed_runner():
    runner = MagicMock(spec=[])

    assert (
        gateway_run.GatewayRunner._post_interrupt_grace_timeout(runner)
        == gateway_run.DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT
    )

@pytest.mark.asyncio
async def test_in_chat_restart_skips_home_shutdown_even_with_active_session():
    runner, adapter = make_restart_runner()
    source = make_restart_source(thread_id="42")
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)
    restart_source = make_restart_source(thread_id="42")
    restart_source.message_id = "restart-command"
    runner._restart_requested = True
    runner._restart_command_source = restart_source
    runner.config.platforms[Platform.TELEGRAM].home_channel = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="home-chat",
        name="Telegram Home",
    )

    await runner._notify_active_sessions_of_shutdown()

    assert len(adapter.sent_calls) == 1
    chat_id, message, metadata = adapter.sent_calls[0]
    assert chat_id == source.chat_id
    assert "Gateway restarting" in message
    assert metadata["telegram_reply_to_message_id"] == "restart-command"


@pytest.mark.asyncio
async def test_gateway_stop_kills_tool_subprocesses_before_adapter_disconnect_on_timeout(monkeypatch):
    """On drain timeout, tool subprocesses must be killed BEFORE adapter
    disconnect so systemd's TimeoutStopSec doesn't SIGKILL the cgroup with
    bash/sleep children still attached (#8202)."""
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.01  # force timeout path

    call_order: list[str] = []

    def _fake_kill_all(task_id=None):
        call_order.append("kill_all")
        return 2

    def _fake_cleanup_envs():
        call_order.append("cleanup_environments")

    def _fake_cleanup_browsers():
        call_order.append("cleanup_browsers")

    async def _disconnect():
        call_order.append("disconnect")

    # Patch the module-level names the stop() helper imports lazily.
    import tools.process_registry as _pr
    import tools.terminal_tool as _tt
    import tools.browser_tool as _bt
    monkeypatch.setattr(_pr.process_registry, "kill_all", _fake_kill_all)
    monkeypatch.setattr(_tt, "cleanup_all_environments", _fake_cleanup_envs)
    monkeypatch.setattr(_bt, "cleanup_all_browsers", _fake_cleanup_browsers)

    adapter.disconnect = _disconnect

    runner._running_agents = {"session": MagicMock()}
    runner._running_agents["session"].interrupt.side_effect = (
        lambda *a, **k: runner._running_agents.clear()
    )

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    # First kill_all must precede the first disconnect.  (Both the eager
    # post-interrupt cleanup and the final catch-all call _kill_tool_
    # subprocesses, so we expect kill_all to appear twice total.)
    assert "kill_all" in call_order
    assert "disconnect" in call_order
    first_kill = call_order.index("kill_all")
    first_disconnect = call_order.index("disconnect")
    assert first_kill < first_disconnect, (
        f"Tool subprocesses must be killed before adapter disconnect on "
        f"drain timeout, got order: {call_order}"
    )
    # Defense-in-depth final cleanup still runs.
    assert call_order.count("kill_all") >= 2


# ---------------------------------------------------------------------------
# gateway_state persistence on shutdown (issue #42675)
#
# On Docker/s6, container_boot.py only auto-starts gateways whose last
# persisted gateway_state was "running". An unexpected external signal
# (the SIGTERM s6/Docker sends on `docker compose up --force-recreate`,
# OOM, bare kill) must NOT persist "stopped" — otherwise the gateway
# stays down after every container restart. An operator-initiated stop
# writes a planned-stop marker first, so it is NOT signal-initiated and
# DOES persist "stopped", respecting the explicit intent.
# ---------------------------------------------------------------------------


def _persisted_states(runner) -> list:
    """All gateway_state values passed to _update_runtime_status, in order."""
    states = []
    for call in runner._update_runtime_status.call_args_list:
        args, kwargs = call
        state = kwargs.get("gateway_state", args[0] if args else None)
        states.append(state)
    return states


def _stopped_state_persisted(runner) -> bool:
    """True iff _update_runtime_status was called with gateway_state='stopped'."""
    return "stopped" in _persisted_states(runner)


@pytest.mark.asyncio
async def test_signal_initiated_shutdown_persists_running_not_stopped(tmp_path, monkeypatch):
    """Unexpected SIGTERM (container restart / OOM / kill) must persist
    gateway_state=running — NOT stopped, and NOT leave the mid-shutdown
    'draining' marker — so container_boot auto-starts on next boot (#42675)."""
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()
    runner._signal_initiated_shutdown = True  # set by handler on unmarked signal

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    assert not _stopped_state_persisted(runner), (
        "signal-initiated shutdown must NOT persist gateway_state=stopped"
    )
    # The FINAL terminal write must be 'running' so container_boot's
    # _AUTOSTART_STATES check passes (it only auto-starts 'running').
    assert _persisted_states(runner)[-1] == "running", (
        f"final state must be 'running', got: {_persisted_states(runner)}"
    )


# ── #42126: zombie PID must be treated as dead in _pid_exists ────────────────
# Under systemd Restart=always, the old gateway becomes a zombie (still in the
# process table, not yet reaped) when the replacement starts. _pid_exists must
# report it dead so --replace proceeds instead of waiting on it and aborting
# with exit 1 (a silent crash loop).


def test_pid_exists_zombie_via_psutil_returns_false(monkeypatch):
    """The live path is psutil. psutil.pid_exists() returns True for a zombie,
    so _pid_exists must additionally check Process.status() == STATUS_ZOMBIE."""
    import sys
    import types

    from gateway import status

    fake_psutil = types.SimpleNamespace()
    fake_psutil.STATUS_ZOMBIE = "zombie"

    class NoSuchProcess(Exception):
        pass

    class PsutilError(Exception):
        pass

    fake_psutil.NoSuchProcess = NoSuchProcess
    fake_psutil.Error = PsutilError

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

        def status(self):
            return "zombie"

    fake_psutil.Process = _Proc
    # Without the zombie guard, this True would make the caller treat the
    # zombie as a live gateway.
    fake_psutil.pid_exists = lambda pid: True

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert status._pid_exists(4242) is False


def test_pid_exists_live_via_psutil_returns_true(monkeypatch):
    """A genuinely running (non-zombie) process is still reported alive."""
    import sys
    import types

    from gateway import status

    fake_psutil = types.SimpleNamespace()
    fake_psutil.STATUS_ZOMBIE = "zombie"
    fake_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
    fake_psutil.Error = type("Error", (Exception,), {})

    class _Proc:
        def __init__(self, pid):
            self.pid = pid

        def status(self):
            return "running"

    fake_psutil.Process = _Proc
    fake_psutil.pid_exists = lambda pid: True

    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    assert status._pid_exists(4242) is True


def test_pid_exists_zombie_via_proc_fallback_returns_false(monkeypatch):
    """When psutil is unavailable, the POSIX fallback reads /proc/<pid>/stat
    and must treat state 'Z' as dead before reaching os.kill."""
    import builtins
    import sys

    from gateway import status

    monkeypatch.setitem(sys.modules, "psutil", None)  # force ImportError
    real_import = builtins.__import__

    def _no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("psutil disabled for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)
    monkeypatch.setattr(status, "_IS_WINDOWS", False)

    fake_stat = "4242 (defunct) Z 1 0 0 0 -1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    fake_path = MagicMock()
    fake_path.read_text.return_value = fake_stat
    monkeypatch.setattr(status, "Path", lambda *_a, **_k: fake_path)

    kill = MagicMock()
    monkeypatch.setattr(status.os, "kill", kill)

    assert status._pid_exists(4242) is False
    kill.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_notice_routes_through_profile_adapter():
    """A secondary-profile session's shutdown/restart notice must go out
    through THAT profile's adapter (its own account), not the default one.

    Regression for the same-class miss flagged on #79: the per-session
    notice loop had a profile-stamped source in scope but still resolved
    the default adapter via self.adapters.get(platform)."""
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    runner, default_adapter = make_restart_runner()
    profile_adapter = RestartTestAdapter()
    runner._profile_adapters = {"equity-analyst": {Platform.TELEGRAM: profile_adapter}}

    source = make_restart_source(thread_id="42")
    source.profile = "equity-analyst"
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)

    await runner._notify_active_sessions_of_shutdown()

    # Delivered through the profile's own adapter, never the default account.
    assert len(profile_adapter.sent_calls) == 1
    assert default_adapter.sent_calls == []
    _chat_id, message, _metadata = profile_adapter.sent_calls[0]
    assert "Gateway" in message


@pytest.mark.asyncio
async def test_shutdown_notice_unstamped_source_uses_default_adapter():
    """A default-profile (unstamped) session's notice stays on the default
    adapter — the fix must not disturb the common single-profile path."""
    runner, default_adapter = make_restart_runner()
    runner._profile_adapters = {}

    source = make_restart_source(thread_id="42")
    session_key = build_session_key(source)
    runner._running_agents = {session_key: MagicMock()}
    runner._cache_session_source(session_key, source)

    await runner._notify_active_sessions_of_shutdown()

    assert len(default_adapter.sent_calls) == 1
    _chat_id, message, _metadata = default_adapter.sent_calls[0]
    assert "Gateway" in message


@pytest.mark.asyncio
async def test_shutdown_notice_dedup_keeps_distinct_profiles_separate():
    """P2 (#79 review): two sessions on the same platform/chat/thread but
    different profiles route to different accounts, so both must receive the
    notice — the dedup key includes the profile and must not collapse them."""
    from tests.gateway.restart_test_helpers import RestartTestAdapter

    runner, default_adapter = make_restart_runner()
    profile_adapter = RestartTestAdapter()
    runner._profile_adapters = {"equity-analyst": {Platform.TELEGRAM: profile_adapter}}

    # Same chat_id + thread_id, different profile → distinct delivery targets.
    default_source = make_restart_source(chat_id="777", thread_id="42")
    profile_source = make_restart_source(chat_id="777", thread_id="42")
    profile_source.profile = "equity-analyst"

    default_key = build_session_key(default_source, profile=default_source.profile)
    profile_key = build_session_key(profile_source, profile=profile_source.profile)
    assert default_key != profile_key  # distinct sessions, distinct namespaces
    runner._running_agents = {default_key: MagicMock(), profile_key: MagicMock()}
    runner._cache_session_source(default_key, default_source)
    runner._cache_session_source(profile_key, profile_source)

    await runner._notify_active_sessions_of_shutdown()

    # Each account got exactly one notice — the profile session was NOT deduped
    # away against the default session sharing the same chat/thread.
    assert len(default_adapter.sent_calls) == 1
    assert len(profile_adapter.sent_calls) == 1


@pytest.mark.asyncio
async def test_shutdown_mcp_servers_nonblocking_keeps_loop_responsive():
    """A wedged MCP shutdown must not freeze the gateway event loop (#82874)."""
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def wedged_shutdown():
        loop.call_soon_threadsafe(started.set)
        import time as _time

        _time.sleep(30)

    heartbeats = 0

    async def heartbeat():
        nonlocal heartbeats
        while True:
            heartbeats += 1
            await asyncio.sleep(0.05)

    hb = asyncio.create_task(heartbeat())
    try:
        with patch("tools.mcp_tool.shutdown_mcp_servers", wedged_shutdown):
            done = await asyncio.wait_for(
                gateway_run._shutdown_mcp_servers_nonblocking(timeout=0.5),
                timeout=5,
            )
    finally:
        hb.cancel()

    assert started.is_set()
    assert done is False  # wedged shutdown exceeded the budget
    # The loop kept running while the shutdown thread was wedged.
    assert heartbeats >= 5


@pytest.mark.asyncio
async def test_shutdown_mcp_servers_nonblocking_completes_fast_path():
    calls = []
    with patch("tools.mcp_tool.shutdown_mcp_servers", lambda: calls.append(1)):
        done = await gateway_run._shutdown_mcp_servers_nonblocking(timeout=5)
    assert done is True
    assert calls == [1]
