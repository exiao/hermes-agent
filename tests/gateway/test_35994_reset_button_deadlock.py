"""Regression test for #35994 and its follow-up: /new must not block on
old-agent cleanup.

The /new confirmation button callback runs the slash-confirm handler on the
asyncio event loop (see GatewayRunner._request_slash_confirm). That handler
calls _handle_reset_command, which tears down the OLD agent
(_cleanup_agent_resources: agent.close() tears down terminal sandboxes /
browser daemons / background processes; shutdown_memory_provider() fires the
memory-session-end plugin, which makes a network LLM call to extract
end-of-session memory).

#35994 first moved that teardown off the event loop (worker thread + bounded
wait) so a stuck close() couldn't wedge the loop. The follow-up goes further:
the reset no longer AWAITS the teardown at all. The old agent is already
invalidated and evicted, so cleanup is scheduled as a tracked background task
and /new returns as soon as the session is rotated — a slow memory-extraction
call can no longer delay the fresh session by up to the 30s cleanup cap.
"""
import asyncio
import logging
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner_with_cached_agent(close_fn):
    """Build a bare GatewayRunner with a cached agent whose close() runs
    ``close_fn`` (used to simulate slow / blocking teardown)."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key, session_id="sess-old",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    new_entry = SessionEntry(
        session_key=session_key, session_id="sess-new",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.reset_session.return_value = new_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    # Enable the cache-lock path (this is what the button callback exercises)
    runner._agent_cache_lock = threading.RLock()
    agent = MagicMock()
    agent.close = close_fn
    agent.shutdown_memory_provider = MagicMock()
    runner._agent_cache = {session_key: agent}
    return runner


async def _drain_background_tasks(runner, timeout=5.0):
    """Await any cleanup tasks the handler scheduled so no worker thread or
    coroutine dangles past the test."""
    tasks = [t for t in getattr(runner, "_background_tasks", set())]
    if tasks:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=timeout
        )


@pytest.mark.asyncio
async def test_reset_does_not_block_event_loop_during_cleanup():
    """#35994: a slow agent.close() must NOT block the event loop. A
    concurrent loop task must keep ticking WHILE close() is still blocking
    (proving cleanup was offloaded to a worker thread, not run inline on
    the loop). With the pre-fix inline call, the loop is frozen for the
    whole duration of close() and no ticks accumulate until it returns."""
    close_started = threading.Event()
    release = threading.Event()

    def slow_close():
        close_started.set()
        # Block the WORKER thread (not the loop) until released.
        release.wait(timeout=5)

    runner = _make_runner_with_cached_agent(slow_close)

    ticks = {"n": 0}
    stop = threading.Event()

    async def _heartbeat():
        while not stop.is_set():
            ticks["n"] += 1
            await asyncio.sleep(0.005)

    hb = asyncio.create_task(_heartbeat())
    reset_task = asyncio.create_task(
        runner._handle_reset_command(_make_event("/new"))
    )

    # Wait until close() has actually started blocking in its worker thread.
    for _ in range(200):
        if close_started.is_set():
            break
        await asyncio.sleep(0.005)
    assert close_started.is_set(), "close() never ran"

    # Now sample ticks while close() is STILL blocking. If the loop were
    # frozen (pre-fix inline call), this stays ~0.
    ticks_at_block = ticks["n"]
    await asyncio.sleep(0.1)
    ticks_during_block = ticks["n"] - ticks_at_block

    release.set()
    await reset_task
    await _drain_background_tasks(runner)
    stop.set()
    await hb

    assert ticks_during_block >= 5, (
        f"event loop was blocked during agent cleanup (#35994): only "
        f"{ticks_during_block} ticks while close() was running"
    )
    runner.session_store.reset_session.assert_called_once()


@pytest.mark.asyncio
async def test_reset_returns_before_slow_cleanup_finishes():
    """Follow-up: /new must not WAIT for the old-agent teardown. The reset
    result must be produced while a slow close() is still blocking in its
    worker thread — i.e. cleanup is fire-and-forget, not awaited inline."""
    close_started = threading.Event()
    release = threading.Event()

    def slow_close():
        close_started.set()
        release.wait(timeout=5)

    runner = _make_runner_with_cached_agent(slow_close)

    reset_task = asyncio.create_task(
        runner._handle_reset_command(_make_event("/new"))
    )

    # The handler should complete (session rotated, notice returned) even
    # though close() is still blocking — it was scheduled, not awaited.
    result = await asyncio.wait_for(reset_task, timeout=2)
    assert result is not None
    assert close_started.is_set(), "cleanup was never scheduled"
    assert not release.is_set(), "test bug: close() released too early"
    # A tracked background cleanup task exists and is still running.
    assert any(
        not t.done() for t in runner._background_tasks
    ), "expected a pending background cleanup task"
    runner.session_store.reset_session.assert_called_once()

    release.set()
    await _drain_background_tasks(runner)


@pytest.mark.asyncio
async def test_reset_completes_when_cleanup_raises():
    """A teardown that raises must not abort /new. Because cleanup now runs in
    a background task, an exception there can never reach the reset path — the
    session still rotates and a notice is returned."""
    def boom_close():
        raise RuntimeError("close blew up")

    runner = _make_runner_with_cached_agent(boom_close)

    result = await asyncio.wait_for(
        runner._handle_reset_command(_make_event("/new")), timeout=3
    )
    assert result is not None
    runner.session_store.reset_session.assert_called_once()
    # The old agent is evicted regardless of cleanup outcome.
    assert not runner._agent_cache
    await _drain_background_tasks(runner)


@pytest.mark.asyncio
async def test_reset_schedules_background_cleanup():
    """The handler must schedule the off-loop cleanup as a tracked background
    task (so shutdown can cancel it) rather than awaiting it inline."""
    runner = _make_runner_with_cached_agent(lambda: None)

    with patch.object(
        runner, "_cleanup_agent_resources_off_loop", AsyncMock()
    ) as _off_loop:
        result = await runner._handle_reset_command(_make_event("/new"))
        await _drain_background_tasks(runner)

    assert result is not None
    _off_loop.assert_awaited_once()
    runner.session_store.reset_session.assert_called_once()
