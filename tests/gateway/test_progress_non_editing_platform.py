"""Tool progress on platforms with no message-edit API (Signal, iMessage).

Before this fix ``send_progress_messages`` unconditionally drained and
dropped every queued progress line whenever the adapter did not override
``BasePlatformAdapter.edit_message``.  That made tool progress
unreachable on Signal at ANY ``tool_progress`` setting: the user saw a
long silence and then the final answer.

Contract now: the drop only happens in the default "accumulate" grouping
(where one editable bubble is the point).  With
``tool_progress_grouping: separate`` the user has explicitly asked for
one message per tool, which is exactly the shape a non-editing platform
can render, so the lines must be SENT.
"""

import asyncio
import queue as queue_mod
import time

import pytest

from gateway.turn_context import TurnContext


class _NoEditAdapter:
    """Duck-typed stand-in for Signal: send works, edit_message doesn't."""

    name = "signal"
    MAX_MESSAGE_LENGTH = 2000

    def __init__(self):
        self.sent = []

    async def send(self, chat_id=None, content=None, **kwargs):
        self.sent.append(content)
        return type("R", (), {"success": True, "message_id": "m1", "error": None})()

    async def send_typing(self, chat_id=None, **kwargs):
        pass


class _EditableAdapter(_NoEditAdapter):
    """Stand-in for Telegram: send and edit are both available."""

    def __init__(self):
        super().__init__()
        self.send_times = []
        self.edited = []

    async def send(self, chat_id=None, content=None, **kwargs):
        self.send_times.append(time.monotonic())
        return await super().send(chat_id, content, **kwargs)

    async def edit_message(self, **kwargs):
        self.edited.append(kwargs["content"])
        return type("R", (), {"success": True, "message_id": "m1", "error": None})()


class _BlockingSendAdapter(_NoEditAdapter):
    """Block the first send so cancellation can interrupt it in flight."""

    def __init__(self):
        super().__init__()
        self.send_started = asyncio.Event()
        self.block_next = True

    async def send(self, chat_id=None, content=None, **kwargs):
        if self.block_next:
            self.block_next = False
            self.send_started.set()
            await asyncio.Event().wait()
        return await super().send(chat_id, content, **kwargs)


def _runner_for(adapter, ctx):
    from gateway.run import TurnRunner

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return adapter

    return TurnRunner(_StubGatewayRunner(), ctx)


def _ctx_with(grouping, lines):
    q = queue_mod.Queue()
    for line in lines:
        q.put(line)
    calls = {"n": 0}

    def _still_current():
        # Keep the turn current while the sender processes both queued lines;
        # the sender checks this callback after dequeuing a line as well as
        # before its next iteration.
        calls["n"] += 1
        return calls["n"] <= 6

    return TurnContext(
        source=type("S", (), {"chat_id": "c1", "platform": None})(),
        progress_queue=q,
        progress_grouping=grouping,
        progress_mode="all",
        tool_progress_enabled=True,
        _run_still_current=_still_current,
    ), q


@pytest.mark.parametrize("grouping", ["accumulate", "grouped"])
def test_editable_grouping_drops_lines_on_non_editing_platform(grouping):
    adapter = _NoEditAdapter()
    ctx, q = _ctx_with(grouping, ["🔧 terminal", "🔍 web_search"])
    runner = _runner_for(adapter, ctx)

    asyncio.run(asyncio.wait_for(runner.send_progress_messages(), 5))

    assert q.empty(), "queue must be drained"
    assert adapter.sent == [], "no bubbles when the bubble can't be edited"


def test_separate_grouping_sends_lines_on_non_editing_platform():
    adapter = _NoEditAdapter()
    ctx, q = _ctx_with("separate", ["🔧 terminal", "🔍 web_search"])
    runner = _runner_for(adapter, ctx)

    asyncio.run(asyncio.wait_for(runner.send_progress_messages(), 10))

    assert len(adapter.sent) == 2, "both progress messages must be sent"
    assert any("terminal" in s for s in adapter.sent)
    assert any("web_search" in s for s in adapter.sent)


def test_separate_grouping_paces_editable_platform():
    adapter = _EditableAdapter()
    ctx, q = _ctx_with("separate", ["🔧 terminal", "🔍 web_search"])
    runner = _runner_for(adapter, ctx)

    asyncio.run(asyncio.wait_for(runner.send_progress_messages(), 10))

    assert q.empty(), "queue must be drained"
    assert adapter.sent == ["🔧 terminal", "🔍 web_search"]
    assert len(adapter.send_times) == 2
    assert adapter.send_times[1] - adapter.send_times[0] >= 1.4


def test_separate_grouping_flushes_queued_editable_progress_on_cancel():
    adapter = _EditableAdapter()
    ctx, q = _ctx_with("separate", ["🔧 terminal", "🔍 web_search"])
    runner = _runner_for(adapter, ctx)

    async def _cancel_after_first_send():
        task = asyncio.create_task(runner.send_progress_messages())
        for _ in range(100):
            await asyncio.sleep(0)
            if adapter.sent:
                break
        else:
            raise AssertionError("sender did not send the first progress line")
        assert not task.done()
        task.cancel()
        await asyncio.wait_for(task, 5)

    asyncio.run(_cancel_after_first_send())

    assert q.empty(), "queue must be drained"
    assert adapter.sent == ["🔧 terminal", "🔍 web_search"]


def test_cancelled_inflight_separate_send_retries_dequeued_line():
    adapter = _BlockingSendAdapter()
    ctx, q = _ctx_with("separate", ["🔧 terminal", "🔍 web_search"])
    runner = _runner_for(adapter, ctx)

    async def _cancel_inflight_send():
        task = asyncio.create_task(runner.send_progress_messages())
        await asyncio.wait_for(adapter.send_started.wait(), 5)
        assert adapter.sent == []
        task.cancel()
        await asyncio.wait_for(task, 5)

    asyncio.run(_cancel_inflight_send())

    assert q.empty(), "queue must be drained"
    assert adapter.sent == ["🔧 terminal", "🔍 web_search"]


def test_cancelled_stale_turn_drops_queued_separate_progress():
    adapter = _NoEditAdapter()
    ctx, q = _ctx_with("separate", ["🔧 terminal", "🔍 web_search"])
    current = [True]
    ctx._run_still_current = lambda: current[0]
    runner = _runner_for(adapter, ctx)

    async def _cancel_after_run_replaced():
        task = asyncio.create_task(runner.send_progress_messages())
        for _ in range(100):
            await asyncio.sleep(0)
            if adapter.sent:
                break
        else:
            raise AssertionError("sender did not send the first progress line")
        current[0] = False
        task.cancel()
        await asyncio.wait_for(task, 5)

    asyncio.run(_cancel_after_run_replaced())

    assert q.empty(), "stale queued progress must be discarded"
    assert adapter.sent == ["🔧 terminal"]


def test_cancelled_stale_turn_drops_line_if_replaced_during_drain_pacing():
    adapter = _EditableAdapter()
    ctx, q = _ctx_with("separate", ["🔧 terminal", "🔍 web_search"])
    current = [True]
    drain_check_started = asyncio.Event()
    checks = {"n": 0}

    def _still_current():
        checks["n"] += 1
        if checks["n"] == 3:
            drain_check_started.set()
        return current[0]

    ctx._run_still_current = _still_current
    runner = _runner_for(adapter, ctx)

    async def _cancel_while_drain_is_pacing():
        task = asyncio.create_task(runner.send_progress_messages())
        for _ in range(100):
            await asyncio.sleep(0)
            if adapter.sent:
                break
        else:
            raise AssertionError("sender did not send the first progress line")
        task.cancel()
        await asyncio.wait_for(drain_check_started.wait(), 5)
        current[0] = False
        await asyncio.wait_for(task, 5)

    asyncio.run(_cancel_while_drain_is_pacing())

    assert q.empty(), "stale queued progress must be discarded"
    assert adapter.sent == ["🔧 terminal"]


def test_cancelled_interrupted_turn_drops_queued_progress():
    adapter = _EditableAdapter()
    ctx, q = _ctx_with("separate", ["🔧 terminal", "🔍 web_search"])
    agent = type("A", (), {"is_interrupted": False})()
    ctx.agent_holder[0] = agent
    runner = _runner_for(adapter, ctx)

    async def _cancel_after_first_send():
        task = asyncio.create_task(runner.send_progress_messages())
        for _ in range(100):
            await asyncio.sleep(0)
            if adapter.sent:
                break
        else:
            raise AssertionError("sender did not send the first progress line")
        agent.is_interrupted = True
        task.cancel()
        await asyncio.wait_for(task, 5)

    asyncio.run(_cancel_after_first_send())

    assert q.empty(), "queue must be drained"
    assert adapter.sent == ["🔧 terminal"]


def test_cancelled_interrupted_turn_drops_line_if_interrupted_during_drain_pacing():
    adapter = _EditableAdapter()
    ctx, q = _ctx_with("separate", ["🔧 terminal", "🔍 web_search"])
    agent = type("A", (), {"is_interrupted": False})()
    ctx.agent_holder[0] = agent
    drain_check_started = asyncio.Event()
    checks = {"n": 0}

    def _still_current():
        checks["n"] += 1
        if checks["n"] == 3:
            drain_check_started.set()
        return True

    ctx._run_still_current = _still_current
    runner = _runner_for(adapter, ctx)

    async def _cancel_while_drain_is_pacing():
        task = asyncio.create_task(runner.send_progress_messages())
        for _ in range(100):
            await asyncio.sleep(0)
            if adapter.sent:
                break
        else:
            raise AssertionError("sender did not send the first progress line")
        task.cancel()
        await asyncio.wait_for(drain_check_started.wait(), 5)
        agent.is_interrupted = True
        await asyncio.wait_for(task, 5)

    asyncio.run(_cancel_while_drain_is_pacing())

    assert q.empty(), "interrupted queued progress must be discarded"
    assert adapter.sent == ["🔧 terminal"]
