"""Tool progress on platforms with no message-edit API (Signal, iMessage).

Before this fix ``send_progress_messages`` unconditionally drained and
dropped every queued progress line whenever the adapter did not override
``BasePlatformAdapter.edit_message``.  That made tool progress
unreachable on Signal at ANY ``tool_progress`` setting: the user saw a
long silence and then the final answer.

Contract now: a missing ``edit_message`` only forces ``can_edit=False``,
which is the already-existing one-message-per-tool path.  The lines are
delivered regardless of grouping.  Nothing is dropped.

This is safe by default because non-editing platforms sit in
``_TIER_LOW``, whose ``tool_progress`` default is ``"off"`` — no line
reaches this queue unless the user opted in.
"""

import asyncio
import queue as queue_mod

import pytest

from gateway.turn_context import TurnContext


class _NoEditAdapter:
    """Duck-typed stand-in for Signal: send works, edit_message doesn't."""

    name = "signal"
    MAX_MESSAGE_LENGTH = 2000

    def __init__(self):
        self.sent = []
        self.edits = []

    async def send(self, chat_id=None, content=None, **kwargs):
        self.sent.append(content)
        return type("R", (), {"success": True, "message_id": "m1", "error": None})()


class _EditAdapter(_NoEditAdapter):
    """Telegram-like: editing supported, so lines accumulate in one bubble."""

    name = "telegram"

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.edits.append(content)
        return type("R", (), {"success": True, "message_id": message_id, "error": None})()


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

    def _still_current():
        # True while draining, False once the queue is empty so the
        # sender loop terminates instead of blocking forever.
        return not q.empty()

    return TurnContext(
        source=type("S", (), {"chat_id": "c1", "platform": None})(),
        progress_queue=q,
        progress_grouping=grouping,
        progress_mode="all",
        tool_progress_enabled=True,
        _run_still_current=_still_current,
    ), q


@pytest.mark.parametrize("grouping", ["accumulate", "separate"])
def test_non_editing_platform_still_delivers_progress(grouping):
    """The regression: these lines used to be silently dropped."""
    adapter = _NoEditAdapter()
    ctx, _q = _ctx_with(grouping, ["🔧 terminal", "🔍 web_search"])
    runner = _runner_for(adapter, ctx)

    asyncio.run(asyncio.wait_for(runner.send_progress_messages(), 10))

    assert adapter.sent, (
        f"grouping={grouping}: progress must reach a platform that cannot "
        "edit messages, one message per line"
    )
    assert any("terminal" in s for s in adapter.sent)


def test_editing_platform_still_accumulates_into_one_bubble():
    """Guard the unchanged Telegram/Discord path: edit, don't spam."""
    adapter = _EditAdapter()
    ctx, _q = _ctx_with("accumulate", ["🔧 terminal", "🔍 web_search"])
    runner = _runner_for(adapter, ctx)

    asyncio.run(asyncio.wait_for(runner.send_progress_messages(), 10))

    # First line is a send; subsequent lines edit that same message.
    assert len(adapter.sent) == 1, "editable platform must not send a second bubble"
