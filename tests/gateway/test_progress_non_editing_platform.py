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

    async def send_typing(self, chat_id, **kwargs):
        # Required: the sender restores the typing indicator after every
        # progress write. Without it the loop raises on the first pass and
        # the edit path is never reached, which silently defeats these tests.
        return None


class _EditAdapter(_NoEditAdapter):
    """Telegram-like: editing supported, so lines accumulate in one bubble."""

    name = "telegram"

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.edits.append(content)
        return type("R", (), {"success": True, "message_id": message_id, "error": None})()


class _DeclaresNoEditAdapter(_EditAdapter):
    """DingTalk-like: implements edit_message but declares it unusable.

    DingTalk's ``SUPPORTS_MESSAGE_EDITING`` is a property returning False
    when AI Cards aren't configured. Checking only for the presence of the
    method would wrongly conclude "can edit" and try to edit a message the
    platform cannot edit.
    """

    name = "dingtalk"
    SUPPORTS_MESSAGE_EDITING = False


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

    # Stop condition. The sender loops until the run is no longer current;
    # an empty queue is NOT the end (it sleeps and re-polls). Two rules:
    #   - while the queue still has lines, always stay current
    #   - once it drains, allow a fixed grace of further passes so the
    #     final throttled edit/send actually lands, then stop
    # Gating purely on `q.empty()` would kill the loop after the FIRST
    # line, so no test would reach the edit path (they'd pass vacuously).
    # A pure call-count budget is equally fragile: it can expire mid-drain.
    grace = {"n": 25}

    def _still_current():
        if not q.empty():
            return True
        grace["n"] -= 1
        return grace["n"] > 0

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


def test_adapter_declaring_no_editing_is_not_edited():
    """SUPPORTS_MESSAGE_EDITING=False wins over a present edit_message."""
    adapter = _DeclaresNoEditAdapter()
    # Three lines, not two: the first line is always a plain send, so a
    # two-line case can finish before any edit would have been attempted
    # and the assertion would pass vacuously.
    ctx, _q = _ctx_with("accumulate", ["🔧 terminal", "🔍 web_search", "📄 read_file"])
    runner = _runner_for(adapter, ctx)

    asyncio.run(asyncio.wait_for(runner.send_progress_messages(), 10))

    assert adapter.edits == [], (
        "an adapter declaring SUPPORTS_MESSAGE_EDITING=False must never be "
        "edited, even though it defines edit_message"
    )
    assert adapter.sent, "progress should still be delivered via send()"


def test_editing_platform_still_accumulates_into_one_bubble():
    """Guard the unchanged Telegram/Discord path: edit, don't spam."""
    adapter = _EditAdapter()
    ctx, _q = _ctx_with("accumulate", ["🔧 terminal", "🔍 web_search", "📄 read_file"])
    runner = _runner_for(adapter, ctx)

    asyncio.run(asyncio.wait_for(runner.send_progress_messages(), 10))

    # First line is a send; subsequent lines edit that same message.
    assert len(adapter.sent) == 1, "editable platform must not send a second bubble"
    assert adapter.edits, "editable platform must actually edit the bubble"
