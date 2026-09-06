"""Tests for reply-to pointer injection in _prepare_inbound_message_text.

The `[Replying to: "..."]` prefix is a *disambiguation pointer*, not
deduplication. It must always be injected when the user explicitly replies
to a prior message — even when the quoted text already exists somewhere
in the conversation history. History can contain the same or similar text
multiple times, and without an explicit pointer the agent has to guess
which prior message the user is referencing.
"""
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_name="DM",
        chat_type="private",
        user_name="Alice",
    )


@pytest.mark.asyncio
async def test_reply_prefix_injected_when_text_absent_from_history():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="What's the best time to go?",
        source=source,
        reply_to_message_id="42",
        reply_to_text="Japan is great for culture, food, and efficiency.",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[{"role": "user", "content": "unrelated"}],
    )

    assert result is not None
    assert result.startswith(
        '[Replying to: "Japan is great for culture, food, and efficiency."]'
    )
    assert result.endswith("What's the best time to go?")


@pytest.mark.asyncio
async def test_telegram_long_reply_reaches_prompt_without_losing_later_items():
    """The native reply already has the full message; preparation must not trim it."""
    from gateway.platforms.base import MessageType
    from tests.gateway.test_telegram_reply_quote import _make_adapter, _make_message

    quoted = "\n".join(
        f"{index}. {company}: " + "Evidence from the supplied list. " * 12
        for index, company in enumerate(
            ["GoCar", "Urban Drive", "DubCar", "GRPS", "Halucar"], 1
        )
    )
    event = _make_adapter()._build_message_event(
        _make_message(text="Review all five companies.", reply_to_text=quoted),
        MessageType.TEXT,
    )
    history = [{"role": "user", "content": "Previous request"}]
    result = await _make_runner()._prepare_inbound_message_text(
        event=event, source=event.source, history=history,
    )
    assert result is not None
    assert quoted in result
    assert result.endswith("Review all five companies.")
    assert history == [{"role": "user", "content": "Previous request"}]


@pytest.mark.asyncio
async def test_quoted_reply_references_stay_literal_while_typed_ones_expand(tmp_path, monkeypatch):
    """The replied-to author's ``@file:`` is quoted text, not the replier's request: no local read.
    The same reference typed in the new message still expands (positive control)."""
    import threading

    payload = tmp_path / "notes.txt"
    payload.write_text("LOCAL-FILE-MARKER", encoding="utf-8")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    runner = _make_runner()
    runner._session_model_overrides, runner._last_resolved_model = {}, {}
    runner._agent_cache, runner._agent_cache_lock = {}, threading.Lock()
    runner._resolve_session_agent_runtime = lambda **kw: ("openai/gpt-4.1-mini", {"base_url": None, "api_key": ""})
    source = _source()

    quoted = ("x " * 300) + f"\nsee @file:{payload.name} for details"
    quoted_ref = MessageEvent(text="what does this say?", source=source, reply_to_message_id="7", reply_to_text=quoted)
    result = await runner._prepare_inbound_message_text(event=quoted_ref, source=source, history=[])
    assert quoted in result
    assert "LOCAL-FILE-MARKER" not in result

    typed_ref = MessageEvent(text=f"read @file:{payload.name}", source=source, reply_to_message_id="7", reply_to_text="short")
    result = await runner._prepare_inbound_message_text(event=typed_ref, source=source, history=[])
    assert result.startswith('[Replying to: "short"]')
    assert "LOCAL-FILE-MARKER" in result


@pytest.mark.asyncio
async def test_reply_prefix_still_injected_when_text_in_history():
    """Regression test: the pointer must survive even when the quoted text
    already appears in history. Previously a `found_in_history` guard
    silently dropped the prefix, leaving the agent to guess which prior
    message the user was referencing."""
    runner = _make_runner()
    source = _source()
    quoted = "Japan is great for culture, food, and efficiency."
    event = MessageEvent(
        text="What's the best time to go?",
        source=source,
        reply_to_message_id="42",
        reply_to_text=quoted,
    )

    history = [
        {"role": "user", "content": "I'm thinking of going to Japan or Italy."},
        {
            "role": "assistant",
            "content": (
                f"{quoted} Italy is better if you prefer a relaxed pace."
            ),
        },
        {"role": "user", "content": "How long should I stay?"},
        {"role": "assistant", "content": "For Japan, 10-14 days is ideal."},
    ]

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=history,
    )

    assert result is not None
    assert result.startswith(f'[Replying to: "{quoted}"]')
    assert result.endswith("What's the best time to go?")


@pytest.mark.asyncio
async def test_own_message_reply_prefix_marks_assistant_message():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="this one",
        source=source,
        reply_to_message_id="42",
        reply_to_text="Use the direct train.",
        reply_to_is_own_message=True,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith('[Replying to your previous message: "Use the direct train."]')
    assert result.endswith("this one")


@pytest.mark.asyncio
async def test_no_prefix_without_reply_context():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"


@pytest.mark.asyncio
async def test_no_text_reply_injects_generic_pointer():
    """reply_to_message_id without text (e.g. a reply to a media-only
    message) injects a generic no-text pointer so the agent knows the
    message is a reply, per the always-inject behavior from #13676."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="hi",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith("[Replying to a previous message (no text")
    assert result.endswith("hi")


@pytest.mark.asyncio
async def test_reply_snippet_truncated_to_2000_chars():
    runner = _make_runner()
    source = _source()
    long_text = "x" * 2500
    event = MessageEvent(
        text="follow-up",
        source=source,
        reply_to_message_id="42",
        reply_to_text=long_text,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith('[Replying to: "' + "x" * 2000 + '"]')
    assert "x" * 2001 not in result


@pytest.mark.asyncio
async def test_no_text_reply_names_the_quoted_media():
    """A quoted image is NAMED, not guessed at.

    Regression: the gateway used to emit "may have been an image or file" for
    every text-less quote, even though signal-cli reports contentType/filename
    in quote.attachments[]. That forced the agent to ask which image was meant.
    """
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="what about this part?",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
        reply_to_media_summary="an image (image/png, coach_cards.png)",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith("[Replying to an image (image/png, coach_cards.png)]")
    assert "may have been" not in result
    assert result.endswith("what about this part?")


@pytest.mark.asyncio
async def test_no_text_reply_includes_local_path_when_we_have_the_file():
    """When the bot sent the quoted image, hand over the on-disk path too."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="this one",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
        reply_to_media_summary="an image (image/png, sheet.png)",
        reply_to_media_paths=["/tmp/sheet.png"],
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "local copy: /tmp/sheet.png" in result
    assert result.startswith("[Replying to an image (image/png, sheet.png)")


@pytest.mark.asyncio
async def test_no_text_reply_translates_local_path_for_active_backend(monkeypatch):
    """Quoted media paths use the backend-visible cache mount, like inbound media."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="this one",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
        reply_to_media_summary="an image (image/png, sheet.png)",
        reply_to_media_paths=["/host/.hermes/cache/sheet.png"],
    )
    monkeypatch.setattr(
        "tools.credential_files.to_agent_visible_cache_path",
        lambda path: path.replace("/host/.hermes", "/root/.hermes"),
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "local copy: /root/.hermes/cache/sheet.png" in result
    assert "/host/.hermes/cache/sheet.png" not in result


@pytest.mark.asyncio
async def test_no_text_reply_falls_back_when_media_unknown():
    """Platforms that expose no quoted-media metadata keep the old pointer."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="hi",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith("[Replying to a previous message (no text")


@pytest.mark.asyncio
async def test_quoted_text_still_wins_over_media_summary():
    """A quote WITH text keeps quoting the text; media summary is the no-text path."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="follow-up",
        source=source,
        reply_to_message_id="42",
        reply_to_text="the original message",
        reply_to_media_summary="an image (image/png, ignored.png)",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith('[Replying to: "the original message"]')
    assert "ignored.png" not in result


@pytest.mark.asyncio
async def test_busy_steer_keeps_quoted_text():
    """A mid-turn quote-reply must still tell the agent WHICH message was quoted.

    Cold path (``_prepare_inbound_message_text``) has always injected
    ``[Replying to: "..."]``. The busy path that steers a follow-up into a
    running agent used to return only ``event.text``, so a Signal quote-reply
    like "and this" arrived as bare text with the quoted body discarded. The
    agent then had no idea which prior message the user was pointing at.
    """
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="and this",
        source=source,
        reply_to_message_id="1785980000000",
        reply_to_text="PR #679 open: https://github.com/cpe-research/cpe/pull/679",
        reply_to_is_own_message=True,
    )

    result = await runner._prepare_busy_steer_text(event)

    assert result.startswith(
        '[Replying to your previous message: "PR #679 open: https://github.com/cpe-research/cpe/pull/679"]'
    )
    assert result.endswith("and this")


@pytest.mark.asyncio
async def test_busy_steer_keeps_media_only_quote_summary():
    """Media-only quotes on the busy path must still name the attachment."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="this",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
        reply_to_media_summary="an image (image/png, coach_cards.png)",
    )

    result = await runner._prepare_busy_steer_text(event)

    assert result.startswith("[Replying to an image (image/png, coach_cards.png)]")
    assert result.endswith("this")


@pytest.mark.asyncio
async def test_busy_steer_without_quote_stays_plain_text():
    """No reply context → busy path still returns bare text, nothing invented."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(text="keep going", source=source)

    result = await runner._prepare_busy_steer_text(event)

    assert result == "keep going"


@pytest.mark.asyncio
async def test_busy_steer_failed_stt_quote_reply_stays_falsy():
    """A voice quote-reply whose STT failed must NOT become a steerable payload.

    Regression: the busy-steer gate reads the return value of
    ``_prepare_busy_steer_text``. When that value was the *formatted pointer*
    rather than the user body, an empty transcript still produced a truthy
    ``[Replying to: "..."]\\n\\n`` prefix — ``can_steer`` passed, ``steer()``
    succeeded with no message content, and the ``if not steered`` requeue was
    skipped. The user's voice note was neither steered nor queued: it was lost.
    """
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="",
        source=source,
        reply_to_message_id="9",
        reply_to_text="the earlier note",
    )

    runner._pending_event_audio_paths = lambda _event: ["/tmp/voice.ogg"]
    runner._adapter_for_source = lambda _source: None

    async def _failed_stt(event, adapter, source, text, log_context=""):
        return (text, [])

    runner._transcribe_and_echo_pending_voice = _failed_stt

    result = await runner._prepare_busy_steer_text(event)

    assert not result


@pytest.mark.asyncio
async def test_busy_steer_empty_text_quote_reply_stays_falsy():
    """A text-less quote-reply must stay falsy so the caller queues it."""
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="",
        source=source,
        reply_to_message_id="9",
        reply_to_text=None,
    )

    result = await runner._prepare_busy_steer_text(event)

    assert not result
