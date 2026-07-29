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
async def test_no_text_reply_translates_quoted_cache_path_for_docker_backend(monkeypatch, tmp_path):
    """Quoted media, like inbound media, must name the sandbox-visible cache path."""
    hermes_home = tmp_path / "hermes"
    host_path = hermes_home / "cache" / "images" / "quoted.png"
    host_path.parent.mkdir(parents=True)
    host_path.touch()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="inspect this",
        source=source,
        reply_to_message_id="42",
        reply_to_media_summary="an image (image/png, quoted.png)",
        reply_to_media_paths=[str(host_path)],
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert "local copy: /root/.hermes/cache/images/quoted.png" in result
    assert str(host_path) not in result


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
