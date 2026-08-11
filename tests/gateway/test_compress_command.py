"""Tests for gateway /compress user-facing messaging."""

import asyncio
import threading
from datetime import datetime
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


def _make_event(text: str = "/compress") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_history() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]


def _make_runner(history: list[dict[str, str]]):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = history
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner.session_store._save = MagicMock()
    runner._session_db = None
    return runner


@pytest.mark.asyncio
async def test_compress_command_works_when_auto_compaction_disabled():
    """compression.enabled: false disables *automatic* compaction only.

    The gateway /compress handler has never gated on the flag — pin that
    contract (every manual-compress surface must allow manual compression
    regardless of the auto toggle, #64438) and the force=True cooldown
    bypass that manual compression relies on."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.compression_enabled = False
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    # Explicit non-lock-skip: MagicMock getattr would return a truthy mock.
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        return 100 if messages == history else 60

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "disabled" not in result.lower()
    assert "Compressed:" in result
    agent_instance._compress_context.assert_called_once()
    assert agent_instance._compress_context.call_args.kwargs.get("force") is True


@pytest.mark.asyncio
async def test_compress_command_surfaces_aux_model_failure_even_when_recovered():
    """When the user's configured ``auxiliary.compression.model`` errors out
    but compression recovers by retrying on the main model, /compress must
    STILL inform the user.  Silent recovery hides broken config the user
    needs to fix."""
    history = _make_history()
    # Compressed transcript — normal successful compression, no placeholder.
    compressed = [
        history[0],
        {"role": "assistant", "content": "summary via main model"},
        history[-1],
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # Fallback placeholder was NOT used — recovery succeeded.
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    # But the configured aux model DID fail before the retry succeeded.
    agent_instance.context_compressor._last_aux_model_failure_model = (
        "gemini-3-flash-preview"
    )
    agent_instance.context_compressor._last_aux_model_failure_error = (
        "404 model not found: gemini-3-flash-preview"
    )
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Compression succeeded
    assert "Compressed:" in result
    # No ⚠️ warning (that's reserved for dropped-turns case)
    assert "⚠️" not in result
    # But there IS an info note about the broken aux model
    assert "ℹ️" in result
    assert "gemini-3-flash-preview" in result
    assert "404" in result
    assert "auxiliary.compression.model" in result
    # The user's context is explicitly called out as intact
    assert "intact" in result
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_passes_session_db_and_persists_rotated_session():
    """session_db must be wired into the /compress temp agent so that
    _compress_context can actually rotate the session and persist the
    compressed transcript — without it compression is a silent no-op."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = object()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    agent_instance.session_id = "sess-1"

    def _compress(messages, *_args, **_kwargs):
        agent_instance.session_id = "sess-2"
        return compressed, ""

    agent_instance._compress_context.side_effect = _compress
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance) as mock_agent_cls,
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    mock_agent_cls.assert_called_once()
    assert mock_agent_cls.call_args.kwargs["session_db"] is runner._session_db
    # The continuation row _compress_context creates must record the real
    # gateway platform, not fall back to source='cli' (which hides post-
    # compression messages from source-filtered SessionDB searches).
    assert mock_agent_cls.call_args.kwargs["platform"] == "telegram"
    assert mock_agent_cls.call_args.kwargs["chat_id"] == "c1"
    runner.session_store._save.assert_called_once()
    runner.session_store.rewrite_transcript.assert_called_once_with(
        "sess-2", compressed
    )
    runner.session_store.update_session.assert_called_once_with(
        build_session_key(_make_source()), last_prompt_tokens=0
    )
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()
    # The temp agent rotates session_id to the continuation session, which the
    # gateway entry now points at. close() must NOT end that live session, so
    # the handler must set _end_session_on_close = False before cleanup. Without
    # it, passing session_db makes AIAgent.close() mark the rotated session
    # ended with agent_close, breaking persistence/search for compressed chats.
    assert agent_instance._end_session_on_close is False


@pytest.mark.asyncio
async def test_compress_command_does_not_repoint_session_when_transcript_write_fails():
    """If the canonical transcript write fails after compression produces a new
    continuation session_id, /compress must NOT repoint the live session onto
    that empty session_id, and must report the failure instead of a success
    banner. Otherwise a transient DB/IO error during compression would silently
    drop the user's active conversation while still claiming success."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = object()
    session_entry = runner.session_store.get_or_create_session.return_value
    # Simulate the canonical DB write failing (lock contention, ENOSPC, ...).
    runner.session_store.rewrite_transcript = MagicMock(return_value=False)
    # Telegram topic re-binding must never run on the failure path.
    runner._sync_telegram_topic_binding = MagicMock()

    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance._last_compaction_in_place = False
    agent_instance.session_id = "sess-1"

    def _compress(messages, *_args, **_kwargs):
        # Compression rotated the session: the agent now holds a NEW session_id.
        agent_instance.session_id = "sess-2"
        return compressed, ""

    agent_instance._compress_context.side_effect = _compress
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        return 100

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # The user sees a failure banner, not a success banner.
    assert "failed" in result.lower()
    assert "Compressed:" not in result
    # The live session was NOT repointed onto the empty new session_id, so the
    # original conversation stays reachable.
    assert session_entry.session_id == "sess-1"
    runner.session_store._save.assert_not_called()
    runner._sync_telegram_topic_binding.assert_not_called()
    # Resources are still cleaned up even though the command errored.
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_preserves_transcript_when_rotation_aborts():
    """If _compress_context can neither rotate the session nor compact in
    place (e.g. the DB split raised and session_id rolled back unchanged),
    the handler must NOT call rewrite_transcript — doing so would delete the
    original searchable history and replace it with only the summary
    (#44794)."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = object()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    # Genuine rotation save failure (NOT a summarizer abort): session_id
    # stays at the original, not in-place, and the compressor did not abort.
    agent_instance.session_id = "sess-1"
    agent_instance._last_compaction_in_place = False
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        return 100 if messages == history else 60

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # The original transcript must be preserved — no destructive rewrite.
    runner.session_store.rewrite_transcript.assert_not_called()
    # Nothing was persisted, so the user must NOT be told compression succeeded;
    # the handler returns the no-op message instead of the "Compressed" summary.
    assert "could not be saved" in result
    assert "Compressed:" not in result


@pytest.mark.asyncio
async def test_compress_command_surfaces_abort_error_not_rotation_failure():
    """When the aux summarizer ABORTS, _compress_context returns the messages
    unchanged with the same session id and sets _last_compress_aborted — but
    NOT _last_compaction_in_place. That lands in the same unchanged-id branch
    as a genuine rotation save failure, so the handler must distinguish them:
    on abort it must fall through to the summary path and surface the real
    "Compression aborted" guidance + error, NOT the misleading
    "could not be saved (session rotation failed)" no-op."""
    history = _make_history()
    runner = _make_runner(history)
    runner._session_db = object()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = False
    # Summarizer abort: unchanged id, not in-place, but abort flag set and the
    # compressor returns the messages unchanged.
    agent_instance.session_id = "sess-1"
    agent_instance._last_compaction_in_place = False
    agent_instance.context_compressor._last_compress_aborted = True
    agent_instance.context_compressor._last_summary_error = "aux model 503"
    agent_instance._compress_context.return_value = (history, "")

    def _estimate(messages, **_kwargs):
        return 100 if messages == history else 60

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    # Original transcript preserved (no destructive rewrite) ...
    runner.session_store.rewrite_transcript.assert_not_called()
    # ... but the user sees the real abort guidance + error, NOT the
    # rotation-failure no-op message.
    assert "Compression aborted" in result
    assert "aux model 503" in result
    assert "could not be saved" not in result


@pytest.mark.asyncio
async def test_compress_command_skips_rewrite_after_in_place_compaction():
    """In-place compaction (compression.in_place, the default) persists via
    the NON-destructive SessionDB.archive_and_compact under the SAME id —
    old turns kept active=0 and FTS-searchable. The handler must NOT then call
    rewrite_transcript, which routes to the DESTRUCTIVE replace_messages and
    would DELETE every row for the id, wiping the archived history."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = object()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = True
    # In-place: same id, _last_compaction_in_place True.
    agent_instance.session_id = "sess-1"
    agent_instance._last_compaction_in_place = True
    agent_instance._compress_context.return_value = (compressed, "")

    def _estimate(messages, **_kwargs):
        return 100 if messages == history else 60

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        await runner._handle_compress_command(_make_event())

    # archive_and_compact already persisted durably; no destructive rewrite.
    runner.session_store.rewrite_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_compress_here_in_place_persists_rejoined_tail():
    """`/compress here` (partial) under in-place compaction: _compress_context
    archives + inserts only the compressed HEAD, then the handler rejoins the
    verbatim tail. The handler must re-persist the rejoined head+tail via the
    non-destructive archive_and_compact so the kept recent exchanges land in
    the live (active=1) set — otherwise the next turn reloads head-only and
    silently drops them."""
    history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
        {"role": "user", "content": "five"},
        {"role": "assistant", "content": "six"},
    ]
    head_compressed = [{"role": "assistant", "content": "summary of head"}]
    runner = _make_runner(history)
    fake_db = MagicMock()
    # The extracted mixin awaits get_session to restore the live system prompt.
    fake_db.get_session = AsyncMock(return_value=None)
    runner._session_db = fake_db
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance._session_db = fake_db
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.compression_in_place = True
    agent_instance.session_id = "sess-1"
    agent_instance._last_compaction_in_place = True
    agent_instance._compress_context.return_value = (head_compressed, "")

    def _estimate(messages, **_kwargs):
        return 100

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        await runner._handle_compress_command(_make_event("/compress here 2"))

    # The rejoined head+tail must be re-persisted non-destructively, and the
    # destructive rewrite path must NOT be used.
    fake_db.archive_and_compact.assert_called_once()
    persisted = fake_db.archive_and_compact.call_args.args[1]
    # Tail (the last kept exchanges) must be present in the persisted set.
    assert {"role": "user", "content": "five"} in persisted
    assert {"role": "assistant", "content": "six"} in persisted
    runner.session_store.rewrite_transcript.assert_not_called()


@pytest.mark.asyncio
async def test_compress_command_preserves_platform_and_gateway_session_key():
    """The temporary compression agent must carry the originating source's
    platform and stable gateway session key, matching a normal gateway turn.
    Without them ``_session_source_for_agent`` falls back to a default "cli"
    host source, so an external context engine misattributes the retained
    transcript tail and later duplicates it on resume (#50422)."""
    history = _make_history()
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")
    agent_instance._compression_skipped_due_to_lock = False

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance) as mock_agent,
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())

    assert mock_agent.call_count == 1
    _, kwargs = mock_agent.call_args
    # Platform preserved as the live turn's config key (TELEGRAM -> "telegram"),
    # not the unbound "cli"/"local" fallback.
    assert kwargs.get("platform") == "telegram"
    # Stable gateway session key preserved, identical to a normal gateway turn.
    assert kwargs.get("gateway_session_key") == runner._session_key_for_source(_make_source())
    assert kwargs["gateway_session_key"]


@pytest.mark.asyncio
async def test_compress_command_passes_tool_messages_to_compressor():
    """Tool results must reach _compress_context (#3854).

    Filtering the transcript to user/assistant-only starved the
    compressor's tool-result pruning — tool messages are usually the bulk
    of the context.
    """
    history = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "BIG RESULT " * 50, "tool_call_id": "t1"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "np"},
    ]
    runner = _make_runner(history)
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())

    args, _kwargs = agent_instance._compress_context.call_args
    passed = args[0]
    roles = [m.get("role") for m in passed]
    assert "tool" in roles, f"tool messages filtered out: {roles}"
    # Assistant tool_calls stubs (content=None) must survive too, or the
    # tool message would dangle without its call.
    assert any(m.get("tool_calls") for m in passed), "assistant tool_calls stub dropped"


@pytest.mark.asyncio
async def test_compress_command_in_place_skips_destructive_rewrite():
    """In-place compaction (compression.in_place / #38763) persists via
    archive_and_compact() inside _compress_context — the previous active rows
    are soft-archived and the compacted set inserted. Calling
    rewrite_transcript() afterwards would invoke
    replace_messages(active_only=False), DELETEing the just-archived rows
    (silent data loss, #61145). The handler must skip the rewrite and still
    report success."""
    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compacted summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner._session_db = object()
    session_entry = runner.session_store.get_or_create_session.return_value
    runner.session_store.rewrite_transcript = MagicMock()

    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    # In-place compaction: session_id is UNCHANGED but marked as a success.
    agent_instance._last_compaction_in_place = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False

    def _estimate(messages, **_kwargs):
        if messages == history:
            return 100
        if messages == compressed:
            return 60
        raise AssertionError(f"unexpected transcript: {messages!r}")

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", side_effect=_estimate),
    ):
        result = await runner._handle_compress_command(_make_event())

    assert "Compressed:" in result
    # The destructive rewrite must NOT run — archive_and_compact() already
    # persisted, and rewrite_transcript would wipe the archived rows.
    runner.session_store.rewrite_transcript.assert_not_called()
    assert session_entry.session_id == "sess-1"
    agent_instance.shutdown_memory_provider.assert_called_once()
    agent_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_multiplexed_runs_under_profile_secret_scope(tmp_path):
    """Manual /compress must install the source profile's secret scope.

    Multiplexed gateways resolve credentials fail-closed (Workstream A):
    ``get_secret`` raises ``UnscopedSecretError`` on any read outside a
    ``set_secret_scope`` block. The agent turn is scoped by ``_run_agent``'s
    wrapper, but slash-command dispatch is not — manual /compress reached the
    compressor's provider resolution unscoped and died with
    ``get_secret('OPENROUTER_BASE_URL') called with no profile secret scope
    active``. The credential read happens inside the executor hop, so this
    also pins that the handler uses the contextvar-preserving executor
    (``_run_in_executor_with_context``), not a bare ``run_in_executor``.
    """
    from agent import secret_scope as ss

    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        multiplex_profiles=True,
    )
    profile_home = tmp_path / "profiles" / "milo"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text(
        "OPENROUTER_BASE_URL=https://scoped.example/v1\n"
    )
    runner._resolve_profile_home_for_source = MagicMock(return_value=profile_home)

    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.context_compressor._last_aux_model_failure_error = None
    agent_instance.session_id = "sess-1"
    agent_instance._compression_skipped_due_to_lock = False

    seen: dict[str, str | None] = {}

    def _compress(*_args, **_kwargs):
        # Runs in the executor thread — exactly where the aux client
        # resolves provider credentials. Fail-closed get_secret raises
        # here unless the profile scope survived the thread hop.
        seen["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return (compressed, "")

    agent_instance._compress_context.side_effect = _compress

    ss.set_multiplex_active(True)
    try:
        with (
            patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
            patch("gateway.run._resolve_gateway_model", return_value="test-model"),
            patch("run_agent.AIAgent", return_value=agent_instance),
            patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
        ):
            result = await runner._handle_compress_command(_make_event())
    finally:
        ss.set_multiplex_active(False)
        runner._shutdown_executor()

    assert "failed" not in result.lower(), result
    assert seen["base_url"] == "https://scoped.example/v1"
    runner._resolve_profile_home_for_source.assert_called_once()


@pytest.mark.asyncio
async def test_compress_command_single_profile_skips_profile_resolution():
    """Multiplexing off → the scope wrapper is a transparent pass-through.

    Single-profile gateways must not pay the profile-resolution path (and
    ``_resolve_profile_home_for_source`` assumes multiplex config exists) —
    mirrors the gating contract of ``_run_agent``'s wrapper.
    """
    history = _make_history()
    runner = _make_runner(history)
    runner._resolve_profile_home_for_source = MagicMock()
    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = MagicMock()
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (list(history), "")
    agent_instance._compression_skipped_due_to_lock = False

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        await runner._handle_compress_command(_make_event())

    runner._resolve_profile_home_for_source.assert_not_called()
    runner._shutdown_executor()


@pytest.mark.asyncio
async def test_compress_command_cleanup_does_not_block_event_loop():
    """Manual /compress must not run agent teardown on the gateway event loop.

    #53175 offloaded session-expiry, hygiene, and shutdown cleanup, but the
    manual /compress finally still called ``_cleanup_agent_resources`` inline.
    A slow ``agent.close()`` there freezes the whole loop and stops the
    runtime-status heartbeat from advancing — the same wedge class as the
    original incident.

    Observation must happen from a side thread: if cleanup blocks the event
    loop, an ``await``-based waiter cannot sample ticks until close returns,
    which falsely looks healthy after the block ends.
    """
    import time

    history = _make_history()
    compressed = [
        history[0],
        {"role": "assistant", "content": "compressed summary"},
        history[-1],
    ]
    runner = _make_runner(history)

    close_started = threading.Event()
    release_close = threading.Event()

    def slow_close():
        close_started.set()
        release_close.wait(timeout=5)

    agent_instance = MagicMock()
    agent_instance.shutdown_memory_provider = MagicMock()
    agent_instance.close = slow_close
    agent_instance._cached_system_prompt = ""
    agent_instance.tools = None
    agent_instance.context_compressor.has_content_to_compress.return_value = True
    agent_instance.context_compressor._last_compress_aborted = False
    agent_instance.context_compressor._last_summary_fallback_used = False
    agent_instance.context_compressor._last_summary_dropped_count = 0
    agent_instance.context_compressor._last_summary_error = None
    agent_instance.context_compressor._last_aux_model_failure_model = None
    agent_instance.context_compressor._last_aux_model_failure_error = None
    agent_instance.session_id = "sess-1"
    agent_instance._compress_context.return_value = (compressed, "")
    agent_instance._compression_skipped_due_to_lock = False
    agent_instance._session_messages = None

    ticks = {"n": 0}
    stop = threading.Event()
    observed = {}

    async def _heartbeat():
        while not stop.is_set():
            ticks["n"] += 1
            await asyncio.sleep(0.005)

    def _observer():
        # threading.Event wait does not need the event loop. Sample ticks
        # while close() is still held so an on-loop teardown is visible.
        if not close_started.wait(timeout=5):
            observed["error"] = "close() never started"
            release_close.set()
            return
        baseline = ticks["n"]
        time.sleep(0.12)
        observed["ticks_during_block"] = ticks["n"] - baseline
        release_close.set()

    hb = asyncio.create_task(_heartbeat())
    observer = threading.Thread(target=_observer, name="compress-cleanup-observer", daemon=True)
    observer.start()

    with (
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "***"}),
        patch("gateway.run._resolve_gateway_model", return_value="test-model"),
        patch("run_agent.AIAgent", return_value=agent_instance),
        patch("agent.model_metadata.estimate_request_tokens_rough", return_value=100),
    ):
        result = await runner._handle_compress_command(_make_event())

    observer.join(timeout=5)
    stop.set()
    await hb
    runner._shutdown_executor()

    assert "Compressed:" in result
    assert "error" not in observed, observed.get("error")
    assert observed.get("ticks_during_block", 0) >= 5, (
        "event loop was blocked during manual /compress cleanup: only "
        f"{observed.get('ticks_during_block')} ticks while agent.close() was running"
    )
