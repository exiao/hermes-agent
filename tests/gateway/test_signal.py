"""Tests for Signal messenger platform adapter."""
import asyncio
import base64
import httpx
import logging
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from urllib.parse import quote

from gateway.config import Platform, PlatformConfig


@pytest.fixture(autouse=True)
def _reset_signal_scheduler():
    """The attachment scheduler is process-wide; drop it between tests
    so a fresh token bucket greets each case."""
    from gateway.platforms.signal_rate_limit import _reset_scheduler
    _reset_scheduler()
    yield
    _reset_scheduler()


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

def _make_signal_adapter(monkeypatch, account="+15551234567", **extra):
    """Create a SignalAdapter with sensible test defaults."""
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", extra.pop("group_allowed", ""))
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": account,
        **extra,
    }
    return SignalAdapter(config)


def _stub_rpc(return_value):
    """Return an async mock for SignalAdapter._rpc that captures call params."""
    captured = []

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        captured.append({"method": method, "params": dict(params)})
        return return_value

    return mock_rpc, captured


# ---------------------------------------------------------------------------
# Platform & Config
# ---------------------------------------------------------------------------

class TestSignalConfigLoading:
    def test_apply_env_overrides_signal(self, monkeypatch):
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:9090")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")

        from gateway.config import GatewayConfig, _apply_env_overrides
        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.SIGNAL in config.platforms
        sc = config.platforms[Platform.SIGNAL]
        assert sc.enabled is True
        assert sc.extra["http_url"] == "http://localhost:9090"
        assert sc.extra["account"] == "+15551234567"


# ---------------------------------------------------------------------------
# Adapter Init & Helpers
# ---------------------------------------------------------------------------

class TestSignalAdapterInit:
    def test_init_parses_config(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, group_allowed="group123,group456")
        assert adapter.http_url == "http://localhost:8080"
        assert adapter.account == "+15551234567"
        assert "group123" in adapter.group_allow_from


class TestSignalConnectCleanup:
    """Regression coverage for failed connect() cleanup."""

    @pytest.mark.asyncio
    async def test_releases_lock_and_closes_client_on_healthcheck_failure(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=503))
        mock_client.aclose = AsyncMock()

        with patch("gateway.platforms.signal.httpx.AsyncClient", return_value=mock_client), \
             patch("gateway.status.acquire_scoped_lock", return_value=(True, None)), \
             patch("gateway.status.release_scoped_lock") as mock_release:
            result = await adapter.connect()

        assert result is False
        # Two pools are built now (outbound RPC + the dedicated SSE stream),
        # and the failed-connect path must close BOTH.
        assert mock_client.aclose.await_count == 2
        mock_release.assert_called_once_with("signal-phone", "+15551234567")
        assert adapter.client is None
        assert adapter.sse_client is None
        assert adapter._platform_lock_identity is None


class TestSignalHelpers:
    def test_redact_phone_long(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("+155****4567") == "+155****4567"

    def test_redact_phone_short(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("+12345") == "+1****45"

    def test_redact_phone_empty(self):
        from gateway.platforms.helpers import redact_phone
        assert redact_phone("") == "<none>"

    def test_parse_comma_list(self):
        from gateway.platforms.signal import _parse_comma_list
        assert _parse_comma_list("+1234, +5678 , +9012") == ["+1234", "+5678", "+9012"]
        assert _parse_comma_list("") == []
        assert _parse_comma_list("  ,  ,  ") == []


    def test_guess_extension_wav_routes_to_audio_cache(self):
        """A detected WAV must route to the audio cache, not the document cache.

        ``.wav`` is already in ``_is_audio_ext``; the bug was purely that
        ``_guess_extension`` never produced ``.wav`` for raw bytes, so the
        attachment was treated as a document and STT never received it.
        """
        from gateway.platforms.signal import _is_audio_ext, _guess_extension
        wav = b"RIFF\x24\x08\x00\x00WAVEfmt " + b"\x00" * 100
        ext = _guess_extension(wav)
        assert ext == ".wav"
        assert _is_audio_ext(ext) is True


    def test_guess_extension_m4a_audio_brand(self):
        """iOS Signal voice notes are MP4-container AAC with an M4A ftyp brand.

        Classifying them as ``.mp4`` sent them to the document cache and made
        STT reject the upload ("Invalid file format") even though the bytes
        were valid audio. Audio brands must resolve to ``.m4a``.
        """
        from gateway.platforms.signal import _guess_extension, _is_audio_ext
        for brand in (b"M4A ", b"M4B ", b"m4a "):
            data = b"\x00\x00\x00\x1cftyp" + brand + b"\x00" * 100
            assert _guess_extension(data) == ".m4a", brand
            assert _is_audio_ext(_guess_extension(data)) is True


    def test_remux_aac_to_m4a_round_trip(self):
        """A real ADTS AAC stream remuxes to a valid MP4 (.m4a) container.

        Generates a short ADTS AAC sample with ffmpeg at runtime so the
        end-to-end remux path actually exercises in CI (skipped only when
        ffmpeg is unavailable), rather than depending on a machine-specific
        file.
        """
        import shutil
        import subprocess
        import tempfile
        from gateway.platforms.signal import _remux_aac_to_m4a

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            import pytest
            pytest.skip("ffmpeg not available in this env")

        # Synthesize 0.5s of silence encoded as raw ADTS AAC.
        with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp:
            adts_path = tmp.name
        try:
            gen = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi",
                 "-i", "anullsrc=r=44100:cl=mono", "-t", "0.5",
                 "-c:a", "aac", "-f", "adts", adts_path],
                capture_output=True, timeout=30,
            )
            if gen.returncode != 0:
                import pytest
                pytest.skip("ffmpeg could not produce an ADTS AAC sample")
            with open(adts_path, "rb") as f:
                aac_data = f.read()
        finally:
            try:
                import os
                os.unlink(adts_path)
            except OSError:
                pass

        result = _remux_aac_to_m4a(aac_data)
        assert result is not None
        m4a_bytes, ext = result
        assert ext == ".m4a"
        # MP4 files start with a 4-byte size, then ``ftyp`` at offset 4.
        assert m4a_bytes[4:8] == b"ftyp", \
            f"expected MP4 ftyp box, got {m4a_bytes[:12]!r}"
        # File must be at least as long as the input (MP4 has overhead).
        assert len(m4a_bytes) >= len(aac_data) * 0.5


    def test_is_image_ext(self):
        from gateway.platforms.signal import _is_image_ext
        assert _is_image_ext(".png") is True
        assert _is_image_ext(".jpg") is True
        assert _is_image_ext(".gif") is True
        assert _is_image_ext(".pdf") is False


    def test_check_requirements(self, monkeypatch):
        from gateway.platforms.signal import check_signal_requirements
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+15551234567")
        assert check_signal_requirements() is True

    def test_render_mentions(self):
        from gateway.platforms.signal import _render_mentions
        text = "Hello \uFFFC, how are you?"
        mentions = [{"start": 6, "length": 1, "number": "+155****9999"}]
        # Without bot_account, other users render as @member
        result = _render_mentions(text, mentions)
        assert "@member" in result
        assert "\uFFFC" not in result
        assert "+155****9999" not in result  # phone number must NOT leak

    def test_render_mentions_bot_self(self):
        from gateway.platforms.signal import _render_mentions
        text = "Hey \uFFFC!"
        mentions = [{"start": 4, "length": 1, "number": "+155****0000"}]
        result = _render_mentions(text, mentions, bot_account="+155****0000")
        assert "@assistant" in result
        assert "+155****0000" not in result


    def test_validate_signal_config_accepts_platform_values(self, monkeypatch):
        monkeypatch.delenv("SIGNAL_HTTP_URL", raising=False)
        monkeypatch.delenv("SIGNAL_ACCOUNT", raising=False)
        from gateway.platforms.signal import validate_signal_config

        config = PlatformConfig(
            enabled=True,
            extra={
                "http_url": "http://localhost:8080",
                "account": "+155****4567",
            },
        )
        assert validate_signal_config(config) is True


# ---------------------------------------------------------------------------
# SSE URL Encoding (Bug Fix: phone numbers with + must be URL-encoded)
# ---------------------------------------------------------------------------

class TestSignalSSEUrlEncoding:
    """Verify that phone numbers with + are URL-encoded in the SSE endpoint."""

    def test_sse_url_encodes_plus_in_account(self):
        """The + in E.164 phone numbers must be percent-encoded in the SSE query string."""
        encoded = quote("+31612345678", safe="")
        assert encoded == "%2B31612345678"

    @pytest.mark.asyncio
    async def test_force_reconnect_closes_consumed_stream_and_retries(self, monkeypatch):
        """Closing an active consumed response must release the listener context."""
        adapter = _make_signal_adapter(monkeypatch)
        monkeypatch.setattr("gateway.platforms.signal.SSE_RETRY_DELAY_INITIAL", 0)

        class BlockingSSEStream(httpx.AsyncByteStream):
            def __init__(self):
                self.started = asyncio.Event()
                self.closed = asyncio.Event()
                self.close_calls = 0

            async def __aiter__(self):
                self.started.set()
                yield b": active\\n\\n"
                await self.closed.wait()

            async def aclose(self):
                self.close_calls += 1
                self.closed.set()

        class StreamContext:
            def __init__(self, response, entered, exited):
                self.response = response
                self.entered = entered
                self.exited = exited

            async def __aenter__(self):
                self.entered.set()
                return self.response

            async def __aexit__(self, *_):
                self.exited.set()
                await self.response.aclose()

        class StreamingClient:
            def __init__(self, contexts):
                self.contexts = iter(contexts)

            def stream(self, *_args, **_kwargs):
                return next(self.contexts)

        request = httpx.Request("GET", "http://localhost:8080/api/v1/events")
        first_stream = BlockingSSEStream()
        first_response = httpx.Response(200, request=request, stream=first_stream)
        first_entered = asyncio.Event()
        first_exited = asyncio.Event()
        second_stream = BlockingSSEStream()
        second_response = httpx.Response(200, request=request, stream=second_stream)
        second_entered = asyncio.Event()
        second_exited = asyncio.Event()
        adapter.sse_client = StreamingClient([
            StreamContext(first_response, first_entered, first_exited),
            StreamContext(second_response, second_entered, second_exited),
        ])
        adapter._running = True
        listener_task = asyncio.create_task(adapter._sse_listener())

        try:
            await asyncio.wait_for(first_stream.started.wait(), timeout=1)
            assert first_response.is_stream_consumed is True

            adapter._force_reconnect()

            await asyncio.wait_for(first_stream.closed.wait(), timeout=1)
            await asyncio.wait_for(first_exited.wait(), timeout=1)
            await asyncio.wait_for(second_entered.wait(), timeout=1)
            assert first_stream.close_calls == 1
        finally:
            adapter._running = False
            await second_response.aclose()
            await listener_task

        assert second_exited.is_set()


# ---------------------------------------------------------------------------
# Attachment Fetch (Bug Fix: parameter must be "id" not "attachmentId")
# ---------------------------------------------------------------------------

class TestSignalAttachmentFetch:
    """Verify that _fetch_attachment uses the correct RPC parameter name."""

    @pytest.mark.asyncio
    async def test_fetch_attachment_uses_id_parameter(self, monkeypatch):
        """RPC getAttachment must use 'id', not 'attachmentId' (signal-cli requirement)."""
        adapter = _make_signal_adapter(monkeypatch)

        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(png_data).decode()

        adapter._rpc, captured = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_image_from_bytes", return_value="/tmp/test.png"):
            await adapter._fetch_attachment("attachment-123")

        call = captured[0]
        assert call["method"] == "getAttachment"
        assert call["params"]["id"] == "attachment-123"
        assert "attachmentId" not in call["params"], "Must NOT use 'attachmentId' — causes NullPointerException in signal-cli"
        assert call["params"]["account"] == "+15551234567"


# ---------------------------------------------------------------------------
# Session Source
# ---------------------------------------------------------------------------

class TestSignalSessionSource:

    def test_session_source_roundtrip(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.SIGNAL,
            chat_id="group:xyz",
            chat_type="group",
            user_id="+15551234567",
            user_id_alt="uuid:abc",
            chat_id_alt="xyz",
        )
        d = source.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.user_id_alt == "uuid:abc"
        assert restored.chat_id_alt == "xyz"
        assert restored.platform == Platform.SIGNAL


# ---------------------------------------------------------------------------
# Phone Redaction in agent/redact.py
# ---------------------------------------------------------------------------

class TestSignalPhoneRedaction:
    @pytest.fixture(autouse=True)
    def _ensure_redaction_enabled(self, monkeypatch):
        # agent.redact snapshots _REDACT_ENABLED at import time from the
        # HERMES_REDACT_SECRETS env var. monkeypatch.delenv is too late —
        # the module was already imported during test collection with
        # whatever value was in the env then. Force the flag directly.
        # See skill: xdist-cross-test-pollution Pattern 5.
        monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)

    def test_us_number(self):
        from agent.redact import redact_sensitive_text
        result = redact_sensitive_text("Call +15551234567 now")
        assert "+15551234567" not in result
        assert "+155" in result  # Prefix preserved
        assert "4567" in result  # Suffix preserved


# ---------------------------------------------------------------------------
# Authorization in run.py
# ---------------------------------------------------------------------------

class TestSignalAuthorization:
    def test_signal_in_allowlist_maps(self):
        """Signal should be in the platform auth maps."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.SIGNAL
        source.user_id = "+15559999999"

        # No allowlists set — should check GATEWAY_ALLOW_ALL_USERS
        with patch.dict("os.environ", {}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False


# ---------------------------------------------------------------------------
# Send Message Tool
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# send_image_file method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendImageFile:
    @pytest.mark.asyncio
    async def test_send_image_file_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_image_file should send image as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "chart.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 100)

        result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is True
        assert len(captured) == 1
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["account"] == adapter.account
        assert captured[0]["params"]["recipient"] == ["+155****4567"]
        assert captured[0]["params"]["attachments"] == [str(img_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        # Typing indicator must be stopped before sending
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        # Timestamp must be tracked for echo-back prevention
        assert 1234567890 in adapter._recent_sent_timestamps
        # Local-file media must also be available if the user quotes it later.
        assert adapter._resolve_quoted_media_paths("1234567890", "+155****4567") == [str(img_path)]


    @pytest.mark.asyncio
    async def test_send_image_file_too_large(self, monkeypatch, tmp_path):
        """send_image_file should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        img_path = tmp_path / "huge.png"
        img_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024  # 200 MB
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_image_file(chat_id="+155****4567", image_path=str(img_path))

        assert result.success is False
        assert "too large" in result.error.lower()


class TestSignalRecipientResolution:

    @pytest.mark.asyncio
    async def test_send_looks_up_uuid_via_list_contacts(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        captured = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            captured.append({"method": method, "params": dict(params)})
            if method == "listContacts":
                return [{
                    "recipient": "351935789098",
                    "number": "+15551230000",
                    "uuid": "68680952-6d86-45bc-85e0-1a4d186d53ee",
                    "isRegistered": True,
                }]
            if method == "send":
                return {"timestamp": 1234567890}
            return None

        adapter._rpc = mock_rpc

        result = await adapter.send(chat_id="+15551230000", content="hello")

        assert result.success is True
        assert captured[0]["method"] == "listContacts"
        assert captured[1]["method"] == "send"
        assert captured[1]["params"]["recipient"] == ["68680952-6d86-45bc-85e0-1a4d186d53ee"]


# ---------------------------------------------------------------------------
# send_voice method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendVoice:
    @pytest.mark.asyncio
    async def test_send_voice_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_voice should send audio as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "reply.ogg"
        audio_path.write_bytes(b"OggS" + b"\x00" * 100)

        result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["attachments"] == [str(audio_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        assert 1234567890 in adapter._recent_sent_timestamps


    @pytest.mark.asyncio
    async def test_send_voice_too_large(self, monkeypatch, tmp_path):
        """send_voice should reject files over 100MB."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        audio_path = tmp_path / "huge.ogg"
        audio_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_voice(chat_id="+155****4567", audio_path=str(audio_path))

        assert result.success is False
        assert "too large" in result.error.lower()


# ---------------------------------------------------------------------------
# send_video method (#5105)
# ---------------------------------------------------------------------------

class TestSignalSendVideo:
    @pytest.mark.asyncio
    async def test_send_video_sends_via_rpc(self, monkeypatch, tmp_path):
        """send_video should send video as attachment via signal-cli RPC."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc({"timestamp": 1234567890})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        vid_path = tmp_path / "demo.mp4"
        vid_path.write_bytes(b"\x00\x00\x00\x18ftyp" + b"\x00" * 100)

        result = await adapter.send_video(chat_id="+155****4567", video_path=str(vid_path))

        assert result.success is True
        assert captured[0]["method"] == "send"
        assert captured[0]["params"]["attachments"] == [str(vid_path)]
        assert captured[0]["params"]["message"] == ""  # caption=None → ""
        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")
        assert 1234567890 in adapter._recent_sent_timestamps


# ---------------------------------------------------------------------------
# MEDIA: tag extraction integration
# ---------------------------------------------------------------------------

class TestSignalMediaExtraction:
    """Verify the full pipeline: MEDIA: tag → extract → send_image_file/send_voice."""

    def test_extract_media_finds_image_tag(self):
        """BasePlatformAdapter.extract_media should find MEDIA: image paths."""
        from gateway.platforms.base import BasePlatformAdapter
        media, cleaned = BasePlatformAdapter.extract_media(
            "Here's the chart.\nMEDIA:/tmp/price_graph.png"
        )
        assert len(media) == 1
        assert media[0][0] == "/tmp/price_graph.png"
        assert "MEDIA:" not in cleaned


# ---------------------------------------------------------------------------
# Inbound attachment message type classification
# ---------------------------------------------------------------------------

def _make_dm_envelope(sender: str, attachments: list, text: str = "") -> dict:
    """Build a minimal signal-cli DM envelope with the given attachments."""
    return {
        "envelope": {
            "sourceNumber": sender,
            "sourceName": "Test User",
            "sourceUuid": "aaaaaaaa-0000-0000-0000-000000000001",
            "timestamp": 1700000000000,
            "dataMessage": {
                "timestamp": 1700000000000,
                "message": text,
                "expiresInSeconds": 0,
                "viewOnce": False,
                "attachments": attachments,
            },
        }
    }


class TestSignalInboundMessageTypeClassification:
    """_handle_envelope must set MessageType.DOCUMENT for application/* and text/* attachments.

    Before the fix, PDFs and other documents left msg_type as MessageType.TEXT,
    so run.py's document-context injection (which gates on MessageType.DOCUMENT)
    silently dropped the file and the agent never saw it.
    """

    async def _dispatch_single_attachment(self, monkeypatch, content_type: str,
                                          att_id: str, fetch_path: str, fetch_ext: str):
        """Helper: run _handle_envelope with one attachment and return the dispatched event."""
        envelope = _make_dm_envelope(
            sender="+15559876543",
            attachments=[{
                "contentType": content_type,
                "id": att_id,
                "size": 1024,
                "filename": None,
                "width": None,
                "height": None,
                "caption": None,
                "uploadTimestamp": 1700000000000,
            }],
        )
        adapter = _make_signal_adapter(monkeypatch)
        adapter._rpc, _ = _stub_rpc(None)
        dispatched = []

        async def _fake_handle_message(event):
            dispatched.append(event)

        adapter.handle_message = _fake_handle_message
        adapter._fetch_attachment = AsyncMock(return_value=(fetch_path, fetch_ext))
        await adapter._handle_envelope(envelope)
        assert dispatched, "_handle_envelope did not dispatch any event"
        return dispatched[0]

    @pytest.mark.asyncio
    async def test_pdf_attachment_sets_document_type(self, monkeypatch):
        """A PDF attachment (application/pdf) must produce MessageType.DOCUMENT, not TEXT."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="application/pdf",
            att_id="6zLO3b-6Yf3zVWeLDctA.pdf",
            fetch_path="/tmp/report.pdf",
            fetch_ext=".pdf",
        )

        assert event.message_type == MessageType.DOCUMENT, (
            f"Expected DOCUMENT, got {event.message_type}. "
            "PDFs must be classified as DOCUMENT so run.py injects file context."
        )
        assert "/tmp/report.pdf" in event.media_urls

    @pytest.mark.asyncio
    async def test_text_plain_attachment_sets_document_type(self, monkeypatch):
        """A text/plain attachment must produce MessageType.DOCUMENT, not TEXT."""
        from gateway.platforms.base import MessageType

        event = await self._dispatch_single_attachment(
            monkeypatch,
            content_type="text/plain",
            att_id="notes.txt",
            fetch_path="/tmp/notes.txt",
            fetch_ext=".txt",
        )

        assert event.message_type == MessageType.DOCUMENT, (
            f"Expected DOCUMENT, got {event.message_type}. "
            "text/plain must be classified as DOCUMENT so run.py injects file context."
        )


# ---------------------------------------------------------------------------
# send_document now routes through _send_attachment (#5105 bonus)
# ---------------------------------------------------------------------------

class TestSignalSendDocumentViaHelper:
    """Verify send_document gained size check and path-in-error via _send_attachment."""

    @pytest.mark.asyncio
    async def test_send_document_too_large(self, monkeypatch, tmp_path):
        """send_document should now reject files over 100MB (was previously missing)."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        doc_path = tmp_path / "huge.pdf"
        doc_path.write_bytes(b"x")

        def mock_stat(self, **kwargs):
            class FakeStat:
                st_size = 200 * 1024 * 1024
            return FakeStat()

        with patch.object(Path, "stat", mock_stat):
            result = await adapter.send_document(chat_id="+155****4567", file_path=str(doc_path))

        assert result.success is False
        assert "too large" in result.error.lower()


# ---------------------------------------------------------------------------
# Signal streaming edit capability / message_id behavior
# ---------------------------------------------------------------------------

class TestSignalStreamingCapabilities:
    """Signal must opt out of edit-based streaming behavior."""

    def test_signal_declares_no_message_editing(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        assert adapter.SUPPORTS_MESSAGE_EDITING is False


class TestSignalSendReturnsMessageId:
    """Signal send() should not pretend sent messages are editable."""

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_even_with_timestamp(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({"timestamp": 1712345678000})
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None

    @pytest.mark.asyncio
    async def test_send_chunks_long_markdown_with_independent_formatting(self, monkeypatch):
        """Long markdown is sent in ordered, separately formatted Signal messages."""
        from gateway.platforms.signal import MAX_MESSAGE_LENGTH

        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        sent = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            sent.append({"method": method, "params": dict(params)})
            return {"timestamp": len(sent)}

        adapter._rpc = mock_rpc
        content = "**Opening paragraph**\n\n" + ("word " * 1600) + "**Closing paragraph**"

        result = await adapter.send(chat_id="+155****4567", content=content)

        assert adapter.splits_long_messages is True
        assert result.success is True
        assert result.message_id is None
        assert [call["method"] for call in sent] == ["send", "send"]
        assert all(len(call["params"]["message"]) <= MAX_MESSAGE_LENGTH for call in sent)
        assert sent[0]["params"]["message"].startswith("Opening paragraph")
        assert "Closing paragraph" in sent[1]["params"]["message"]
        assert all("**" not in call["params"]["message"] for call in sent)
        for call in sent:
            styles = [call["params"].get("textStyle", ""), *call["params"].get("textStyles", [])]
            assert any(style.endswith(":BOLD") for style in styles)
        assert {1, 2} <= set(adapter._recent_sent_timestamps)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("marker", "style"),
        [
            ("**", "BOLD"),
            ("__", "BOLD"),
            ("*", "ITALIC"),
            ("_", "ITALIC"),
            ("~~", "STRIKETHROUGH"),
        ],
    )
    async def test_send_preserves_inline_style_across_chunk_boundary(self, monkeypatch, marker, style):
        """An inline style span remains native-formatted when it needs two sends."""
        from gateway.platforms.signal import MAX_MESSAGE_LENGTH

        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        sent = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            sent.append(dict(params))
            return {"timestamp": len(sent)}

        adapter._rpc = mock_rpc

        result = await adapter.send(
            chat_id="+155****4567",
            content=marker + ("x" * (MAX_MESSAGE_LENGTH + 100)) + marker,
        )

        assert result.success is True
        assert len(sent) == 2
        assert all(marker not in params["message"] for params in sent)
        for params in sent:
            styles = [params.get("textStyle", ""), *params.get("textStyles", [])]
            assert any(text_style.endswith(f":{style}") for text_style in styles)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("marker", "style"),
        [("**", "BOLD"), ("__", "BOLD"), ("~~", "STRIKETHROUGH")],
    )
    async def test_send_preserves_style_when_closing_delimiter_straddles_chunk_boundary(
        self, monkeypatch, marker, style
    ):
        """A two-character closing delimiter can itself straddle the raw split."""
        from gateway.platforms.signal import MAX_MESSAGE_LENGTH

        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        sent = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            sent.append(dict(params))
            return {"timestamp": len(sent)}

        adapter._rpc = mock_rpc
        result = await adapter.send(
            chat_id="+155****4567",
            content=marker + ("x" * 7983) + marker + ("z" * 100),
        )

        assert result.success is True
        assert len(sent) == 2
        assert all(marker not in params["message"] for params in sent)
        first_styles = [sent[0].get("textStyle", ""), *sent[0].get("textStyles", [])]
        assert any(text_style.endswith(f":{style}") for text_style in first_styles)

    @pytest.mark.asyncio
    async def test_send_does_not_carry_italic_after_closing_marker_before_space(self, monkeypatch):
        from gateway.platforms.signal import MAX_MESSAGE_LENGTH

        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        sent = []

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            sent.append(dict(params))
            return {"timestamp": len(sent)}

        adapter._rpc = mock_rpc
        result = await adapter.send(
            chat_id="+155****4567",
            content="Prefix *important* " + ("x" * (MAX_MESSAGE_LENGTH + 100)),
        )

        assert result.success is True
        assert "*" not in sent[0]["message"]
        assert not any(
            style.endswith(":ITALIC")
            for style in [sent[1].get("textStyle", ""), *sent[1].get("textStyles", [])]
        )

    @pytest.mark.asyncio
    async def test_send_ignores_markers_inside_code_spans(self, monkeypatch):
        from gateway.platforms.signal import MAX_MESSAGE_LENGTH

        for content in (
            "```\n" + ("x" * 100) + "**" + ("x" * MAX_MESSAGE_LENGTH) + "\n```",
            "`**` " + ("x" * (MAX_MESSAGE_LENGTH + 100)),
        ):
            adapter = _make_signal_adapter(monkeypatch)
            adapter._stop_typing_indicator = AsyncMock()
            sent = []

            async def mock_rpc(method, params, rpc_id=None, **kwargs):
                sent.append(dict(params))
                return {"timestamp": len(sent)}

            adapter._rpc = mock_rpc
            result = await adapter.send(chat_id="+155****4567", content=content)

            assert result.success is True
            assert "".join(params["message"] for params in sent).count("**") == 1

    @pytest.mark.asyncio
    async def test_send_marks_partial_delivery_on_later_chunk_failure(self, monkeypatch):
        from gateway.platforms.signal import MAX_MESSAGE_LENGTH

        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()
        calls = 0

        async def mock_rpc(method, params, rpc_id=None, **kwargs):
            nonlocal calls
            calls += 1
            return {"timestamp": 1} if calls == 1 else None

        adapter._rpc = mock_rpc
        result = await adapter.send(
            chat_id="+155****4567", content="x" * (MAX_MESSAGE_LENGTH + 100)
        )

        assert result.success is False
        assert result.partial_delivery is True
        assert result.error == "Partial send after 1/2 chunks: RPC send failed"

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_when_no_timestamp(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({})  # No timestamp key
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None

    @pytest.mark.asyncio
    async def test_send_returns_none_message_id_for_non_dict(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc("ok")  # Non-dict result
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")

        assert result.success is True
        assert result.message_id is None


class TestSignalSendResultValidation:
    """Verify that send() validates recipient-level delivery results."""

    @pytest.mark.asyncio
    async def test_send_preserves_ambiguous_rpc_error(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        response = MagicMock()
        response.raise_for_status.side_effect = RuntimeError(
            "org.signal.network.exceptions.NonSuccessfulResponseCodeException: [500] Bad response: 500"
        )
        adapter.client = AsyncMock()
        adapter.client.post = AsyncMock(return_value=response)
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="group:group123", content="hello")

        assert not result.success
        assert result.error is not None
        assert "[500] Bad response: 500" in result.error

    @pytest.mark.asyncio
    async def test_send_success_when_results_has_success(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"number": "+155****4567"},
                    "type": "SUCCESS"
                }
            ]
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_failure_when_results_has_failure_type(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"number": "+155****4567"},
                    "type": "UNREGISTERED_FAILURE"
                }
            ]
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")
        assert result.success is False
        assert result.error == "UNREGISTERED_FAILURE"

    @pytest.mark.asyncio
    async def test_send_failure_when_results_has_success_false(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, _ = _stub_rpc({
            "timestamp": 1712345678000,
            "results": [
                {
                    "recipientAddress": {"number": "+155****4567"},
                    "success": False,
                    "failure": "Some connection error"
                }
            ]
        })
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        result = await adapter.send(chat_id="+155****4567", content="hello")
        assert result.success is False
        assert result.error == "Some connection error"


# ---------------------------------------------------------------------------
# stop_typing() delegates to _stop_typing_indicator (#4647)
# ---------------------------------------------------------------------------

class TestSignalStopTyping:
    """Signal must expose a public stop_typing() so base adapter's
    _keep_typing finally block can clean up platform-level typing tasks."""

    @pytest.mark.asyncio
    async def test_stop_typing_calls_private_method(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.stop_typing("+155****4567")

        adapter._stop_typing_indicator.assert_awaited_once_with("+155****4567")


# ---------------------------------------------------------------------------
# Reply quote extraction
# ---------------------------------------------------------------------------

class TestSignalQuoteExtraction:
    """Verify Signal reply quote fields are propagated to MessageEvent."""

    @pytest.mark.asyncio
    async def test_handle_envelope_sets_reply_context_from_quote(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "yes I agree",
                    "quote": {
                        "id": 99,
                        "text": "want to grab lunch?",
                        "author": "+15550002222",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.text == "yes I agree"
        assert event.reply_to_message_id == "99"
        assert event.reply_to_text == "want to grab lunch?"

    @pytest.mark.asyncio
    async def test_handle_envelope_without_quote_leaves_reply_fields_none(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "plain message",
                },
            }
        })

        event = captured["event"]
        assert event.text == "plain message"
        assert event.reply_to_message_id is None
        assert event.reply_to_text is None

    @pytest.mark.asyncio
    async def test_handle_envelope_quote_without_text_sets_only_reply_id(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "reply without quote text",
                    "quote": {
                        "id": 123,
                        "author": "+15550002222",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.reply_to_message_id == "123"
        assert event.reply_to_text is None


# ---------------------------------------------------------------------------
# Typing-indicator backoff on repeated failures (Signal RPC spam fix)
# ---------------------------------------------------------------------------

class TestSignalTypingBackoff:
    """When base.py's _keep_typing refresh loop calls send_typing every ~2s
    and the recipient is unreachable (NETWORK_FAILURE), the adapter must:

    - log WARNING only for the first failure (subsequent failures use DEBUG
      via log_failures=False on the _rpc call)
    - after 3 consecutive failures, skip the RPC entirely during an
      exponential cooldown window instead of hammering signal-cli every 2s
    - reset counters on a successful sendTyping
    - reset counters when _stop_typing_indicator() is called for the chat
    """

    @pytest.mark.asyncio
    async def test_first_failure_logs_at_warning_subsequent_at_debug(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        calls = []

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            calls.append({"log_failures": log_failures})
            return None  # simulate NETWORK_FAILURE

        adapter._rpc = _fake_rpc

        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")

        assert len(calls) == 2
        assert calls[0]["log_failures"] is True   # first failure — warn
        assert calls[1]["log_failures"] is False  # subsequent — debug

    @pytest.mark.asyncio
    async def test_three_consecutive_failures_trigger_cooldown(
        self, monkeypatch
    ):
        adapter = _make_signal_adapter(monkeypatch)
        call_count = {"n": 0}

        async def _fake_rpc(method, params, rpc_id=None, *, log_failures=True):
            call_count["n"] += 1
            return None

        adapter._rpc = _fake_rpc

        # Three failures engage the cooldown.
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        assert call_count["n"] == 3
        assert "+155****4567" in adapter._typing_skip_until

        # Fourth, fifth, ... calls during the cooldown window are short-
        # circuited — the RPC is not issued at all.
        await adapter.send_typing("+155****4567")
        await adapter.send_typing("+155****4567")
        assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# _stop_typing_indicator sends explicit sendTyping(stop=True) RPC
# ---------------------------------------------------------------------------

class TestSignalStopTypingExplicitRPC:
    """Cancelling the typing indicator must issue an explicit
    sendTyping(stop=True) RPC so the recipient's device drops the indicator
    immediately, instead of waiting for Signal's built-in ~5s timeout.

    The stop RPC is best-effort: any failure must not prevent the per-chat
    backoff state from being cleared.
    """


    @pytest.mark.asyncio
    async def test_stop_typing_indicator_best_effort_on_rpc_failure(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._resolve_recipient = AsyncMock(return_value="uuid-recipient")

        # Drive the chat into backoff so we can confirm cleanup still happens
        # even when the stop RPC itself fails.
        async def _noop(method, params, rpc_id=None, **kwargs):
            return None

        adapter._rpc = _noop
        for _ in range(3):
            await adapter.send_typing("+155****0000")

        assert adapter._typing_failures.get("+155****0000") == 3
        assert "+155****0000" in adapter._typing_skip_until

        # Now make the stop RPC raise — backoff state must still be cleared.
        async def failing_rpc(method, params, rpc_id=None, **kwargs):
            raise RuntimeError("signal-cli unreachable")

        adapter._rpc = failing_rpc

        await adapter._stop_typing_indicator("+155****0000")

        assert "+155****0000" not in adapter._typing_failures
        assert "+155****0000" not in adapter._typing_skip_until


# ---------------------------------------------------------------------------
# Reply quote extraction
# ---------------------------------------------------------------------------

class TestSignalQuoteExtraction:
    """Verify Signal reply quote fields are propagated to MessageEvent."""

    @pytest.mark.asyncio
    async def test_handle_envelope_sets_reply_context_from_quote(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+15550001111",
                "sourceUuid": "uuid-sender",
                "sourceName": "Tester",
                "timestamp": 1000000000,
                "dataMessage": {
                    "message": "yes I agree",
                    "quote": {
                        "id": 99,
                        "text": "want to grab lunch?",
                        "author": "other-author",
                    },
                },
            }
        })

        event = captured["event"]
        assert event.text == "yes I agree"
        assert event.reply_to_message_id == "99"
        assert event.reply_to_text == "want to grab lunch?"
        assert event.reply_to_author_id == "other-author"
        assert event.reply_to_is_own_message is False


    @pytest.mark.asyncio
    async def test_track_sent_timestamp_keeps_reply_detection_cache_after_echo_discard(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._track_sent_timestamp({"timestamp": 111222333})
        # Echo suppression consumes the entry from the recent-sent ring; the
        # separate reply-detection cache must still retain it.
        adapter._consume_sent_timestamp(111222333)

        assert "111222333" in adapter._sent_message_timestamps
        assert adapter._quote_references_own_message("111222333", None) is True


# ---------------------------------------------------------------------------
# _rpc rate-limit detection
# ---------------------------------------------------------------------------

class _FakeHttpResponse:
    """Minimal stand-in for httpx.Response — only what _rpc touches."""

    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


def _install_fake_client(adapter, json_data):
    """Replace adapter.client.post with an async fn returning json_data."""
    from types import SimpleNamespace

    async def _post(url, json=None, timeout=None):
        return _FakeHttpResponse(json_data)

    adapter.client = SimpleNamespace(post=_post)


class TestSignalRpcRateLimit:
    """_rpc opt-in 429 detection and SignalRateLimitError propagation."""


    @pytest.mark.asyncio
    async def test_default_swallows_rate_limit_returns_none(self, monkeypatch):
        """Without opt-in, 429 stays swallowed — preserves backwards compat."""
        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {"message": "[429] Rate Limited"},
        })

        result = await adapter._rpc("send", {})
        assert result is None


    @pytest.mark.asyncio
    async def test_raises_with_retry_after_from_v0_14_3_payload(self, monkeypatch):
        """signal-cli ≥ v0.14.3 surfaces server Retry-After under
        ``error.data.response.results[*].retryAfterSeconds`` — _rpc
        carries that value through SignalRateLimitError.retry_after."""
        from gateway.platforms.signal_rate_limit import (
            SignalRateLimitError, SIGNAL_RPC_ERROR_RATELIMIT,
        )

        adapter = _make_signal_adapter(monkeypatch)
        _install_fake_client(adapter, {
            "error": {
                "code": SIGNAL_RPC_ERROR_RATELIMIT,
                "message": "Failed to send message due to rate limiting",
                "data": {
                    "response": {
                        "timestamp": 0,
                        "results": [
                            {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 90},
                        ],
                    }
                },
            },
        })

        with pytest.raises(SignalRateLimitError) as exc_info:
            await adapter._rpc("send", {}, raise_on_rate_limit=True)

        assert exc_info.value.retry_after == 90.0


# ---------------------------------------------------------------------------
# send_multiple_images — chunking, pacing, rate-limit retry
# ---------------------------------------------------------------------------


def _make_image_files(tmp_path, count, prefix="img"):
    """Materialize `count` tiny PNG files and return file:// URIs for them."""
    uris = []
    for i in range(count):
        p = tmp_path / f"{prefix}_{i}.png"
        p.write_bytes(b"\x89PNG" + b"\x00" * 32)
        uris.append((f"file://{p}", ""))
    return uris


def _stub_rpc_responses(responses):
    """Build an _rpc replacement that pops a response per call.

    Each entry in `responses` is either:
      * a return value (dict / None) → returned to the caller, or
      * an Exception subclass instance → raised.
    Captures (params, kwargs) per call for inspection.
    """
    captured = []
    queue = list(responses)

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        captured.append({"method": method, "params": dict(params), "kwargs": kwargs})
        await asyncio.sleep(0)
        if not queue:
            raise AssertionError("Unexpected extra _rpc call")
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return mock_rpc, captured


def _patch_scheduler_sleep(monkeypatch, capture: list):
    """Capture sleeps inside the scheduler so tests don't actually wait.
    Zero-second sleeps (e.g. event-loop yields from mock RPCs) are
    delegated to the real asyncio.sleep so they don't pollute the
    capture list."""
    _real_sleep = asyncio.sleep
    offset = [0.0]

    async def fake_sleep(seconds):
        if seconds > 0:
            capture.append(seconds)
            offset[0] += seconds
        else:
            await _real_sleep(0)

    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.asyncio.sleep", fake_sleep
    )
    monkeypatch.setattr(
        "gateway.platforms.signal_rate_limit.time.monotonic", lambda: offset[0]
    )


class TestSignalSendMultipleImages:
    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(chat_id="+155****4567", images=[])

        assert captured == []
        adapter._stop_typing_indicator.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_bad_files_no_rpc(self, monkeypatch, tmp_path):
        """If every image is missing/invalid, no RPC fires."""
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=[(f"file://{tmp_path}/missing_a.png", ""),
                    (f"file://{tmp_path}/missing_b.png", "")],
        )

        assert captured == []

    @pytest.mark.asyncio
    async def test_single_batch_under_limit(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([{"timestamp": 1}])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        images = _make_image_files(tmp_path, 5)
        await adapter.send_multiple_images(chat_id="+155****4567", images=images)

        assert len(captured) == 1
        params = captured[0]["params"]
        assert params["recipient"] == ["+155****4567"]
        assert params["message"] == ""
        assert len(params["attachments"]) == 5
        # raise_on_rate_limit must be opted into so the retry loop sees 429s
        assert captured[0]["kwargs"].get("raise_on_rate_limit") is True


    @pytest.mark.asyncio
    async def test_429_without_retry_after_uses_default_rate(
        self, monkeypatch, tmp_path
    ):
        """signal-cli < v0.14.3 doesn't surface Retry-After. The
        scheduler keeps its default refill rate (1 token / 4s), so a
        retry of n=3 waits 12s."""
        from gateway.platforms.signal_rate_limit import (
            SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER,
            SignalRateLimitError,
        )

        adapter = _make_signal_adapter(monkeypatch)
        mock_rpc, captured = _stub_rpc_responses([
            SignalRateLimitError("[429] Rate Limited", retry_after=None),
            {"timestamp": 99},
        ])
        adapter._rpc = mock_rpc
        adapter._stop_typing_indicator = AsyncMock()

        sleep_calls: list = []
        _patch_scheduler_sleep(monkeypatch, sleep_calls)

        await adapter.send_multiple_images(
            chat_id="+155****4567",
            images=_make_image_files(tmp_path, 3),
        )

        assert len(captured) == 2
        assert sleep_calls == [
            pytest.approx(3 * SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER, abs=1.0)
        ]


class TestSignalRateLimitDetection:
    """Coverage for the typed-code + substring detection helpers."""


    def test_extract_retry_after_from_results(self):
        from gateway.platforms.signal import _extract_retry_after_seconds
        err = {
            "code": -5,
            "message": "Failed to send message due to rate limiting",
            "data": {
                "response": {
                    "timestamp": 0,
                    "results": [
                        {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 30},
                        {"type": "RATE_LIMIT_FAILURE", "retryAfterSeconds": 45},
                    ],
                }
            },
        }
        assert _extract_retry_after_seconds(err) == 45.0


    def test_detect_retry_later_exception_substring(self):
        """libsignal-net's RetryLaterException leaks through as
        AttachmentInvalidException → UnexpectedErrorException when the
        rate-limit fires inside attachment upload. Detect it by substring."""
        from gateway.platforms.signal import _is_signal_rate_limit_error
        err = {
            "code": -32603,
            "message": (
                "Failed to send message: /home/max/sync/Memes/fengshui.jpeg: "
                "org.signal.libsignal.net.RetryLaterException: Retry after 4 seconds "
                "(AttachmentInvalidException) (UnexpectedErrorException)"
            ),
        }
        assert _is_signal_rate_limit_error(err) is True


class TestSignalSendTimeout:
    """Deadline policy for Signal send RPCs."""

    def test_zero_attachments_uses_default(self):
        from gateway.platforms.signal import _signal_send_timeout
        # Text-only sends never leave the local daemon, so a hang is a real
        # failure and stays bounded.
        assert _signal_send_timeout(0) == 30.0

    def test_attachment_sends_leave_the_upload_leg_open(self):
        from gateway.platforms.signal import _signal_send_timeout
        # signal-cli receives paths and uploads to Signal's servers itself,
        # so the client is waiting on a remote uplink of unknown speed. Any
        # read deadline here is a hidden minimum-bandwidth assumption.
        for count in (1, 5, 32):
            assert _signal_send_timeout(count).read is None

    def test_attachment_sends_still_bound_connect(self):
        from gateway.platforms.signal import _signal_send_timeout
        # An unreachable or dead daemon must still fail fast rather than
        # hanging the send forever.
        assert _signal_send_timeout(1).connect == 60.0


# ---------------------------------------------------------------------------
# Contentless Envelope Filtering (profile key updates, empty messages)
# ---------------------------------------------------------------------------

class TestSignalContentlessEnvelope:
    """Verify that profile key updates and empty Signal messages are skipped."""

    @pytest.mark.asyncio
    async def test_skips_profile_key_update_no_message_field(self, monkeypatch):
        """Profile key updates may carry a dataMessage without 'message' field.
        Must be skipped to avoid triggering agent turns for metadata."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        # Profile key update: dataMessage exists but has no "message" field
        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****9999",
                "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                "sourceName": "Elliott McManis",
                "timestamp": 1777600696077,
                "dataMessage": {
                    # No "message" field — profile key update metadata only
                    "profileKey": "some-profile-key-data",
                },
            }
        })

        assert "event" not in captured, "Profile key update should be skipped"


    @pytest.mark.asyncio
    async def test_allows_message_with_attachment_no_text(self, monkeypatch):
        """Messages with attachments but no text should still be processed."""
        adapter = _make_signal_adapter(monkeypatch)
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        # Mock attachment fetch to return a cached image
        png_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        b64_data = base64.b64encode(png_data).decode()
        adapter._rpc, _ = _stub_rpc({"data": b64_data})

        with patch("gateway.platforms.signal.cache_image_from_bytes", return_value="/tmp/img.png"):
            await adapter._handle_envelope({
                "envelope": {
                    "sourceNumber": "+155****9999",
                    "sourceUuid": "05668cf3-8ffa-467e-9b24-f5eefa5cf475",
                    "sourceName": "Elliott McManis",
                    "timestamp": 1777600696077,
                    "dataMessage": {
                        "message": "",  # No text
                        "attachments": [{"id": "att-123", "size": 200}],
                    },
                }
            })

        assert "event" in captured, "Message with attachment should NOT be skipped"
        assert captured["event"].media_urls == ["/tmp/img.png"]


class TestSignalSyncMessageHandling:
    """signal-cli running as a linked secondary device receives the user's
    own messages as ``syncMessage.sentMessage`` envelopes. Two cases must
    be handled:

      1. Note to Self (destination == self): promote to dataMessage so the
         user can talk to the agent in their own self-chat.
      2. Group sync-sent (destination is None, groupInfo set): promote so
         single-user / personal groups work.

    In both cases, the bot's own outbound replies bounce back as
    sync-sents and must be suppressed via the recently-sent timestamp ring.
    """


    @pytest.mark.asyncio
    async def test_note_to_self_echo_of_own_reply_is_suppressed(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch, account="+155****4567")
        # Simulate that the bot just sent a reply with timestamp 3000000000
        adapter._track_sent_timestamp({"timestamp": 3000000000})
        called = []

        async def fake_handle(event):
            called.append(event)

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "sourceUuid": "uuid-self",
                "timestamp": 3000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": "+155****4567",
                        "destination": "+155****4567",
                        "timestamp": 3000000000,
                        "message": "this is the bot's own reply echo",
                    }
                },
            }
        })

        assert called == [], "Echo of bot's own reply must be suppressed"
        # Consumed: timestamp must be removed from the ring
        assert 3000000000 not in adapter._recent_sent_timestamps

    @pytest.mark.asyncio
    async def test_group_sync_sent_promoted_to_inbound(self, monkeypatch):
        """User sends a message in a group from their primary phone; the
        linked device receives it as a sync-sent with destination=None and
        a groupInfo block. It must be treated as inbound so the agent can
        respond in groups when the user is the only human participant."""
        adapter = _make_signal_adapter(
            monkeypatch, account="+155****4567", group_allowed="abc123=="
        )
        captured = {}

        async def fake_handle(event):
            captured["event"] = event

        adapter.handle_message = fake_handle

        await adapter._handle_envelope({
            "envelope": {
                "sourceNumber": "+155****4567",
                "sourceUuid": "uuid-self",
                "timestamp": 4000000000,
                "syncMessage": {
                    "sentMessage": {
                        "destinationNumber": None,
                        "destination": None,
                        "timestamp": 4000000000,
                        "message": "ping the group",
                        "groupInfo": {
                            "groupId": "abc123==",
                            "type": "DELIVER",
                        },
                    }
                },
            }
        })

        assert "event" in captured, "Group sync-sent must reach handle_message"
        assert captured["event"].text == "ping the group"
        assert captured["event"].source.chat_id == "group:abc123=="


class TestRecentSentTimestampRing:
    """Verify the LRU+TTL behaviour of the echo-suppression ring."""


    def test_ttl_evicts_stale_entries(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._recent_sent_ttl_seconds = 100.0

        # Drive time.monotonic deterministically.
        import gateway.platforms.signal as sig_mod
        fake_now = [1000.0]
        monkeypatch.setattr(sig_mod.time, "monotonic", lambda: fake_now[0])

        adapter._track_sent_timestamp({"timestamp": 1})
        fake_now[0] = 1050.0
        adapter._track_sent_timestamp({"timestamp": 2})
        fake_now[0] = 1200.0  # 200s elapsed since ts=1 (>TTL), 150s since ts=2 (>TTL)
        adapter._track_sent_timestamp({"timestamp": 3})
        # Both 1 and 2 should be evicted on TTL, only 3 remains
        assert list(adapter._recent_sent_timestamps.keys()) == [3]


# ---------------------------------------------------------------------------
# Quoted attachment metadata (reply-to an image)
# ---------------------------------------------------------------------------

class TestQuotedAttachments:
    """signal-cli reports quote.attachments[] with contentType + filename.

    The adapter used to ignore it, so quoting an image produced a generic
    "may have been an image or file" pointer and the agent had to ask which
    one was meant.
    """

    def test_describes_a_quoted_image(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        summary = adapter._describe_quoted_attachments({
            "id": 1753650000000,
            "attachments": [{"contentType": "image/png", "filename": "sheet.png"}],
        })
        assert summary == "an image (image/png, sheet.png)"

    def test_describes_video_audio_and_generic_files(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter._describe_quoted_attachments(
            {"attachments": [{"contentType": "video/mp4"}]}
        ) == "a video (video/mp4)"
        assert adapter._describe_quoted_attachments(
            {"attachments": [{"contentType": "audio/ogg"}]}
        ) == "an audio file (audio/ogg)"
        assert adapter._describe_quoted_attachments(
            {"attachments": [{"contentType": "application/pdf", "filename": "memo.pdf"}]}
        ) == "a file (application/pdf, memo.pdf)"

    def test_handles_missing_filename(self, monkeypatch):
        """iOS often sends no filename; still say it was an image."""
        adapter = _make_signal_adapter(monkeypatch)
        summary = adapter._describe_quoted_attachments({
            "attachments": [{"contentType": "image/jpeg"}],
        })
        assert summary == "an image (image/jpeg)"

    def test_handles_multiple_attachments(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        summary = adapter._describe_quoted_attachments({
            "attachments": [
                {"contentType": "image/png", "filename": "a.png"},
                {"contentType": "image/png", "filename": "b.png"},
            ],
        })
        assert summary.startswith("2 attachments:")
        assert "a.png" in summary and "b.png" in summary

    def test_returns_none_when_quote_has_no_attachments(self, monkeypatch):
        """A text-only quote must not be described as media."""
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter._describe_quoted_attachments({"id": 1, "text": "hi"}) is None
        assert adapter._describe_quoted_attachments({"attachments": []}) is None
        assert adapter._describe_quoted_attachments({}) is None
        assert adapter._describe_quoted_attachments(None) is None

    def test_survives_malformed_attachment_entries(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter._describe_quoted_attachments({"attachments": ["nonsense", 42]}) is None
        assert adapter._describe_quoted_attachments(
            {"attachments": [{}]}
        ) == "a file"

    def test_resolves_local_path_for_an_image_we_sent(self, monkeypatch, tmp_path):
        """Quoting an image the bot sent resolves back to the file on disk."""
        adapter = _make_signal_adapter(monkeypatch)
        real_file = tmp_path / "chart.png"
        real_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        adapter._remember_sent_attachments(1753650000000, "+15559998888", [str(real_file)])

        assert adapter._resolve_quoted_media_paths("1753650000000", "+15559998888") == [str(real_file)]

    def test_no_local_path_for_media_the_user_sent(self, monkeypatch):
        """A photo the USER took was never on this machine — no path, no lie."""
        adapter = _make_signal_adapter(monkeypatch)
        assert adapter._resolve_quoted_media_paths("1753650000000", "+15559998888") == []
        assert adapter._resolve_quoted_media_paths(None, "+15559998888") == []

    def test_drops_paths_that_no_longer_exist(self, monkeypatch, tmp_path):
        """A since-deleted temp file must not be advertised to the agent."""
        adapter = _make_signal_adapter(monkeypatch)
        gone = tmp_path / "deleted.png"
        gone.write_bytes(b"x")
        adapter._remember_sent_attachments(1753650000000, "+15559998888", [str(gone)])
        gone.unlink()

        assert adapter._resolve_quoted_media_paths("1753650000000", "+15559998888") == []

    def test_sent_attachment_cache_is_bounded_and_fifo(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._max_sent_attachment_entries = 3
        for i in range(5):
            adapter._remember_sent_attachments(i, "+15559998888", [f"/tmp/{i}.png"])

        assert len(adapter._sent_attachment_paths) == 3
        # Oldest evicted first.
        assert ("+15559998888", "0") not in adapter._sent_attachment_paths
        assert ("+15559998888", "4") in adapter._sent_attachment_paths

    def test_same_timestamp_in_two_conversations_both_survive(self, monkeypatch, tmp_path):
        """Concurrent sends sharing a millisecond must not evict each other."""
        adapter = _make_signal_adapter(monkeypatch)
        first = tmp_path / "a.png"
        second = tmp_path / "b.png"
        first.write_bytes(b"a")
        second.write_bytes(b"b")

        adapter._remember_sent_attachments(1753650000000, "+15559998888", [str(first)])
        adapter._remember_sent_attachments(1753650000000, "+15551112222", [str(second)])

        assert adapter._resolve_quoted_media_paths("1753650000000", "+15559998888") == [str(first)]
        assert adapter._resolve_quoted_media_paths("1753650000000", "+15551112222") == [str(second)]

    def test_remembering_is_a_noop_without_paths(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        adapter._remember_sent_attachments(123, "+15559998888", [])
        adapter._remember_sent_attachments(None, "+15559998888", ["/tmp/x.png"])
        assert len(adapter._sent_attachment_paths) == 0


@pytest.mark.asyncio
async def test_quoted_image_envelope_end_to_end(monkeypatch, tmp_path):
    """Drive a realistic signal-cli quote envelope through the real handler.

    Envelope shape mirrors signal-cli's JsonQuote/JsonQuotedAttachment records:
    quote.attachments[] carries contentType + filename, and quote.id is the
    timestamp of the quoted (bot-sent) message.

    handle_message() dispatches onto a background task, so we capture the event
    at that seam rather than racing the task.
    """
    adapter = _make_signal_adapter(monkeypatch)

    sent_image = tmp_path / "cc_verification_sheet.png"
    sent_image.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter._remember_sent_attachments(1753650000000, "+15559998888", [str(sent_image)])
    adapter._remember_sent_message_timestamp(1753650000000)

    captured = {}

    async def _capture(event):
        captured["event"] = event

    monkeypatch.setattr(adapter, "handle_message", _capture)

    envelope = {
        "envelope": {
            "source": "+15559998888",
            "sourceName": "E X",
            "sourceNumber": "+15559998888",
            "timestamp": 1753650500000,
            "dataMessage": {
                "message": "what about this part?",
                "timestamp": 1753650500000,
                "quote": {
                    "id": 1753650000000,
                    "author": "+15201234567",
                    "text": None,
                    "attachments": [
                        {"contentType": "image/png", "filename": "cc_verification_sheet.png"}
                    ],
                },
                "attachments": [],
            },
        }
    }

    await adapter._handle_envelope(envelope)

    event = captured.get("event")
    assert event is not None, "handler did not emit a MessageEvent"
    assert event.reply_to_text is None
    assert event.reply_to_media_summary == "an image (image/png, cc_verification_sheet.png)"
    assert event.reply_to_media_paths == [str(sent_image)]


@pytest.mark.asyncio
async def test_quoted_image_from_user_has_summary_but_no_path(monkeypatch):
    """A photo the USER sent is described but has no local copy to offer."""
    adapter = _make_signal_adapter(monkeypatch)

    captured = {}

    async def _capture(event):
        captured["event"] = event

    monkeypatch.setattr(adapter, "handle_message", _capture)

    envelope = {
        "envelope": {
            "source": "+15559998888",
            "sourceName": "E X",
            "timestamp": 1753650500000,
            "dataMessage": {
                "message": "this photo",
                "timestamp": 1753650500000,
                "quote": {
                    "id": 1753649000000,
                    "author": "+15559998888",
                    "text": None,
                    "attachments": [{"contentType": "image/jpeg"}],
                },
                "attachments": [],
            },
        }
    }

    await adapter._handle_envelope(envelope)

    event = captured.get("event")
    assert event is not None
    assert event.reply_to_media_summary == "an image (image/jpeg)"
    assert event.reply_to_media_paths == []


@pytest.mark.asyncio
async def test_uuid_only_quote_envelope_preserves_number_alias(monkeypatch, tmp_path):
    """A UUID-only reply must retain the number learned from an earlier envelope."""
    adapter = _make_signal_adapter(monkeypatch)
    peer_number = "+15559998888"
    peer_uuid = "aaaaaaaa-0000-0000-0000-000000000002"
    sent_image = tmp_path / "quoted.png"
    sent_image.write_bytes(b"\x89PNG\r\n\x1a\n")
    quoted_timestamp = 1753650000000
    adapter._remember_sent_attachments(quoted_timestamp, peer_number, [str(sent_image)])
    adapter._remember_sent_message_timestamp(quoted_timestamp)

    captured = []

    async def _capture(event):
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", _capture)

    await adapter._handle_envelope({
        "envelope": {
            "sourceNumber": peer_number,
            "sourceUuid": peer_uuid,
            "timestamp": 1753650100000,
            "dataMessage": {"message": "sent earlier", "timestamp": 1753650100000},
        }
    })
    await adapter._handle_envelope({
        "envelope": {
            "sourceUuid": peer_uuid,
            "timestamp": 1753650200000,
            "dataMessage": {
                "message": "what about this image?",
                "timestamp": 1753650200000,
                "quote": {
                    "id": quoted_timestamp,
                    "author": adapter._account_normalized,
                    "attachments": [{"contentType": "image/png"}],
                },
            },
        }
    })

    assert len(captured) == 2
    assert adapter._recipient_number_by_uuid[peer_uuid] == peer_number
    assert captured[-1].reply_to_media_paths == [str(sent_image)]


@pytest.mark.asyncio
async def test_quoted_timestamp_collision_from_another_chat_has_no_local_path(monkeypatch, tmp_path):
    """A user quote cannot retrieve media solely by colliding with a cached timestamp."""
    adapter = _make_signal_adapter(monkeypatch)
    sent_image = tmp_path / "other-chat.png"
    sent_image.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter._remember_sent_attachments(1753650000000, "+15558887777", [str(sent_image)])

    captured = {}

    async def _capture(event):
        captured["event"] = event

    monkeypatch.setattr(adapter, "handle_message", _capture)
    envelope = {
        "envelope": {
            "source": "+15559998888",
            "sourceName": "E X",
            "timestamp": 1753650500000,
            "dataMessage": {
                "message": "what is this?",
                "timestamp": 1753650500000,
                "quote": {
                    "id": 1753650000000,
                    "author": "+15559998888",
                    "attachments": [{"contentType": "image/png", "filename": "other-chat.png"}],
                },
                "attachments": [],
            },
        }
    }

    await adapter._handle_envelope(envelope)

    assert captured["event"].reply_to_media_paths == []


class TestQuotedMediaConversationIdentity:
    """A quote must resolve across Signal's number<->UUID identifier swap."""

    def _adapter(self):
        from gateway.platforms.signal import SignalAdapter
        a = object.__new__(SignalAdapter)
        from collections import OrderedDict
        a._sent_attachment_paths = OrderedDict()
        a._max_sent_attachment_entries = 200
        a._recipient_uuid_by_number = {}
        a._recipient_number_by_uuid = {}
        return a

    def test_quote_resolves_when_reply_arrives_under_the_peer_uuid(self, tmp_path):
        """Sent under E.164, quoted under sourceUuid: same peer, must resolve."""
        f = tmp_path / "chart.png"
        f.write_bytes(b"x")
        a = self._adapter()
        uuid_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        a._remember_recipient_identifiers("+15551234567", uuid_id)
        a._remember_sent_attachments("1700000000000", "+15551234567", [str(f)])

        assert a._resolve_quoted_media_paths("1700000000000", uuid_id) == [str(f)]

    def test_quote_resolves_when_reply_arrives_under_the_peer_number(self, tmp_path):
        """The inverse direction must work too."""
        f = tmp_path / "chart.png"
        f.write_bytes(b"x")
        a = self._adapter()
        uuid_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        a._remember_recipient_identifiers("+15551234567", uuid_id)
        a._remember_sent_attachments("1700000000000", uuid_id, [str(f)])

        assert a._resolve_quoted_media_paths("1700000000000", "+15551234567") == [str(f)]

    def test_unrelated_conversation_still_isolated(self, tmp_path):
        """Widening to a peer alias must not leak across different chats."""
        f = tmp_path / "chart.png"
        f.write_bytes(b"x")
        a = self._adapter()
        a._remember_recipient_identifiers("+15551234567", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        a._remember_sent_attachments("1700000000000", "+15551234567", [str(f)])

        assert a._resolve_quoted_media_paths("1700000000000", "+15559999999") == []
        assert a._resolve_quoted_media_paths("1700000000000", "group:other") == []


class TestUuidOnlyEnvelopePreservesNumberAlias:
    """A UUID-only envelope must not erase a learned number<->UUID alias."""

    def _adapter(self):
        from gateway.platforms.signal import SignalAdapter
        from collections import OrderedDict
        a = object.__new__(SignalAdapter)
        a._sent_attachment_paths = OrderedDict()
        a._max_sent_attachment_entries = 200
        a._recipient_uuid_by_number = {}
        a._recipient_number_by_uuid = {}
        return a

    def test_self_mapping_does_not_clobber_the_real_alias(self, tmp_path):
        """sender == sourceUuid (UUID-only envelope) must be a no-op.

        _handle_envelope passes sender as `number`, which IS the uuid when the
        envelope carries no E.164. Storing uuid->uuid destroyed the alias that
        quote resolution depends on.
        """
        f = tmp_path / "chart.png"
        f.write_bytes(b"x")
        a = self._adapter()
        uid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

        # Learned from an earlier envelope that carried both identifiers.
        a._remember_recipient_identifiers("+15551234567", uid)
        a._remember_sent_attachments("1700000000000", "+15551234567", [str(f)])

        # Now a UUID-only envelope arrives: sender and sourceUuid are the same.
        a._remember_recipient_identifiers(uid, uid)

        assert a._recipient_number_by_uuid[uid] == "+15551234567"
        assert a._resolve_quoted_media_paths("1700000000000", uid) == [str(f)]


# ---------------------------------------------------------------------------
# SSE reconnect escalation (#255)
#
# #211 made _force_reconnect actually close the stream. It left two gaps:
# the request was one-shot (the handle was cleared before the close landed,
# so every later attempt was a silent no-op), and a close that never landed
# had no fallback. Inbound Signal stayed dead while the monitor logged
# "forcing reconnect" every 30s.
# ---------------------------------------------------------------------------
class TestSignalSSEReconnectEscalation:

    @pytest.mark.asyncio
    async def test_repeat_requests_are_not_silent_noops(self, monkeypatch):
        """A pending reconnect must escalate, not re-fire at a cleared handle."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._running = True
        adapter._sse_task = None

        request = httpx.Request("GET", "http://localhost:8080/api/v1/events")
        response = httpx.Response(200, request=request)
        adapter._sse_response = response

        # First request: clears the handle and records the generation.
        adapter._request_reconnect()
        assert adapter._sse_response is None
        assert adapter._reconnect_requested_at_generation == adapter._sse_generation

        restarts = []
        async def fake_restart():
            restarts.append(True)
        monkeypatch.setattr(adapter, "_restart_sse_listener", fake_restart)
        monkeypatch.setattr(
            "gateway.platforms.signal.HEALTH_CHECK_INTERVAL", 0.01
        )
        monkeypatch.setattr(
            "gateway.platforms.signal.HEALTH_CHECK_STALE_THRESHOLD", 0.0
        )
        # Stream never reconnects, so the generation never advances.
        adapter._last_sse_activity = 0.0

        monitor = asyncio.create_task(adapter._health_monitor())
        try:
            for _ in range(200):
                await asyncio.sleep(0.01)
                if restarts:
                    break
        finally:
            adapter._running = False
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

        assert restarts, (
            "health monitor never escalated: a reconnect that never lands "
            "must recreate the listener task, not spin on no-ops"
        )

    @pytest.mark.asyncio
    async def test_listener_reconnect_clears_pending_request(self, monkeypatch):
        """Re-establishing the stream must reset the escalation state."""
        adapter = _make_signal_adapter(monkeypatch)
        monkeypatch.setattr("gateway.platforms.signal.SSE_RETRY_DELAY_INITIAL", 0)

        class OneShotStream(httpx.AsyncByteStream):
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def __aiter__(self):
                self.started.set()
                yield b": keepalive\n\n"
                await self.release.wait()

            async def aclose(self):
                self.release.set()

        class Ctx:
            def __init__(self, response):
                self.response = response
            async def __aenter__(self):
                return self.response
            async def __aexit__(self, *_):
                await self.response.aclose()

        class Client:
            def __init__(self, ctxs):
                self.ctxs = iter(ctxs)
            def stream(self, *_a, **_k):
                return next(self.ctxs)

        request = httpx.Request("GET", "http://localhost:8080/api/v1/events")
        s1, s2 = OneShotStream(), OneShotStream()
        adapter.sse_client = Client([
            Ctx(httpx.Response(200, request=request, stream=s1)),
            Ctx(httpx.Response(200, request=request, stream=s2)),
        ])
        adapter._running = True

        task = asyncio.create_task(adapter._sse_listener())
        try:
            await asyncio.wait_for(s1.started.wait(), timeout=2)
            gen_before = adapter._sse_generation

            adapter._request_reconnect()
            assert adapter._reconnect_requested_at_generation == gen_before

            await asyncio.wait_for(s2.started.wait(), timeout=2)
            # The listener re-established: the pending request is satisfied.
            assert adapter._sse_generation > gen_before
            assert adapter._reconnect_requested_at_generation is None
            assert adapter._stale_checks_since_reconnect == 0
        finally:
            adapter._running = False
            s1.release.set()
            s2.release.set()
            try:
                await asyncio.wait_for(task, timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

    @pytest.mark.asyncio
    async def test_stale_listener_cannot_use_replacement_client(self, monkeypatch):
        """A listener that outlives restart must not subscribe on the new pool."""
        adapter = _make_signal_adapter(monkeypatch)
        adapter._running = True

        class BlockingStream(httpx.AsyncByteStream):
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def __aiter__(self):
                self.started.set()
                await self.release.wait()
                yield b": stale\n\n"

            async def aclose(self):
                self.release.set()

        class Context:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *_):
                await self.response.aclose()

        class Client:
            def __init__(self, context):
                self.context = context
                self.stream_calls = 0

            def stream(self, *_args, **_kwargs):
                self.stream_calls += 1
                return self.context

        request = httpx.Request("GET", "http://localhost:8080/api/v1/events")
        stale_stream = BlockingStream()
        stale_client = Client(
            Context(httpx.Response(200, request=request, stream=stale_stream))
        )
        replacement_client = Client(None)
        adapter.sse_client = stale_client
        adapter._sse_listener_generation = 1
        stale_task = asyncio.create_task(adapter._sse_listener(1))

        try:
            await asyncio.wait_for(stale_stream.started.wait(), timeout=2)
            adapter._sse_listener_generation = 2
            adapter.sse_client = replacement_client
            stale_stream.release.set()
            await asyncio.wait_for(stale_task, timeout=2)
        finally:
            adapter._running = False
            if not stale_task.done():
                stale_task.cancel()
                await stale_task

        assert stale_client.stream_calls == 1
        assert replacement_client.stream_calls == 0

    @pytest.mark.asyncio
    async def test_gap_after_reconnect_is_logged(self, monkeypatch, caplog):
        """signal-cli does not replay: a delivery gap must not pass silently."""
        adapter = _make_signal_adapter(monkeypatch)
        monkeypatch.setattr("gateway.platforms.signal.SSE_RETRY_DELAY_INITIAL", 0)

        class Stream(httpx.AsyncByteStream):
            def __init__(self):
                self.started = asyncio.Event()
                self.release = asyncio.Event()
            async def __aiter__(self):
                self.started.set()
                yield b": keepalive\n\n"
                await self.release.wait()
            async def aclose(self):
                self.release.set()

        class Ctx:
            def __init__(self, response):
                self.response = response
            async def __aenter__(self):
                return self.response
            async def __aexit__(self, *_):
                await self.response.aclose()

        class Client:
            def __init__(self, ctx):
                self.ctx = ctx
            def stream(self, *_a, **_k):
                return self.ctx

        request = httpx.Request("GET", "http://localhost:8080/api/v1/events")
        stream = Stream()
        adapter.sse_client = Client(Ctx(httpx.Response(200, request=request, stream=stream)))
        adapter._running = True
        # Last event was well beyond the stale threshold: a gap happened.
        adapter._last_sse_activity = time.time() - 600

        with caplog.at_level(logging.WARNING, logger="gateway.platforms.signal"):
            task = asyncio.create_task(adapter._sse_listener())
            try:
                await asyncio.wait_for(stream.started.wait(), timeout=2)
            finally:
                adapter._running = False
                stream.release.set()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    task.cancel()

        assert any(
            "were dropped" in r.getMessage() for r in caplog.records
        ), "reconnect after a long gap must warn that messages were lost"
