"""Regression test: queued follow-up first-response delivers MEDIA attachments.

Bug: when a `/queue` (alias `/q`) turn completes and the gateway sends the
"first response" before draining the queued follow-up, it used a raw
``adapter.send(text)`` that skips MEDIA:/image extraction. A first response
carrying a ``MEDIA:/path`` tag (e.g. from send_file) had the tag delivered as
literal text and the attachment was dropped.

The fix routes that delivery through ``_send_response_with_media``, which
mirrors the background-task / /btw paths: extract media + images, send the
cleaned text, then dispatch each attachment by type.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.run import GatewayRunner
from gateway.platforms.base import (
    BasePlatformAdapter,
    PlatformConfig,
    Platform,
)


class _RecordingAdapter(BasePlatformAdapter):
    """Adapter that records what it was asked to send."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent_text = []
        self.sent_image_files = []
        self.sent_documents = []
        self.sent_videos = []
        self.sent_voices = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        from gateway.platforms.base import SendResult
        self.sent_text.append(content)
        return SendResult(success=True, message_id="msg-1")

    async def send_image_file(self, chat_id, image_path, caption=None, reply_to=None, metadata=None, **kwargs):
        from gateway.platforms.base import SendResult
        self.sent_image_files.append(image_path)
        return SendResult(success=True, message_id="img-1")

    async def send_document(self, chat_id, file_path, caption=None, file_name=None, reply_to=None, metadata=None, **kwargs):
        from gateway.platforms.base import SendResult
        self.sent_documents.append(file_path)
        return SendResult(success=True, message_id="doc-1")

    async def send_video(self, chat_id, video_path, caption=None, reply_to=None, metadata=None, **kwargs):
        from gateway.platforms.base import SendResult
        self.sent_videos.append(video_path)
        return SendResult(success=True, message_id="vid-1")

    async def send_voice(self, chat_id, audio_path, caption=None, reply_to=None, metadata=None, **kwargs):
        from gateway.platforms.base import SendResult
        self.sent_voices.append(audio_path)
        return SendResult(success=True, message_id="voi-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# _send_response_with_media does not touch instance state, so a bare stub
# stands in for `self` and keeps the test off GatewayRunner's heavy __init__.
_send = GatewayRunner._send_response_with_media


def _source():
    return SimpleNamespace(chat_id="123", platform=Platform.TELEGRAM)


def test_image_media_tag_is_delivered_as_attachment(tmp_path):
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    adapter = _RecordingAdapter()
    response = f"Here you go.\n\nMEDIA:{img}"

    _run(_send(SimpleNamespace(), adapter, _source(), response))

    assert adapter.sent_image_files == [str(img)], "image attachment must be delivered"
    # The MEDIA: tag must not survive into the text body.
    joined = "\n".join(adapter.sent_text)
    assert "MEDIA:" not in joined
    assert "Here you go." in joined


def test_document_media_tag_routes_to_send_document(tmp_path):
    doc = tmp_path / "report.pdf"
    doc.write_bytes(b"%PDF-1.4\n" + b"0" * 64)
    adapter = _RecordingAdapter()

    _run(_send(SimpleNamespace(), adapter, _source(), f"Report attached. MEDIA:{doc}"))

    assert adapter.sent_documents == [str(doc)]
    assert adapter.sent_image_files == []


def test_text_only_response_sends_no_attachments():
    adapter = _RecordingAdapter()

    _run(_send(SimpleNamespace(), adapter, _source(), "just text, no media"))

    assert adapter.sent_text == ["just text, no media"]
    assert adapter.sent_image_files == []
    assert adapter.sent_documents == []


def test_media_only_response_sends_attachment_without_empty_text(tmp_path):
    img = tmp_path / "only.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    adapter = _RecordingAdapter()

    _run(_send(SimpleNamespace(), adapter, _source(), f"MEDIA:{img}"))

    assert adapter.sent_image_files == [str(img)]
    # No empty/whitespace-only text message should be sent.
    assert all(t.strip() for t in adapter.sent_text)
