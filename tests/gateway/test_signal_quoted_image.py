"""Quoting an image on Signal hands the agent the image itself (vision), not just its name."""

import pytest

from tests.gateway.test_signal import _make_signal_adapter


class TestQuotedImageIsVisible:
    """Quoting an image must hand the agent the image itself, so vision sees it."""

    @staticmethod
    def _envelope(quote):
        return {"envelope": {"sourceNumber": "+15559876543", "sourceUuid": "u-1", "sourceName": "Eric",
                             "timestamp": 1792700001000,
                             "dataMessage": {"message": "what is this", "quote": quote}}}

    @staticmethod
    async def _capture(adapter, envelope):
        seen = []

        async def handle(event):
            seen.append(event)

        adapter.handle_message = handle
        await adapter._handle_envelope(envelope)
        assert len(seen) == 1
        return seen[0]

    @pytest.mark.asyncio
    async def test_a_quoted_image_we_sent_is_attached(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        img = tmp_path / "chart.png"
        img.write_bytes(b"png")
        adapter._remember_sent_attachments(1792700000000, "+15559876543", [str(img)])
        event = await self._capture(adapter, self._envelope({
            "id": 1792700000000, "author": "+15551234567",
            "attachments": [{"contentType": "image/png", "filename": "chart.png"}]}))
        assert event.media_urls == [str(img)]
        assert event.media_types == ["image/png"]
        assert event.message_type.value == "text"  # a text reply, not a photo burst

    @pytest.mark.asyncio
    async def test_a_quoted_photo_from_the_user_uses_its_thumbnail(self, monkeypatch, tmp_path):
        adapter = _make_signal_adapter(monkeypatch)
        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"jpg")
        fetched = []

        async def fetch(att_id, filename=""):
            fetched.append(att_id)
            return str(thumb), ".jpg"

        adapter._fetch_attachment = fetch
        event = await self._capture(adapter, self._envelope({
            "id": 1792600000000, "author": "+15559876543",
            "attachments": [{"contentType": "image/jpeg", "thumbnail": {"id": "thumb-1", "contentType": "image/jpeg"}}]}))
        assert fetched == ["thumb-1"]
        assert event.media_urls == [str(thumb)]
        assert event.reply_to_media_paths == [str(thumb)]
        assert event.message_type.value == "text"

    @pytest.mark.asyncio
    async def test_a_failed_thumbnail_still_delivers_the_text(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)

        async def fetch(att_id, filename=""):
            raise RuntimeError("gone")

        adapter._fetch_attachment = fetch
        event = await self._capture(adapter, self._envelope({
            "id": 1792600000000, "attachments": [{"contentType": "image/jpeg", "thumbnail": {"id": "t"}}]}))
        assert event.media_urls == []
        assert event.reply_to_media_summary == "an image (image/jpeg)"

    @pytest.mark.asyncio
    async def test_an_oversized_thumbnail_is_not_fetched(self, monkeypatch):
        adapter = _make_signal_adapter(monkeypatch)
        fetched = []

        async def fetch(att_id, filename=""):
            fetched.append(att_id)
            return None, ""

        adapter._fetch_attachment = fetch
        event = await self._capture(adapter, self._envelope({
            "id": 1792600000000,
            "attachments": [{"contentType": "image/jpeg", "thumbnail": {"id": "t", "size": 200 * 1024 * 1024}}]}))
        assert fetched == []
        assert event.media_urls == []
