"""Regression coverage for Signal's outbound snapshot/quoted-reply race."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from gateway.config import PlatformConfig


def _make_signal_adapter(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from gateway.platforms.signal import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": "+15551234567"}
    return SignalAdapter(config)


@pytest.mark.asyncio
async def test_quoted_reply_waits_for_snapshot_copy_before_building_event(monkeypatch, tmp_path):
    """A quote received during the copy must retain the eventual local path."""
    import gateway.platforms.signal as signal_module
    adapter = _make_signal_adapter(monkeypatch)
    adapter._stop_typing_indicator = AsyncMock()
    source = tmp_path / "chart.png"
    source.write_bytes(b"image bytes")
    copy_started = asyncio.Event()
    allow_copy = asyncio.Event()

    async def delayed_copy(_copyfile, _source, destination):
        copy_started.set()
        await allow_copy.wait()
        Path(destination).write_bytes(source.read_bytes())

    monkeypatch.setattr(signal_module.asyncio, "to_thread", delayed_copy)
    adapter._rpc = AsyncMock(return_value={"timestamp": 1753650000000})
    send_task = asyncio.create_task(adapter.send_image_file("+15559998888", str(source)))
    await copy_started.wait()
    captured = {}

    async def capture(event):
        captured["event"] = event

    monkeypatch.setattr(adapter, "handle_message", capture)
    quote_task = asyncio.create_task(adapter._handle_envelope({
        "envelope": {
            "source": "+15559998888", "sourceNumber": "+15559998888",
            "timestamp": 1753650500000,
            "dataMessage": {"message": "what about this?", "quote": {
                "id": 1753650000000, "author": adapter.account,
            }},
        },
    }))
    await asyncio.sleep(0)
    allow_copy.set()
    await send_task
    await quote_task
    assert captured["event"].reply_to_media_paths


@pytest.mark.asyncio
async def test_snapshot_wait_timeout_removes_pending_entry(monkeypatch, tmp_path):
    """A stalled optional copy cannot retain its pending map entry."""
    import gateway.platforms.signal as signal_module

    adapter = _make_signal_adapter(monkeypatch)
    source = tmp_path / "chart.png"
    source.write_bytes(b"image bytes")
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_copy(_copyfile, _source, destination):
        started.set()
        await release.wait()
        Path(destination).write_bytes(source.read_bytes())

    monkeypatch.setattr(signal_module.asyncio, "to_thread", delayed_copy)
    adapter._pending_sent_attachment_snapshot_ttl_seconds = 60
    adapter._max_pending_sent_attachment_snapshots = 1
    task = asyncio.create_task(adapter._remember_sent_attachments(123, "+15559998888", [str(source)]))
    await started.wait()
    original_wait = signal_module.SIGNAL_QUOTE_SNAPSHOT_WAIT_SECONDS
    monkeypatch.setattr(signal_module, "SIGNAL_QUOTE_SNAPSHOT_WAIT_SECONDS", 0.01)
    assert await adapter._await_quoted_media_paths("123", "+15559998888") == []
    assert not adapter._pending_sent_attachment_snapshots
    release.set()
    await task
    monkeypatch.setattr(signal_module, "SIGNAL_QUOTE_SNAPSHOT_WAIT_SECONDS", original_wait)


@pytest.mark.asyncio
async def test_same_timestamp_in_two_chats_keeps_both_attachment_snapshots(monkeypatch, tmp_path):
    """Signal timestamps may collide across chats without cancelling either copy."""
    import gateway.platforms.signal as signal_module

    adapter = _make_signal_adapter(monkeypatch)
    first_source = tmp_path / "first.png"
    second_source = tmp_path / "second.png"
    first_source.write_bytes(b"first image")
    second_source.write_bytes(b"second image")
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_copy(_copyfile, source, destination):
        (first_started if source == first_source else second_started).set()
        await release.wait()
        Path(destination).write_bytes(Path(source).read_bytes())

    monkeypatch.setattr(signal_module.asyncio, "to_thread", delayed_copy)
    adapter._rpc = AsyncMock(return_value={"timestamp": 123})
    first_send = asyncio.create_task(adapter.send_image_file("+15550000001", str(first_source)))
    await first_started.wait()
    second_send = asyncio.create_task(adapter.send_image_file("+15550000002", str(second_source)))
    await second_started.wait()
    release.set()

    results = await asyncio.gather(first_send, second_send, return_exceptions=True)

    assert not any(isinstance(result, BaseException) for result in results)
    assert adapter._resolve_quoted_media_paths("123", "+15550000001")
    assert adapter._resolve_quoted_media_paths("123", "+15550000002")
