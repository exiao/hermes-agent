"""Deterministic concurrency coverage for Signal quoted attachment snapshots."""

import asyncio
from pathlib import Path

import pytest


CHAT_ID = "+15559998888"
QUOTE_TIMESTAMP = 1753650000000
INBOUND_TIMESTAMP = 1753650500000


def _make_signal_adapter(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from gateway.config import PlatformConfig
    from gateway.platforms.signal import SignalAdapter

    config = PlatformConfig()
    config.enabled = True
    config.extra = {
        "http_url": "http://localhost:8080",
        "account": "+15551234567",
    }
    return SignalAdapter(config)


def _quote_envelope():
    return {
        "envelope": {
            "source": CHAT_ID,
            "sourceName": "E X",
            "sourceNumber": CHAT_ID,
            "timestamp": INBOUND_TIMESTAMP,
            "dataMessage": {
                "message": "what about this image?",
                "timestamp": INBOUND_TIMESTAMP,
                "quote": {
                    "id": QUOTE_TIMESTAMP,
                    "author": "+15551234567",
                    "text": None,
                    "attachments": [
                        {
                            "contentType": "image/png",
                            "filename": "chart.png",
                        }
                    ],
                },
                "attachments": [],
            },
        }
    }


async def _stub_send_rpc(method, params, **_kwargs):
    if method == "send":
        return {"timestamp": QUOTE_TIMESTAMP}
    return {}


@pytest.mark.asyncio
async def test_quoted_reply_waits_for_snapshot_copy_before_building_event(
    monkeypatch, tmp_path
):
    """An inbound quote must not freeze the event before the worker copy is ready."""
    import gateway.platforms.signal as signal_module

    adapter = _make_signal_adapter(monkeypatch)
    source = tmp_path / "chart.png"
    source.write_bytes(b"original chart bytes")
    copy_started = asyncio.Event()
    release_copy = asyncio.Event()

    async def delayed_copy(copyfile, source_path, snapshot_path):
        copy_started.set()
        await release_copy.wait()
        return copyfile(source_path, snapshot_path)

    monkeypatch.setattr(signal_module.asyncio, "to_thread", delayed_copy)
    monkeypatch.setattr(adapter, "_rpc", _stub_send_rpc)
    captured = {}

    async def capture_event(event):
        captured["event"] = event

    monkeypatch.setattr(adapter, "handle_message", capture_event)
    send_task = asyncio.create_task(adapter.send_image_file(CHAT_ID, str(source)))
    try:
        await asyncio.wait_for(copy_started.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        send_task.cancel()
        await asyncio.gather(send_task, return_exceptions=True)
        pytest.fail("send path never scheduled the attachment snapshot copy")

    inbound_task = asyncio.create_task(adapter._handle_envelope(_quote_envelope()))
    try:
        await asyncio.sleep(0)
        assert not inbound_task.done(), (
            "quoted event was built before the in-flight snapshot copy completed"
        )
        release_copy.set()
        await inbound_task
        assert (await send_task).success is True
        event = captured["event"]
        assert len(event.reply_to_media_paths) == 1
        assert Path(event.reply_to_media_paths[0]).read_bytes() == source.read_bytes()
    finally:
        release_copy.set()
        await asyncio.gather(send_task, inbound_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_quoted_reply_resolves_snapshot_after_completed_send(monkeypatch, tmp_path):
    """The same real send and inbound paths preserve the already-ready happy path."""
    adapter = _make_signal_adapter(monkeypatch)
    source = tmp_path / "chart.png"
    source.write_bytes(b"original chart bytes")
    monkeypatch.setattr(adapter, "_rpc", _stub_send_rpc)
    captured = {}

    async def capture_event(event):
        captured["event"] = event

    monkeypatch.setattr(adapter, "handle_message", capture_event)

    result = await adapter.send_image_file(CHAT_ID, str(source))
    assert result.success is True
    await adapter._handle_envelope(_quote_envelope())

    event = captured["event"]
    assert event.reply_to_media_summary == "an image (image/png, chart.png)"
    assert len(event.reply_to_media_paths) == 1
    assert Path(event.reply_to_media_paths[0]).read_bytes() == source.read_bytes()
