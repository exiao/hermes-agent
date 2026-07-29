"""Single-attachment Signal sends must use a size-aware RPC timeout."""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.signal_rate_limit import _signal_send_timeout


def test_signal_send_timeout_scales_with_attachment_size():
    """A 100 MiB attachment gets more time than a small single attachment."""
    assert _signal_send_timeout(1) >= 60.0
    assert _signal_send_timeout(1) > 30.0
    assert _signal_send_timeout(1, 100 * 1024 * 1024) > _signal_send_timeout(1)


@pytest.mark.asyncio
async def test_send_document_passes_size_aware_timeout(tmp_path):
    """Documents must not fall back to the client default timeout."""
    from gateway.platforms.signal import SignalAdapter

    attachment = tmp_path / "slow-upload.txt"
    attachment.write_text("payload")
    adapter = object.__new__(SignalAdapter)
    adapter.account = "+15551234567"
    adapter._stop_typing_indicator = AsyncMock()
    adapter._rpc = AsyncMock(return_value=None)

    await adapter.send_document("group:group-id", str(attachment))

    assert adapter._rpc.await_args.kwargs["timeout"] == _signal_send_timeout(1, attachment.stat().st_size)


@pytest.mark.asyncio
async def test_send_image_passes_size_aware_timeout(tmp_path):
    """file:// images must use the attachment-scaled RPC timeout too."""
    from gateway.platforms.signal import SignalAdapter

    image = tmp_path / "slow-upload.png"
    image.write_bytes(b"image")
    adapter = object.__new__(SignalAdapter)
    adapter.account = "+15551234567"
    adapter._stop_typing_indicator = AsyncMock()
    adapter._rpc = AsyncMock(return_value=None)

    await adapter.send_image("group:group-id", image.as_uri())

    assert adapter._rpc.await_args.kwargs["timeout"] == _signal_send_timeout(1, image.stat().st_size)


@pytest.mark.asyncio
async def test_send_document_scales_timeout_with_large_attachment(tmp_path):
    """The accepted 100 MiB document must not receive only the 60s floor."""
    from gateway.platforms.signal import SIGNAL_MAX_ATTACHMENT_SIZE, SignalAdapter

    attachment = tmp_path / "slow-upload.bin"
    with attachment.open("wb") as file:
        file.truncate(SIGNAL_MAX_ATTACHMENT_SIZE)
    adapter = object.__new__(SignalAdapter)
    adapter.account = "+15551234567"
    adapter._stop_typing_indicator = AsyncMock()
    adapter._rpc = AsyncMock(return_value=None)

    await adapter.send_document("group:group-id", str(attachment))

    assert adapter._rpc.await_args.kwargs["timeout"] > _signal_send_timeout(1)
