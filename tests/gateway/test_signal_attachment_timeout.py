"""Single-attachment Signal sends must use the attachment-scaled RPC timeout.

Regression test for the phantom "RPC send file failed" bug: send_document /
send_image_file / send_voice / send_video all route through _send_attachment,
which called self._rpc("send", params) with no timeout and therefore inherited
the 30s httpx client default. signal-cli uploads the attachment serially inside
the call, so a document upload that takes longer than 30s was aborted client
side and logged as a failure even though signal-cli completed the send.

The batched-image path (send_multiple_images) already used
_signal_send_timeout(n); this asserts the single-attachment paths do too.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.signal_rate_limit import _signal_send_timeout


def test_signal_send_timeout_floor_beats_httpx_default():
    """A single attachment must get more than the 30s httpx default."""
    assert _signal_send_timeout(1) >= 60.0
    assert _signal_send_timeout(1) > 30.0


@pytest.mark.asyncio
async def test_send_document_passes_scaled_timeout(tmp_path):
    """Documents must not fall back to the client default timeout."""
    from gateway.platforms.signal import SignalAdapter

    attachment = tmp_path / "slow-upload.txt"
    attachment.write_text("payload")
    adapter = object.__new__(SignalAdapter)
    adapter.account = "+15551234567"
    adapter._stop_typing_indicator = AsyncMock()
    adapter._rpc = AsyncMock(return_value=None)

    await adapter.send_document("group:group-id", str(attachment))

    assert adapter._rpc.await_args.kwargs["timeout"] == _signal_send_timeout(1)


@pytest.mark.asyncio
async def test_send_image_passes_scaled_timeout(tmp_path):
    """file:// images must use the attachment-scaled RPC timeout too."""
    from gateway.platforms.signal import SignalAdapter

    image = tmp_path / "slow-upload.png"
    image.write_bytes(b"image")
    adapter = object.__new__(SignalAdapter)
    adapter.account = "+15551234567"
    adapter._stop_typing_indicator = AsyncMock()
    adapter._rpc = AsyncMock(return_value=None)

    await adapter.send_image("group:group-id", image.as_uri())

    assert adapter._rpc.await_args.kwargs["timeout"] == _signal_send_timeout(1)
