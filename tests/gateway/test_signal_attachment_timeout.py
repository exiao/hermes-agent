"""Single-attachment Signal sends must use a size-aware RPC timeout."""

from unittest.mock import AsyncMock

import pytest

from gateway.platforms.signal_rate_limit import _signal_send_timeout


def test_text_only_send_keeps_a_bounded_deadline():
    """A send with no attachments is daemon-local, so it stays deadlined."""
    assert _signal_send_timeout(0) == 30.0


def test_attachment_send_does_not_deadline_the_upload_leg():
    """No read deadline on attachment sends, at any size.

    The RPC body carries paths, not bytes: signal-cli uploads to Signal's
    servers after the local POST returns. A read deadline here is a hidden
    minimum-bandwidth assumption, which is exactly what produced the phantom
    failure this PR fixes.
    """
    timeout = _signal_send_timeout(1)
    assert timeout.read is None
    # Connect/write/pool stay bounded so an unreachable daemon still fails fast.
    assert timeout.connect is not None
    assert timeout.write is not None


def test_attachment_timeout_is_size_independent():
    """A 100 MiB attachment and a 1-byte one get the same (undeadlined) read."""
    small = _signal_send_timeout(1, 1)
    huge = _signal_send_timeout(1, 100 * 1024 * 1024)
    assert small.read is huge.read is None
    assert small.connect == huge.connect


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
async def test_send_document_does_not_deadline_large_upload(tmp_path):
    """The accepted 100 MiB document must not carry a read deadline."""
    from gateway.platforms.signal import SIGNAL_MAX_ATTACHMENT_SIZE, SignalAdapter

    attachment = tmp_path / "slow-upload.bin"
    with attachment.open("wb") as file:
        file.truncate(SIGNAL_MAX_ATTACHMENT_SIZE)
    adapter = object.__new__(SignalAdapter)
    adapter.account = "+15551234567"
    adapter._stop_typing_indicator = AsyncMock()
    adapter._rpc = AsyncMock(return_value=None)

    await adapter.send_document("group:group-id", str(attachment))

    assert adapter._rpc.await_args.kwargs["timeout"].read is None


@pytest.mark.asyncio
async def test_send_multiple_images_passes_batch_byte_total(tmp_path):
    """Large image batches must not fall back to the attachment-count floor."""
    from gateway.platforms.signal import SIGNAL_MAX_ATTACHMENT_SIZE, SignalAdapter

    images = []
    for index in range(2):
        image = tmp_path / f"slow-upload-{index}.png"
        with image.open("wb") as file:
            file.truncate(SIGNAL_MAX_ATTACHMENT_SIZE)
        images.append((image.as_uri(), ""))

    adapter = object.__new__(SignalAdapter)
    adapter.account = "+15551234567"
    adapter._stop_typing_indicator = AsyncMock()
    adapter._rpc = AsyncMock(return_value=None)

    await adapter.send_multiple_images("group:group-id", images)

    assert adapter._rpc.await_args.kwargs["timeout"] == _signal_send_timeout(
        2, 2 * SIGNAL_MAX_ATTACHMENT_SIZE
    )


@pytest.mark.asyncio
async def test_send_message_tool_signal_path_does_not_deadline_upload(tmp_path, monkeypatch):
    """The send_message tool's Signal path must match the adapter's policy.

    This is the path an agent MEDIA: reply actually travels, and it was still
    calling the helper with attachment count only.
    """
    from tools.send_message_tool import _send_signal

    attachment = tmp_path / "big.bin"
    with attachment.open("wb") as file:
        file.truncate(64 * 1024 * 1024)

    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"result": {"timestamp": 1}}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, *args, **kwargs):
            return _FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    await _send_signal(
        {"http_url": "http://127.0.0.1:8080", "account": "+15551234567"},
        "group:group-id",
        "here you go",
        media_files=[(str(attachment), False)],
    )

    assert captured["timeout"].read is None
