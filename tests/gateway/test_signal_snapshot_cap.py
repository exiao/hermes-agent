"""Regression coverage for the Signal snapshot copy-task capacity bound."""

import asyncio
from pathlib import Path

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
async def test_expired_snapshot_entries_do_not_admit_more_than_cap_copy_tasks(
    monkeypatch, tmp_path
):
    """TTL expiry must not hide copy tasks that are still queued or running."""
    import gateway.platforms.signal as signal_module

    adapter = _make_signal_adapter(monkeypatch)
    cap = adapter._max_pending_sent_attachment_snapshots
    source = tmp_path / "attachment.bin"
    source.write_bytes(b"attachment bytes")
    release_copies = asyncio.Event()
    outstanding_copy_tasks = set()
    all_initial_tasks_started = asyncio.Event()
    original_copy = adapter._copy_sent_attachments

    async def track_copy_task(key, paths, entry):
        outstanding_copy_tasks.add(key)
        if len(outstanding_copy_tasks) == cap:
            all_initial_tasks_started.set()
        try:
            await original_copy(key, paths, entry)
        finally:
            outstanding_copy_tasks.discard(key)

    async def blocked_to_thread(copyfile, source_path, destination):
        await release_copies.wait()
        copyfile(source_path, destination)

    monkeypatch.setattr(adapter, "_copy_sent_attachments", track_copy_task)
    monkeypatch.setattr(signal_module.asyncio, "to_thread", blocked_to_thread)

    first_batch = [
        asyncio.create_task(
            adapter._remember_sent_attachments(
                timestamp, "+15559998888", [str(source)]
            )
        )
        for timestamp in range(cap)
    ]
    await all_initial_tasks_started.wait()
    assert len(adapter._pending_sent_attachment_snapshots) == cap

    try:
        # This is the same callback scheduled by call_later for TTL expiry. Call
        # it directly so the test controls the expiry without wall-clock sleeps.
        for key in tuple(adapter._pending_sent_attachment_snapshots):
            adapter._expire_pending_sent_attachment_snapshot(key)
        assert not adapter._pending_sent_attachment_snapshots

        extra_task = asyncio.create_task(
            adapter._remember_sent_attachments(cap, "+15559998888", [str(source)])
        )
        while len(outstanding_copy_tasks) < cap + 1:
            await asyncio.sleep(0)

        observed = len(outstanding_copy_tasks)
        assert observed <= cap, (
            f"observed {observed} outstanding Signal snapshot copy tasks after "
            f"TTL expiry, exceeding cap {cap}; "
            f"pending map reports {len(adapter._pending_sent_attachment_snapshots)}"
        )
    finally:
        release_copies.set()
        await asyncio.gather(*first_batch, extra_task, return_exceptions=True)

    assert Path(source).exists()
