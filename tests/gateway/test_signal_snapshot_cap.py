"""Regression coverage for bounded Signal snapshot-copy admission."""

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
async def test_expired_snapshot_entries_do_not_admit_more_than_cap_copy_tasks(monkeypatch, tmp_path):
    """Quote expiry resolves waiters without releasing outstanding-copy capacity."""
    import gateway.platforms.signal as signal_module

    adapter = _make_signal_adapter(monkeypatch)
    source = tmp_path / "chart.png"
    source.write_bytes(b"image bytes")
    copy_slots_started = asyncio.Event()
    release_copies = asyncio.Event()
    copies_started = 0

    async def blocked_copy(_copyfile, _source, destination):
        nonlocal copies_started
        copies_started += 1
        if copies_started == 4:
            copy_slots_started.set()
        await release_copies.wait()
        Path(destination).write_bytes(source.read_bytes())

    monkeypatch.setattr(signal_module.asyncio, "to_thread", blocked_copy)
    cap = adapter._max_pending_sent_attachment_snapshots
    copies = [
        asyncio.create_task(adapter._remember_sent_attachments(timestamp, "+15559998888", [str(source)]))
        for timestamp in range(cap)
    ]
    candidate = None
    try:
        await copy_slots_started.wait()
        for timestamp in range(cap):
            adapter._expire_pending_sent_attachment_snapshot(("+15559998888", str(timestamp)))

        candidate = asyncio.create_task(
            adapter._remember_sent_attachments(cap, "+15559998888", [str(source)])
        )
        await asyncio.sleep(0)

        assert len(adapter._pending_sent_attachment_snapshots) == cap
        assert adapter._snapshot_capacity_rejected_total == 1
        assert copies_started == 4
    finally:
        release_copies.set()
        await asyncio.gather(*copies, *([candidate] if candidate else []), return_exceptions=True)
