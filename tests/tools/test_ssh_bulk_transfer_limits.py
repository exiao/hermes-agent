"""Bulk transfer budgets scale with payload instead of a fixed ceiling."""

import pytest

from tools.environments.ssh import (
    _BULK_UPLOAD_MAX_TIMEOUT,
    _BULK_UPLOAD_MIN_TIMEOUT,
    _bulk_upload_timeout,
)


def _write(tmp_path, name, size):
    p = tmp_path / name
    p.write_bytes(b"x" * size)
    return (str(p), f"/root/.hermes/{name}")


def test_small_payload_keeps_the_floor(tmp_path):
    files = [_write(tmp_path, "a.txt", 1024)]
    assert _bulk_upload_timeout(files) == _BULK_UPLOAD_MIN_TIMEOUT


def test_empty_file_list_keeps_the_floor():
    assert _bulk_upload_timeout([]) == _BULK_UPLOAD_MIN_TIMEOUT


def test_large_payload_gets_more_time_than_the_old_fixed_ceiling(tmp_path):
    # The regression: a cold sync is hundreds of MB and died at a flat 120s.
    files = [_write(tmp_path, f"f{i}.bin", 1_000_000) for i in range(300)]
    assert _bulk_upload_timeout(files) > _BULK_UPLOAD_MIN_TIMEOUT


def test_budget_grows_with_payload(tmp_path):
    small = [_write(tmp_path, "small.bin", 10_000_000)]
    large = [_write(tmp_path, "large.bin", 200_000_000)]
    assert _bulk_upload_timeout(large) > _bulk_upload_timeout(small)


def test_budget_is_capped(tmp_path):
    huge = [(str(tmp_path / "missing"), "/root/.hermes/x")]
    # Simulate an enormous payload without writing it to disk.
    files = huge * 10
    assert _bulk_upload_timeout(files) <= _BULK_UPLOAD_MAX_TIMEOUT


def test_unreadable_file_does_not_break_estimation(tmp_path):
    files = [
        _write(tmp_path, "real.txt", 2048),
        (str(tmp_path / "does-not-exist"), "/root/.hermes/nope"),
    ]
    assert _bulk_upload_timeout(files) == _BULK_UPLOAD_MIN_TIMEOUT
