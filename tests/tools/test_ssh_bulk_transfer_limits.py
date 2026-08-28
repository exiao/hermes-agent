"""Bulk transfer budgets scale with payload instead of a fixed ceiling."""

import os
import subprocess

import pytest

from tools.environments.ssh import (
    _BULK_UPLOAD_MAX_TIMEOUT,
    _BULK_UPLOAD_MIN_TIMEOUT,
    _bulk_upload_timeout,
    _tar_stderr_is_only_concurrent_change,
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


class TestTarExitOneClassification:
    """Cases that a substring-anywhere match would get wrong."""

    def test_warning_alongside_a_real_error_is_rejected(self):
        # The dangerous case: a benign warning must not launder a genuine read
        # failure in the same stderr into an accepted, truncated archive.
        assert not _tar_stderr_is_only_concurrent_change(
            "tar: cache/x: file changed as we read it\n"
            "tar: secret: Cannot open: Permission denied"
        )

    def test_summary_line_does_not_decide_the_outcome(self):
        assert _tar_stderr_is_only_concurrent_change(
            "tar: logs/a.log: File removed before we read it\n"
            "tar: Exiting with failure status due to previous errors"
        )

    def test_empty_stderr_with_exit_one_is_not_benign(self):
        assert not _tar_stderr_is_only_concurrent_change("")

    def test_real_tar_missing_member_is_classified_as_an_error(self, tmp_path):
        """Run the actual tar binary rather than a mocked subprocess."""
        (tmp_path / "d").mkdir()
        (tmp_path / "d" / "a.txt").write_text("hi")
        result = subprocess.run(
            ["tar", "cf", os.devnull, "-C", str(tmp_path), "d", "nosuchfile"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1, "expected tar to exit 1 on a missing member"
        assert not _tar_stderr_is_only_concurrent_change(result.stderr)

    def test_real_tar_clean_run_exits_zero(self, tmp_path):
        (tmp_path / "d").mkdir()
        (tmp_path / "d" / "a.txt").write_text("hi")
        result = subprocess.run(
            ["tar", "cf", os.devnull, "-C", str(tmp_path), "d"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
