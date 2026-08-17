"""Regression tests for cron.lifecycle_guard._read_referenced_script.

Bug (Aug 2026): the guard scans a command for referenced shell scripts and
reads each candidate path. ``os.open`` raises ``ValueError`` (not ``OSError``)
when a path contains an embedded NUL byte, and only ``OSError`` was caught.
The ValueError escaped through ``_contains_unsafe_gateway_action`` and
``contains_gateway_lifecycle_command_or_referenced_script`` all the way out of
``terminal_tool``, aborting an ordinary user command with:

    Failed to execute command: open: embedded null character in path

A path we cannot open is not a referenced shell script, so it must be treated
like a missing file (nothing to scan, not unsafe) rather than crashing the
caller. These tests assert that contract at both the unit and the entrypoint
level, against real files in a temp dir (no mocks).
"""
import os
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cron import lifecycle_guard as lg


def test_nul_in_path_does_not_raise():
    """The exact crash: a NUL byte in the candidate path."""
    text, unsafe = lg._read_referenced_script(Path("foo\x00bar"))
    assert text is None
    assert unsafe is False


def test_missing_file_still_returns_not_unsafe(tmp_path):
    """Pre-existing OSError behavior must be unchanged."""
    text, unsafe = lg._read_referenced_script(tmp_path / "nope.sh")
    assert text is None
    assert unsafe is False


def test_regular_file_is_still_read(tmp_path):
    """The fix must not swallow legitimate reads."""
    script = tmp_path / "real.sh"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    text, unsafe = lg._read_referenced_script(script)
    assert text is not None
    assert "echo hi" in text
    assert unsafe is False


def test_directory_is_ignored(tmp_path):
    """Directories are not script candidates and should be ignored."""
    text, unsafe = lg._read_referenced_script(tmp_path)
    assert text is None
    assert unsafe is False


def test_binary_is_skipped_not_unsafe(tmp_path):
    """NUL bytes *inside* the file mean binary, which is skipped (#76762)."""
    binary = tmp_path / "a.out"
    binary.write_bytes(b"\x7fELF\x00\x00\x00\x00payload")
    text, unsafe = lg._read_referenced_script(binary)
    assert text is None
    assert unsafe is False


def test_entrypoint_survives_nul_bearing_command(tmp_path):
    """End-to-end smoke: the public guard must not raise on a NUL command.

    NOTE: verified to pass both WITH and WITHOUT the fix — the tokenizer does
    not always turn a NUL-bearing token into a candidate script path, so this
    is a smoke test, not the discriminator. ``test_nul_in_path_does_not_raise``
    is the one that actually fails on unpatched code (confirmed: 1 failed,
    6 passed against the pre-fix tree). Kept because the real-world crash was
    observed at this level, via terminal_tool's ``read_remote_script``
    recursion decoding binary content into junk paths.
    """
    command = "python /tmp/x\x00y.py 'a prompt' --json"
    result = lg.contains_gateway_lifecycle_command_or_referenced_script(
        command, cwd=str(tmp_path)
    )
    assert result is False


def test_entrypoint_still_flags_a_real_lifecycle_script(tmp_path):
    """The guard must still DO its job after the fix.

    A negative-only test would pass even if the guard were stubbed to return
    False, so assert the positive case too.
    """
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    result = lg.contains_gateway_lifecycle_command_or_referenced_script(
        f"sh {script}", cwd=str(tmp_path)
    )
    assert result is True
