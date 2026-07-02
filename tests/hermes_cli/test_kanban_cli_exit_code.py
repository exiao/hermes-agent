"""Regression: `hermes kanban <subcommand>` must propagate its shell exit code.

Root cause (main.py dispatch): `args.func(args)` discarded the handler's return
value, so `cmd_kanban` -> `kanban_command` returning 1 on a handled failure (e.g.
commenting on an unknown task) still exited the process 0. That broke the kanban
card-drop receiver, which maps unknown-card -> 404 only when `proc.returncode != 0`.

These run the REAL CLI as a subprocess against a throwaway HERMES_HOME so the actual
`main()` exit path is exercised (an in-process handler call would bypass the bug).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest


def _run_kanban(home: str, *argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["HERMES_HOME"] = home
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *argv],
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def isolated_home():
    home = tempfile.mkdtemp(prefix="kanban_exit_code_")
    os.makedirs(os.path.join(home, "profiles", "default"), exist_ok=True)
    yield home


def test_comment_unknown_task_exits_nonzero(isolated_home):
    """The repro from the bug report: an unknown task id must exit non-zero."""
    proc = _run_kanban(
        isolated_home, "comment", "--author", "card-drop", "--", "t_nonexistent", "x"
    )
    assert proc.returncode != 0, (
        f"expected non-zero exit for unknown task, got {proc.returncode}\n"
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "unknown task" in proc.stderr.lower()


def test_list_success_exits_zero(isolated_home):
    """A successful subcommand still exits 0 (guards the isinstance fix)."""
    proc = _run_kanban(isolated_home, "list")
    assert proc.returncode == 0, (
        f"expected 0 for successful list, got {proc.returncode}\n"
        f"stderr={proc.stderr!r}"
    )


def test_unknown_board_exits_nonzero(isolated_home):
    """`--board <typo>` on a non-existent board is a handled failure -> non-zero."""
    proc = _run_kanban(isolated_home, "--board", "does-not-exist", "list")
    assert proc.returncode != 0, (
        f"expected non-zero exit for unknown board, got {proc.returncode}\n"
        f"stderr={proc.stderr!r}"
    )
