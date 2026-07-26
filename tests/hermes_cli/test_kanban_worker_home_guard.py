"""Tests: kanban worker spawn preserves valid homes and pins board context.

A child environment begins as ``dict(os.environ)``, so a nested dispatcher can
carry stale kanban-path pins. The dispatcher must derive fresh child pins from
the accepted worker home. An explicit ``HERMES_HOME`` remains valid even for a
fresh Docker/custom root containing only ``.env``, ``kanban.db``, or other state.

An explicit ``HERMES_KANBAN_HOME`` is the stronger task-dispatch context. It
wins over an ambient home inherited from another gateway request or test without
mutating the gateway process's ``os.environ``.
"""

from __future__ import annotations

import os
import subprocess


def _make_task(kb, *, assignee: str = "w"):
    return kb.Task(
        id="t_home",
        title="home guard",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=1,
    )


def _capture_spawn_env(kb, monkeypatch, workspace: str, *, assignee: str) -> dict:
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])

    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(_make_task(kb, assignee=assignee), workspace)
    return captured


def test_default_profile_worker_uses_explicit_board_home_over_ambient_home(
    monkeypatch, tmp_path,
):
    """An explicit board root must beat a stray ambient HERMES_HOME."""
    from hermes_cli import kanban_db as kb

    stray = tmp_path / "e2ehome-stray"
    stray.mkdir()
    board_home = tmp_path / "dispatcher-home"
    board_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(stray))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_home))

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(
        kb, monkeypatch, str(workspace), assignee="default",
    )

    assert captured["env"]["HERMES_HOME"] == str(board_home)
    assert captured["env"]["HERMES_KANBAN_DB"] == str(board_home / "kanban.db")
    assert os.environ["HERMES_HOME"] == str(stray)

def test_named_profile_worker_uses_explicit_board_home_over_ambient_home(
    monkeypatch, tmp_path,
):
    """An explicit board root also anchors a named-profile worker."""
    from hermes_cli import kanban_db as kb

    stray = tmp_path / "e2ehome-stray"
    stray.mkdir()
    board_home = tmp_path / "dispatcher-home"
    (board_home / "profiles" / "dev").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(stray))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_home))

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="dev")

    assert captured["env"]["HERMES_HOME"] == str(board_home / "profiles" / "dev")

def test_unprovisioned_profile_keeps_explicit_board_home(monkeypatch, tmp_path):
    """An unknown profile retains the board root for consistent child context."""
    from hermes_cli import kanban_db as kb

    stray = tmp_path / "e2ehome-stray"
    stray.mkdir()
    board_home = tmp_path / "dispatcher-home"
    board_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(stray))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_home))

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(
        kb, monkeypatch, str(workspace), assignee="no-such-profile-xyz",
    )

    assert captured["env"]["HERMES_HOME"] == str(board_home)

def test_relocated_home_with_config_is_still_honored(monkeypatch, tmp_path):
    """A REAL relocated home (Docker/custom) must keep working.

    The guard must only reject bare scratch dirs. A home carrying a
    ``config.yaml`` is a deliberate deployment choice, not a leak, and
    dropping it would break Docker and custom installs.
    """
    from hermes_cli import kanban_db as kb

    real_home = tmp_path / "opt-data"
    (real_home / "profiles" / "dev").mkdir(parents=True)
    (real_home / "config.yaml").write_text("agent:\n  interface: tui\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(real_home))

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="dev")

    assert captured["env"]["HERMES_HOME"] == str(real_home / "profiles" / "dev"), (
        "a legitimate relocated home must still resolve its profiles"
    )


def test_relocated_configless_home_is_still_honored(monkeypatch, tmp_path):
    """A custom root with only state files is still an explicit deployment root."""
    from hermes_cli import kanban_db as kb

    real_home = tmp_path / "opt-data"
    real_home.mkdir()
    (real_home / "kanban.db").touch()
    monkeypatch.setenv("HERMES_HOME", str(real_home))

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="default")

    assert captured["env"]["HERMES_HOME"] == str(real_home)
    assert captured["env"]["HERMES_KANBAN_DB"] == str(real_home / "kanban.db")


def test_spawn_replaces_inherited_kanban_path_pins(monkeypatch, tmp_path):
    """Stale child pins cannot redirect a new worker away from its home."""
    from hermes_cli import kanban_db as kb

    real_home = tmp_path / "real-home"
    real_home.mkdir()
    foreign_db = tmp_path / "foreign" / "kanban.db"
    foreign_workspaces = tmp_path / "foreign" / "workspaces"
    monkeypatch.setenv("HERMES_HOME", str(real_home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(foreign_db))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(foreign_workspaces))

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="default")

    assert captured["env"]["HERMES_KANBAN_DB"] == str(real_home / "kanban.db")
    assert captured["env"]["HERMES_KANBAN_WORKSPACES_ROOT"] == str(
        real_home / "kanban" / "workspaces"
    )
