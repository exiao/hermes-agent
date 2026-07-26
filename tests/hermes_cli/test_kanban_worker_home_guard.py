"""Regression coverage for worker home and shared-board spawn pins.

``HERMES_HOME`` is the worker profile root and can be a deliberate custom path
without config.yaml or a profiles tree. ``HERMES_KANBAN_HOME`` and the per-path
Kanban overrides are independent shared-board locations. Worker spawning must
preserve both scopes without mutating the gateway process environment.
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


def test_default_worker_preserves_configless_custom_home(monkeypatch, tmp_path):
    """A custom root with only state files is still an explicit deployment root."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / "opt-data"
    home.mkdir()
    (home / "kanban.db").touch()
    monkeypatch.setenv("HERMES_HOME", str(home))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="default")

    assert captured["env"]["HERMES_HOME"] == str(home)
    assert captured["env"]["HERMES_KANBAN_DB"] == str(home / "kanban.db")


def test_named_worker_keeps_profile_home_separate_from_board_home(
    monkeypatch, tmp_path,
):
    """A shared board location must not replace the worker profile's home."""
    from hermes_cli import kanban_db as kb

    profile_root = tmp_path / "profile-root"
    (profile_root / "profiles" / "dev").mkdir(parents=True)
    board_home = tmp_path / "shared-board"
    board_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_home))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="dev")

    assert captured["env"]["HERMES_HOME"] == str(profile_root / "profiles" / "dev")
    assert captured["env"]["HERMES_KANBAN_DB"] == str(board_home / "kanban.db")


def test_default_worker_keeps_profile_home_separate_from_board_home(
    monkeypatch, tmp_path,
):
    """The same separation applies to the default profile."""
    from hermes_cli import kanban_db as kb

    profile_home = tmp_path / "profile-home"
    profile_home.mkdir()
    board_home = tmp_path / "shared-board"
    board_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(board_home))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="default")

    assert captured["env"]["HERMES_HOME"] == str(profile_home)
    assert captured["env"]["HERMES_KANBAN_DB"] == str(board_home / "kanban.db")


def test_spawn_preserves_explicit_per_path_board_overrides(monkeypatch, tmp_path):
    """The dispatcher's highest-precedence DB/workspace pins reach the child."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / "profile-home"
    home.mkdir()
    board_db = tmp_path / "external-board" / "kanban.db"
    board_workspaces = tmp_path / "external-board" / "workspaces"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(board_db))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(board_workspaces))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="default")

    assert captured["env"]["HERMES_KANBAN_DB"] == str(board_db)
    assert captured["env"]["HERMES_KANBAN_WORKSPACES_ROOT"] == str(board_workspaces)


def test_spawn_never_mutates_gateway_hermes_home(monkeypatch, tmp_path):
    """Worker derivation cannot race concurrent gateway users of os.environ."""
    from hermes_cli import kanban_db as kb

    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="default")

    assert os.environ["HERMES_HOME"] == str(home)
