"""Tests: kanban worker spawn never inherits a stray ambient HERMES_HOME.

Regression coverage for the leaked-home bug: ``_default_spawn`` builds the child
env from ``dict(os.environ)`` and then calls ``resolve_profile_env(profile)`` to
point the worker at its profile home. For every NAMED profile that resolves to a
fixed ``<root>/profiles/<name>`` path, so a stray ambient ``HERMES_HOME`` is
harmless. But ``resolve_profile_env("default")`` returns
``get_default_hermes_root()``, which READS ``HERMES_HOME`` from the ambient
environment.

So when the dispatching process has a stray ``HERMES_HOME`` (an e2e test home, a
Docker path, a leftover export), every ``default``-profile worker silently
inherits it: the child resolves its config, memories, skills AND its kanban DB
under that foreign home. Workers then run against an empty board and report
completions the real board never records.

The guard: when the ROOT (non-profile) home is what we're handing the child,
resolve it from the real platform default rather than the ambient env var.
"""

from __future__ import annotations

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


def test_default_profile_worker_ignores_stray_ambient_hermes_home(
    monkeypatch, tmp_path,
):
    """A stray ambient HERMES_HOME must not redefine the default profile's home.

    This is the real-world failure: an e2e test exported
    ``HERMES_HOME=/tmp/e2ehome2`` into the gateway process. Every dispatched
    ``default`` worker then resolved its kanban DB under that throwaway home and
    operated on an empty board.
    """
    from hermes_cli import kanban_db as kb

    stray = tmp_path / "e2ehome-stray"
    stray.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(stray))

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(
        kb, monkeypatch, str(workspace), assignee="default",
    )

    child_home = captured["env"].get("HERMES_HOME", "")
    assert child_home, "worker spawn must always pin an explicit HERMES_HOME"
    assert str(stray) not in child_home, (
        "default-profile worker inherited the stray ambient HERMES_HOME "
        f"({child_home!r}); a leaked test/Docker home must never redefine "
        "which board the worker reads"
    )


def test_named_profile_worker_ignores_bare_scratch_hermes_home(
    monkeypatch, tmp_path,
):
    """A bare scratch home is ignored for named profiles too.

    The leak is not default-only: ``<root>`` itself derives from
    ``HERMES_HOME``, so a stray value relocates named profiles as well.
    """
    from hermes_cli import kanban_db as kb

    stray = tmp_path / "e2ehome-stray"
    stray.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(stray))

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace), assignee="dev")

    child_home = captured["env"].get("HERMES_HOME", "")
    assert str(stray) not in child_home, (
        "named-profile worker resolved under the stray ambient HERMES_HOME "
        f"({child_home!r})"
    )


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
