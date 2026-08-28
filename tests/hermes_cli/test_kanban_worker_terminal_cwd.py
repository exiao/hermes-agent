"""Tests: kanban worker spawn pins the right cwd for its terminal backend.

Regression coverage for #34619 and #41312. ``_default_spawn`` launched the
worker with ``cwd=workspace`` and ``TERMINAL_CWD`` unset, so workers inherited
the dispatching gateway's cwd: relative writes landed in the gateway user's
home (#41312) and the wrong profile's ``AGENTS.md`` loaded (#34619). A HOST
backend (local, or docker bind-mounting the workspace) must be pinned to the
task workspace; a REMOTE backend must never receive a host-only path.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ABSENT = object()  # key must not be in the child env
WS = object()      # value must equal the task workspace
NEUTRAL = object()  # empty dir that is neither the workspace nor the dispatcher cwd


def _make_task(kb):
    return kb.Task(
        id="t_cwd", title="cwd pin", body=None, assignee="w", status="running",
        priority=0, created_by="test", created_at=1, started_at=None,
        completed_at=None, workspace_kind="dir", workspace_path=None,
        claim_lock="lock", claim_expires=None, tenant=None, current_run_id=1,
    )


def _spawn(kb, monkeypatch, workspace: str) -> dict:
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    captured: dict = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["env"] = dict(kwargs.get("env") or {})
        captured["cwd"] = kwargs.get("cwd")
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(_make_task(kb), workspace)
    return captured


def _setup(tmp_path, monkeypatch, profile_terminal: str, root_terminal: str, env: dict):
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n" + profile_terminal, encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\n" + root_terminal, encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    for k, v in env.items():
        monkeypatch.setenv(k, v)


MODAL = "terminal:\n  backend: modal\n"
SSH = "terminal:\n  backend: ssh\n"
HOST_CWD = {"TERMINAL_CWD": "/host/gateway"}

# (id, profile terminal cfg, root terminal cfg, env, expected TERMINAL_CWD,
#  expected HERMES_KANBAN_WORKSPACE, expected spawn cwd)
CASES = [
    ("local_default", "", "", {}, WS, WS, WS),
    ("modal_profile", MODAL, "terminal:\n  backend: local\n", HOST_CWD, ABSENT, ABSENT, NEUTRAL),
    ("docker_mount_profile",
     "terminal:\n  backend: docker\n  docker_mount_cwd_to_workspace: true\n",
     "terminal:\n  backend: local\n", HOST_CWD, WS, WS, WS),
    ("env_only_modal", "", "", {"TERMINAL_ENV": "modal"}, None, ABSENT, NEUTRAL),
    ("stale_docker_flag_on_modal",
     MODAL + "  docker_mount_cwd_to_workspace: true\n", "", {}, None, ABSENT, NEUTRAL),
    ("env_docker_with_mount", "", "",
     {"TERMINAL_ENV": "docker", "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE": "true"},
     WS, WS, WS),
    ("ssh_profile", SSH, "", HOST_CWD, ABSENT, ABSENT, None),
    ("ssh_profile_explicit_cwd", SSH + "  cwd: /srv/work\n", "", HOST_CWD,
     "/srv/work", None, None),
    ("env_only_ssh_keeps_remote_cwd", "", "",
     {"TERMINAL_ENV": "ssh", "TERMINAL_CWD": "/srv/work"}, "/srv/work", ABSENT, NEUTRAL),
    ("remote_drops_inherited_workspace_var", MODAL, "",
     {"HERMES_KANBAN_WORKSPACE": "/host/parent-workspace"}, None, ABSENT, NEUTRAL),
]


@pytest.mark.parametrize(
    "profile_cfg,root_cfg,env,want_cwd,want_ws,want_spawn",
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_worker_spawn_cwd(
    monkeypatch, tmp_path, profile_cfg, root_cfg, env, want_cwd, want_ws, want_spawn
):
    _setup(tmp_path, monkeypatch, profile_cfg, root_cfg, env)
    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    got = _spawn(kb, monkeypatch, str(workspace))

    for key, want in (("TERMINAL_CWD", want_cwd), ("HERMES_KANBAN_WORKSPACE", want_ws)):
        if want is ABSENT:
            assert key not in got["env"], f"{key} leaked to a remote worker"
        elif want is WS:
            assert got["env"][key] == str(workspace)
        elif want is not None:
            assert got["env"][key] == want

    if want_spawn is WS:
        assert got["cwd"] == str(workspace)
    elif want_spawn is NEUTRAL:
        # cwd=None or the workspace would make the worker resolve relative
        # paths and AGENTS.md against a path absent from the sandbox.
        assert got["cwd"] not in (None, os.getcwd(), str(workspace))
        assert not any(Path(got["cwd"]).iterdir())


def test_terminal_cwd_not_pinned_for_nonexistent_workspace(monkeypatch, tmp_path):
    """A non-directory workspace must NOT clobber the inherited TERMINAL_CWD.

    file_tools rejects relative / sentinel TERMINAL_CWD values, so writing a
    nonexistent path would be worse than leaving the inherited one.
    """
    _setup(tmp_path, monkeypatch, "", "", {"TERMINAL_CWD": "/pre/existing/anchor"})
    from hermes_cli import kanban_db as kb

    got = _spawn(kb, monkeypatch, str(tmp_path / "does-not-exist"))

    assert got["env"]["TERMINAL_CWD"] == "/pre/existing/anchor"


# Config-derived TERMINAL_* the dispatcher's bridge writes into os.environ when
# the ROOT config has a ``terminal:`` section. _default_spawn copies os.environ,
# and the child's bridge backfills with override=False, so these would silently
# override the assignee profile's own backend, mounts, injected env and egress.
LEAKED = {
    "TERMINAL_ENV": "local",
    "TERMINAL_MODAL_MODE": "auto",
    "TERMINAL_DOCKER_IMAGE": "nikolaik/python-nodejs:python3.11-nodejs20",
    "TERMINAL_MODAL_IMAGE": "nikolaik/python-nodejs:python3.11-nodejs20",
    "TERMINAL_SINGULARITY_IMAGE": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
    "TERMINAL_DAYTONA_IMAGE": "nikolaik/python-nodejs:python3.11-nodejs20",
    "TERMINAL_CONTAINER_PERSISTENT": "True",
    "TERMINAL_CONTAINER_CPU": "1",
    "TERMINAL_CONTAINER_MEMORY": "5120",
    "TERMINAL_CONTAINER_DISK": "51200",
    "TERMINAL_LIFETIME_SECONDS": "300",
    "TERMINAL_DOCKER_VOLUMES": "/host:/container",
    "TERMINAL_DOCKER_ENV": "SECRET=root",
    "TERMINAL_DOCKER_NETWORK": "host",
    "TERMINAL_DOCKER_EXTRA_ARGS": "--privileged",
    "TERMINAL_DOCKER_FORWARD_ENV": "PATH",
    "TERMINAL_SANDBOX_DIR": "/root/sandbox",
}


def test_inherited_terminal_config_vars_scrubbed(monkeypatch, tmp_path):
    """Leaked dispatcher TERMINAL_* config vars must NOT reach the worker."""
    _setup(tmp_path, monkeypatch, "", "terminal:\n  backend: local\n", LEAKED)
    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    got = _spawn(kb, monkeypatch, str(workspace))

    for key in LEAKED:
        assert key not in got["env"], f"{key} leaked into the worker env"
    # TERMINAL_CWD is a deliberate set, not a leak.
    assert got["env"]["TERMINAL_CWD"] == str(workspace)


def test_explicit_terminal_env_preserved_when_root_has_no_terminal_section(
    monkeypatch, tmp_path
):
    """Without a root ``terminal:`` section the bridge never clobbered anything,
    so an operator-exported TERMINAL_* is an explicit choice and must survive."""
    _setup(tmp_path, monkeypatch, "", "",
           {"TERMINAL_ENV": "modal", "TERMINAL_MODAL_IMAGE": "im-operatorExplicit123"})
    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()
    got = _spawn(kb, monkeypatch, str(workspace))

    assert got["env"]["TERMINAL_ENV"] == "modal"
    assert got["env"]["TERMINAL_MODAL_IMAGE"] == "im-operatorExplicit123"
