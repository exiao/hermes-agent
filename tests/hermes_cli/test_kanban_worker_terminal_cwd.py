"""Tests: kanban worker spawn pins TERMINAL_CWD to the task workspace.

Regression coverage for #34619 and #41312 (same root cause): ``_default_spawn``
launched the worker subprocess with ``cwd=workspace`` and set
``HERMES_KANBAN_WORKSPACE``, but did NOT set ``TERMINAL_CWD``. Because
``TERMINAL_CWD`` takes precedence over the process cwd in both
``tools/file_tools.py::_resolve_base_dir`` (relative ``write_file`` paths) and
``agent_init``'s context-file loader (``AGENTS.md`` discovery), workers inherited
the dispatching gateway's cwd — relative writes landed in the gateway user's
home (#41312) and the wrong profile's ``AGENTS.md`` was loaded (#34619).
Pinning ``TERMINAL_CWD`` to the workspace fixes both.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _make_task(kb, *, assignee: str = "w"):
    return kb.Task(
        id="t_cwd",
        title="cwd pin",
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


def _capture_spawn_env(kb, monkeypatch, workspace: str) -> dict:
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
    kb._default_spawn(_make_task(kb), workspace)
    return captured


def test_terminal_cwd_pinned_to_workspace(monkeypatch, tmp_path):
    """A real, absolute workspace dir is pinned as TERMINAL_CWD."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == str(workspace)
    # The subprocess cwd and TERMINAL_CWD must agree — both anchor the workspace.
    assert captured["cwd"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)


def test_remote_worker_does_not_receive_host_workspace_cwd(monkeypatch, tmp_path):
    """Modal workers must start in their sandbox, not a host-only path."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: modal\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_CWD", "/host/gateway")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert "TERMINAL_CWD" not in captured["env"]
    assert "HERMES_KANBAN_WORKSPACE" not in captured["env"]
    # cwd=None would inherit the dispatcher's checkout, which is exactly what
    # resolve_context_cwd()/_resolve_base_dir fall back to once TERMINAL_CWD is
    # gone. The worker must be launched in a neutral empty dir instead.
    assert captured["cwd"] is not None
    assert captured["cwd"] != str(workspace)
    assert captured["cwd"] != os.getcwd()
    assert not any(Path(captured["cwd"]).iterdir())


def test_docker_worker_still_receives_host_workspace_cwd(monkeypatch, tmp_path):
    """Docker bind-mounts host paths, so its workspace must stay pinned."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: docker\n"
        "  docker_mount_cwd_to_workspace: true\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_CWD", "/host/gateway")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert captured["cwd"] == str(workspace)


def test_inherited_terminal_backend_controls_remote_worker(monkeypatch, tmp_path):
    """An exported backend remains effective when both configs omit terminal."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_ENV", "modal")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_ENV"] == "modal"
    assert "HERMES_KANBAN_WORKSPACE" not in captured["env"]
    assert captured["cwd"] not in (None, os.getcwd(), str(workspace))


def test_non_docker_mount_flag_does_not_enable_host_workspace(monkeypatch, tmp_path):
    """A stale Docker mount flag must not make remote backends local."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: modal\n"
        "  docker_mount_cwd_to_workspace: true\n",
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert "HERMES_KANBAN_WORKSPACE" not in captured["env"]
    assert captured["cwd"] not in (None, os.getcwd(), str(workspace))


def test_terminal_cwd_not_pinned_for_nonexistent_workspace(monkeypatch, tmp_path):
    """A non-directory workspace must NOT clobber the inherited TERMINAL_CWD.

    file_tools rejects relative / sentinel TERMINAL_CWD values, so writing a
    meaningless (nonexistent) path would be worse than leaving the inherited
    one. The guard requires an existing absolute dir.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_CWD", "/pre/existing/anchor")

    from hermes_cli import kanban_db as kb

    missing = tmp_path / "does-not-exist"

    captured = _capture_spawn_env(kb, monkeypatch, str(missing))

    # Inherited value is preserved (not overwritten with a bogus path).
    assert captured["env"]["TERMINAL_CWD"] == "/pre/existing/anchor"


def test_inherited_terminal_config_vars_scrubbed(monkeypatch, tmp_path):
    """Leaked dispatcher TERMINAL_* config vars must NOT reach the worker.

    Regression: the dispatcher's lazy terminal-config bridge overwrites os.environ
    with the ROOT config's TERMINAL_* values (backend, images, container limits,
    docker mounts/env/network) whenever the root config has a ``terminal:``
    section. _default_spawn does ``env = dict(os.environ)``, so those leak into
    the child. Because the child's own bridge early-returns when TERMINAL_ENV is
    already present and otherwise backfills with override=False, the inherited
    root values silently override the assignee profile's terminal.* config. The
    spawn must scrub the config-derived TERMINAL_* vars so the child re-derives
    them from its own profile config. TERMINAL_CWD / TERMINAL_TIMEOUT are pinned
    deliberately and must survive.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    # Root has a `terminal:` section — this is what makes the dispatcher bridge
    # run with override=True and clobber os.environ, so scrubbing is warranted.
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: local\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))

    # Simulate the dispatcher's leaked env: backend, images, container limits, AND
    # the docker mount/env/network settings the reviewer flagged (P1). Every one
    # of these must be scrubbed so a docker/modal worker can't inherit the root
    # profile's backend, bind mounts, injected env, or egress policy.
    leaked = {
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
    for k, v in leaked.items():
        monkeypatch.setenv(k, v)

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    # None of the leaked config-derived vars survive into the child env.
    for key in leaked:
        assert key not in captured["env"], f"{key} leaked into the worker env"
    # TERMINAL_CWD is a deliberate set, not a leak — it must still be present.
    assert captured["env"]["TERMINAL_CWD"] == str(workspace)


def test_explicit_terminal_env_preserved_when_root_has_no_terminal_section(monkeypatch, tmp_path):
    """An operator's explicit TERMINAL_* must survive when root has no terminal cfg.

    When the ROOT config.yaml has NO ``terminal:`` section, the dispatcher's
    bridge ran with override=False and left any operator-exported TERMINAL_*
    (e.g. a gateway launched with ``TERMINAL_ENV=modal``) untouched — that IS an
    explicit selection. Scrubbing it would wrongly fall the worker back to its
    default backend, so the spawn must NOT scrub in this case.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    # No `terminal:` section in the root config.
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    # Operator explicitly exported a backend + image on the gateway.
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    monkeypatch.setenv("TERMINAL_MODAL_IMAGE", "im-operatorExplicit123")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    # The explicit operator selection is preserved, not scrubbed.
    assert captured["env"]["TERMINAL_ENV"] == "modal"
    assert captured["env"]["TERMINAL_MODAL_IMAGE"] == "im-operatorExplicit123"


def test_inherited_docker_mount_flag_keeps_workspace_pin(monkeypatch, tmp_path):
    """An env-only docker+mount selection must still pin the task workspace.

    When both configs omit ``terminal:`` and the gateway was launched with
    TERMINAL_ENV=docker plus TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE=true, the
    child inherits both. A probe that copies only the backend resolves
    mount_cwd as false, leaves cwd unpinned, and Docker then bind-mounts the
    dispatcher checkout at /workspace instead of the task workspace.
    """
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == str(workspace)
    assert captured["env"]["HERMES_KANBAN_WORKSPACE"] == str(workspace)
    assert captured["cwd"] == str(workspace)


def test_ssh_worker_drops_inherited_host_cwd(monkeypatch, tmp_path):
    """SSH runs `cd <cwd> || exit 126` remotely, so a host path must not survive."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: ssh\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_CWD", "/host/gateway")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert "TERMINAL_CWD" not in captured["env"]
    assert "HERMES_KANBAN_WORKSPACE" not in captured["env"]


def test_ssh_worker_keeps_its_own_profile_cwd(monkeypatch, tmp_path):
    """An explicit profile cwd is a real remote path and must be bridged."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: ssh\n  cwd: /srv/work\n",
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_CWD", "/host/gateway")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == "/srv/work"


def test_environment_only_ssh_keeps_explicit_remote_cwd(monkeypatch, tmp_path):
    """An env-only SSH selection may intentionally target a remote directory."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "w").mkdir(parents=True)
    (root / "profiles" / "w" / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "/srv/work")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert captured["env"]["TERMINAL_CWD"] == "/srv/work"
    assert "HERMES_KANBAN_WORKSPACE" not in captured["env"]
    assert captured["cwd"] not in (None, os.getcwd(), str(workspace))


def test_remote_worker_drops_inherited_kanban_workspace(monkeypatch, tmp_path):
    """Remote workers must not retain a parent's host workspace variable."""
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "w"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "toolsets:\n  - kanban\nterminal:\n  backend: modal\n",
        encoding="utf-8",
    )
    root.joinpath("config.yaml").write_text("toolsets:\n  - kanban\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/host/parent-workspace")

    from hermes_cli import kanban_db as kb

    workspace = tmp_path / "ws"
    workspace.mkdir()

    captured = _capture_spawn_env(kb, monkeypatch, str(workspace))

    assert "HERMES_KANBAN_WORKSPACE" not in captured["env"]
    assert captured["cwd"] not in (None, os.getcwd(), str(workspace))
