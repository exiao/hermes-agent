"""Tests for docker container_config key propagation in file_tools."""

from unittest.mock import patch, MagicMock
import tools.file_tools as file_tools


def _make_env_config(**overrides):
    base = {
        "env_type": "docker",
        "docker_image": "test-image:latest",
        "singularity_image": "docker://test",
        "modal_image": "test",
        "daytona_image": "test",
        "cwd": "/workspace",
        "host_cwd": None,
        "timeout": 180,
        "lifetime_seconds": 300,
        "container_cpu": 2,
        "container_memory": 4096,
        "container_disk": 20480,
        "container_persistent": False,
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["MY_SECRET", "API_KEY"],
    }
    base.update(overrides)
    return base


class TestFileToolsContainerConfig:
    def _run(self, env_config, task_id, task_env_overrides=None):
        captured = {}
        mock_env = MagicMock()

        def fake_create_env(**kwargs):
            captured.update(kwargs)
            return mock_env

        with patch("tools.terminal_tool._get_env_config", return_value=env_config), \
             patch("tools.terminal_tool._task_env_overrides", task_env_overrides or {}), \
             patch("tools.terminal_tool._active_environments", {}), \
             patch("tools.terminal_tool._creation_locks", {}), \
             patch("tools.terminal_tool._creation_locks_lock", __import__("threading").Lock()), \
             patch("tools.terminal_tool._create_environment", side_effect=fake_create_env), \
             patch("tools.terminal_tool._start_cleanup_thread"), \
             patch("tools.terminal_tool._check_disk_usage_warning"), \
             patch("tools.file_tools._file_ops_cache", {}), \
             patch("tools.file_tools._file_ops_lock", __import__("threading").Lock()):
            file_tools._get_file_ops(task_id)

        return captured

    def test_docker_mount_cwd_to_workspace_passed(self):
        """docker_mount_cwd_to_workspace is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_mount_cwd_to_workspace=True), "t1").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is True

    def test_docker_forward_env_passed(self):
        """docker_forward_env is forwarded to container_config."""
        cc = self._run(_make_env_config(docker_forward_env=["MY_SECRET"]), "t2").get("container_config", {})
        assert cc.get("docker_forward_env") == ["MY_SECRET"]

    def test_docker_mount_cwd_defaults_to_false(self):
        """docker_mount_cwd_to_workspace defaults to False when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_mount_cwd_to_workspace"]
        cc = self._run(cfg, "t3").get("container_config", {})
        assert cc.get("docker_mount_cwd_to_workspace") is False

    def test_docker_forward_env_defaults_to_empty_list(self):
        """docker_forward_env defaults to [] when absent from config."""
        cfg = _make_env_config()
        del cfg["docker_forward_env"]
        cc = self._run(cfg, "t4").get("container_config", {})
        assert cc.get("docker_forward_env") == []

    def test_cwd_only_raw_task_override_reaches_file_environment(self):
        """CWD-only task overrides collapse to default but must keep their cwd."""
        captured = self._run(
            _make_env_config(env_type="local", cwd="/config-cwd"),
            "desktop-session-cwd",
            task_env_overrides={"desktop-session-cwd": {"cwd": "/workspace/session"}},
        )

        assert captured["task_id"] == "default"
        assert captured["cwd"] == "/workspace/session"

    def test_multiplexed_profile_scopes_do_not_share_file_operations(self, monkeypatch):
        """File tools must use the profile-qualified terminal environment key."""
        from tools import terminal_tool
        from tools.terminal_config import (
            reset_terminal_config_scope,
            set_terminal_config_scope,
        )

        created = []

        def fake_create_environment(**kwargs):
            environment = MagicMock()
            environment.cwd = kwargs["cwd"]
            created.append((kwargs, environment))
            return environment

        monkeypatch.setattr(file_tools, "_file_ops_cache", {})
        monkeypatch.setattr(terminal_tool, "_active_environments", {})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_environment_lifetimes", {})
        monkeypatch.setattr(terminal_tool, "_creation_locks", {})
        monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _make_env_config())
        monkeypatch.setattr(terminal_tool, "_create_environment", fake_create_environment)
        monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

        profile_a = set_terminal_config_scope({}, environment_key="profile:profile-a")
        try:
            file_ops_a = file_tools._get_file_ops("session-a")
        finally:
            reset_terminal_config_scope(profile_a)

        profile_b = set_terminal_config_scope({}, environment_key="profile:profile-b")
        try:
            file_ops_b = file_tools._get_file_ops("session-b")
        finally:
            reset_terminal_config_scope(profile_b)

        assert file_ops_a is not file_ops_b
        assert {kwargs["task_id"] for kwargs, _environment in created} == {
            "profile:profile-a:default",
            "profile:profile-b:default",
        }
        assert terminal_tool._environment_lifetimes == {
            "profile:profile-a:default": 300,
            "profile:profile-b:default": 300,
        }

        unscoped_first = file_tools._get_file_ops("session-c")
        unscoped_second = file_tools._get_file_ops("session-d")

        assert unscoped_first is unscoped_second
        assert created[-1][0]["task_id"] == "default"
