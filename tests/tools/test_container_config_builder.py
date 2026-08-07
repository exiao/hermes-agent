"""Every caller of ``_create_environment`` must build the SAME container config.

The four call sites used to hand-copy their own subset of keys, and the subsets
drifted: ``file_tools`` and ``code_execution_tool`` never passed ``modal_mode``,
``docker_shm_size``, ``docker_extra_args`` or ``docker_persist_across_processes``.
So the same config.yaml produced a different container depending on which tool
happened to create the environment first.
"""

import ast
from pathlib import Path

import pytest

from tools.terminal_tool import (
    CONTAINER_ENV_TYPES,
    _CONTAINER_CONFIG_DEFAULTS,
    build_container_config,
)

CALL_SITES = [
    "tools/terminal_tool.py",
    "tools/file_tools.py",
    "tools/code_execution_tool.py",
    "agent/prompt_builder.py",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestBuilder:
    @pytest.mark.parametrize("env_type", sorted(CONTAINER_ENV_TYPES))
    def test_every_container_backend_gets_every_key(self, env_type):
        assert set(build_container_config(env_type, {})) == set(
            _CONTAINER_CONFIG_DEFAULTS
        )

    @pytest.mark.parametrize("env_type", ["local", "ssh", ""])
    def test_non_container_backends_get_none(self, env_type):
        assert build_container_config(env_type, {}) is None

    def test_user_config_wins_over_defaults(self):
        cfg = build_container_config("modal", {"container_idle_timeout": 900})
        assert cfg["container_idle_timeout"] == 900
        assert cfg["modal_mode"] == _CONTAINER_CONFIG_DEFAULTS["modal_mode"]

    def test_keys_the_drift_dropped_are_present(self):
        """These four are exactly what the smaller hand-copied dicts omitted."""
        cfg = build_container_config("docker", {})
        for key in (
            "modal_mode",
            "docker_shm_size",
            "docker_extra_args",
            "docker_persist_across_processes",
        ):
            assert key in cfg

    def test_result_is_not_shared_state(self):
        """A mutable default must not leak between environments."""
        first = build_container_config("docker", {})
        first["docker_volumes"].append("/tmp:/tmp")
        assert build_container_config("docker", {})["docker_volumes"] == []


def test_no_call_site_hand_builds_a_container_config():
    """Guards against a fifth copy being added instead of calling the builder.

    An inline ``{"container_cpu": ...}`` literal is how the drift started, so
    the builder is the only place that dict may be constructed.
    """
    offenders = []
    for rel in CALL_SITES:
        tree = ast.parse((REPO_ROOT / rel).read_text())
        # The canonical defaults dict is the one legitimate literal.
        canonical = {
            n.value.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.AnnAssign | ast.Assign)
            and isinstance(n.value, ast.Dict)
            and "_CONTAINER_CONFIG_DEFAULTS" in ast.dump(n)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict) or node.lineno in canonical:
                continue
            keys = {
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            # Only a dict that is ENTIRELY container-config keys is a copy of
            # the builder; _get_env_config holds these alongside many others.
            if keys >= {"container_cpu", "container_memory"} and keys <= set(
                _CONTAINER_CONFIG_DEFAULTS
            ):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, (
        "hand-built container_config found; call build_container_config instead: "
        f"{offenders}"
    )
