"""Regression guard: ``tools.kanban_tools`` must be in the discovery list that
``model_tools`` executes at import time.

The kanban_* model-tools were invisible to every dispatcher-spawned worker
because ``tools.kanban_tools`` self-registers (so the registry's auto-glob
``discover_builtin_tools`` finds it) but was never added to the hardcoded
``_modules`` list inside ``model_tools._discover_tools`` — and that hardcoded
list, not the glob, is what runs at agent init. A worker therefore loaded zero
``kanban_*`` tools and could neither read (``kanban_show``) nor terminate
(``kanban_complete`` / ``kanban_block``) its own task.

These tests assert the fix two ways: the module is listed for discovery, and the
tools actually reach the model's tool surface when a worker enables the kanban
toolset.
"""
from __future__ import annotations

import inspect
import re


def _hardcoded_discovery_modules() -> set[str]:
    """The ``"tools.x"`` entries in model_tools._discover_tools, read from source.

    Read from source (not by executing the list) so the test does not depend on
    optional third-party tool imports succeeding in CI.
    """
    import model_tools

    src = inspect.getsource(model_tools._discover_tools)
    return set(re.findall(r'"(tools\.[a-zA-Z0-9_]+)"', src))


def test_kanban_tools_module_is_discovered():
    """kanban_tools must be in the list model_tools imports at init."""
    assert "tools.kanban_tools" in _hardcoded_discovery_modules()


def test_kanban_tools_reach_worker_tool_surface(monkeypatch):
    """A worker (HERMES_KANBAN_TASK set) with the kanban toolset enabled must
    see the kanban_* lifecycle tools in its resolved schema."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")

    import model_tools as mt
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    defs = mt.get_tool_definitions(
        enabled_toolsets=["terminal", "file", "kanban", "skills"],
        quiet_mode=True,
    )
    names = {d["function"]["name"] for d in defs}
    # kanban_show / kanban_complete / kanban_block are the must-haves: without
    # them a worker cannot orient or terminate its run.
    for required in ("kanban_show", "kanban_complete", "kanban_block"):
        assert required in names, f"{required} missing from worker tool surface"
