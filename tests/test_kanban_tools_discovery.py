"""Regression guard: dispatcher-spawned kanban workers must receive the
``kanban_*`` lifecycle tools in their model schema.

The kanban_* model-tools were invisible to every worker because
``tools.kanban_tools`` self-registers (so the registry's auto-glob
``discover_builtin_tools`` finds it) but was never added to the hardcoded
``_modules`` list inside ``model_tools._discover_tools`` — and that hardcoded
list, not the glob, is what runs at agent init. A worker therefore loaded zero
``kanban_*`` tools and could neither read (``kanban_show``) nor terminate
(``kanban_complete`` / ``kanban_block``) its own task.

This is a behavior invariant, not a source snapshot: it asserts what a worker's
resolved tool surface actually contains, so it stays valid no matter which
discovery mechanism (hardcoded list or auto-glob) does the importing.
"""
from __future__ import annotations


def test_kanban_tools_reach_worker_tool_surface(monkeypatch):
    """A worker (HERMES_KANBAN_TASK set) with the kanban toolset enabled must
    see the kanban_* lifecycle tools in its resolved schema."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")

    import model_tools as mt
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    mt._clear_tool_defs_cache()
    defs = mt.get_tool_definitions(
        enabled_toolsets=["terminal", "file", "kanban", "skills"],
        quiet_mode=True,
    )
    names = {d["function"]["name"] for d in defs}
    # kanban_show / kanban_complete / kanban_block are the must-haves: without
    # them a worker cannot orient or terminate its run.
    for required in ("kanban_show", "kanban_complete", "kanban_block"):
        assert required in names, f"{required} missing from worker tool surface"
