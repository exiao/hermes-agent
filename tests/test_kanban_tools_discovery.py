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

def test_worker_lifecycle_survives_profile_disabled_kanban(monkeypatch):
    """A real worker whose profile lists ``kanban`` in disabled_toolsets (for
    token/cost reasons) must STILL receive the lifecycle tools — the force-
    append must not be defeated by the global disabled list, or the task can
    never finish via the lifecycle protocol."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")

    import model_tools as mt
    from tools.registry import invalidate_check_fn_cache

    invalidate_check_fn_cache()
    mt._clear_tool_defs_cache()
    defs = mt.get_tool_definitions(
        enabled_toolsets=["terminal", "file", "skills"],
        disabled_toolsets=["kanban"],
        quiet_mode=True,
    )
    names = {d["function"]["name"] for d in defs}
    for required in ("kanban_show", "kanban_complete", "kanban_block"):
        assert required in names, f"{required} stripped by profile disabled_toolsets"


def test_delegated_child_does_not_get_worker_lifecycle_tools(monkeypatch):
    """A delegated review child inherits the parent worker's HERMES_KANBAN_TASK
    but must NEVER receive lifecycle tools (it must not mutate the parent's
    task). The delegate-child ownership mask — NOT the disabled list — is what
    suppresses the force-append."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")

    import model_tools as mt
    from tools.registry import invalidate_check_fn_cache
    from tools.delegate_tool import delegated_child_kanban_env

    invalidate_check_fn_cache()
    mt._clear_tool_defs_cache()
    with delegated_child_kanban_env():
        defs = mt.get_tool_definitions(
            enabled_toolsets=["terminal", "file", "skills"],
            disabled_toolsets=["kanban"],
            quiet_mode=True,
        )
    names = {d["function"]["name"] for d in defs}
    for forbidden in ("kanban_complete", "kanban_block"):
        assert forbidden not in names, (
            f"{forbidden} must not reach a delegated child's tool surface"
        )
