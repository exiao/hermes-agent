"""Tests for the kanban dashboard board-load short-TTL cache.

``plugin_api.get_board`` is a sync FastAPI route whose ~40ms diagnostics pass
runs under the GIL; concurrent fetches serialize and pile up. A short-TTL
in-process cache keyed by the full resolved query-param tuple collapses that
pileup — within the window every caller replays one memoized payload instead of
recomputing.

These tests build a real on-disk board DB via ``kanban_db`` and drive
``get_board`` directly (it is a plain function; FastAPI's ``Query`` defaults are
just objects, so we pass explicit args). We spy on ``_compute_task_diagnostics``
(the diagnostics pass) to prove when recomputation does and doesn't happen.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from hermes_cli import kanban_db as kb


def _load_plugin_module():
    """Load plugins/kanban/dashboard/plugin_api.py by file path.

    The dashboard plugin dir is not an importable package. Mirrors the loader
    in test_dashboard_diagnostics_scope.py.
    """
    repo_root = Path(__file__).resolve().parents[3]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_ttl_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


plugin_api = _load_plugin_module()


def _get_board(**overrides):
    """Call get_board with real argument values (the signature's ``Query(...)``
    defaults are FastAPI marker objects, not usable values)."""
    kwargs = dict(
        tenant=None,
        assignee=None,
        include_archived=False,
        board=None,
        workflow_template_id=None,
        current_step_key=None,
        done_limit=plugin_api._DONE_LIMIT_DEFAULT,
        done_since=None,
    )
    kwargs.update(overrides)
    return plugin_api.get_board(**kwargs)


def _make_board_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = tmp_path / "kanban.db"
    return kb.connect(db_path=db_path)


def _reset_cache():
    with plugin_api._BOARD_CACHE_LOCK:
        plugin_api._BOARD_CACHE.clear()


def _spy_diagnostics(monkeypatch):
    """Wrap ``_compute_task_diagnostics`` with a call counter, returning the
    counter dict. The wrapper preserves the real behaviour."""
    real = plugin_api._compute_task_diagnostics
    counter = {"n": 0}

    def _wrapped(conn, task_ids=None):
        counter["n"] += 1
        return real(conn, task_ids=task_ids)

    monkeypatch.setattr(plugin_api, "_compute_task_diagnostics", _wrapped)
    return counter


def test_two_calls_within_ttl_return_same_object_without_recompute(tmp_path, monkeypatch):
    _reset_cache()
    conn = _make_board_db(tmp_path, monkeypatch)
    kb.create_task(conn, title="live one", assignee="x")
    conn.close()

    counter = _spy_diagnostics(monkeypatch)

    first = _get_board()
    second = _get_board()

    # Same memoized object handed back — no recompute on the second call.
    assert second is first
    assert counter["n"] == 1, "diagnostics recomputed inside the TTL window"


def test_call_after_ttl_expiry_recomputes(tmp_path, monkeypatch):
    _reset_cache()
    conn = _make_board_db(tmp_path, monkeypatch)
    kb.create_task(conn, title="live one", assignee="x")
    conn.close()

    counter = _spy_diagnostics(monkeypatch)

    # Freeze monotonic clock so we control expiry deterministically.
    clock = {"t": 1000.0}
    monkeypatch.setattr(plugin_api.time, "monotonic", lambda: clock["t"])

    first = _get_board()
    assert counter["n"] == 1

    # Still inside the window: no recompute.
    clock["t"] += plugin_api._BOARD_CACHE_TTL_SECONDS - 0.5
    second = _get_board()
    assert second is first
    assert counter["n"] == 1

    # Past the TTL: recompute, fresh payload.
    clock["t"] += 1.0  # now beyond expiry
    third = _get_board()
    assert third is not first
    assert counter["n"] == 2


def test_different_query_params_get_different_cache_entries(tmp_path, monkeypatch):
    _reset_cache()
    conn = _make_board_db(tmp_path, monkeypatch)
    kb.create_task(conn, title="live one", assignee="alice")
    kb.create_task(conn, title="live two", assignee="bob")
    conn.close()

    counter = _spy_diagnostics(monkeypatch)

    a1 = _get_board(assignee="alice")
    b1 = _get_board(assignee="bob")
    # Two distinct keys → two computes.
    assert counter["n"] == 2

    # Re-fetch each within the window → served from cache, no recompute.
    a2 = _get_board(assignee="alice")
    b2 = _get_board(assignee="bob")
    assert a2 is a1
    assert b2 is b1
    assert counter["n"] == 2

    # Two entries live in the cache.
    assert len(plugin_api._BOARD_CACHE) == 2


def test_concurrent_warm_cache_serves_memoized_payload(tmp_path, monkeypatch):
    """8 concurrent get_board calls against a WARM cache all return the one
    memoized payload without extra diagnostics computes — the pileup fix."""
    import threading

    _reset_cache()
    conn = _make_board_db(tmp_path, monkeypatch)
    for i in range(5):
        kb.create_task(conn, title=f"live {i}", assignee="x")
    conn.close()

    counter = _spy_diagnostics(monkeypatch)

    # Warm the cache once.
    warm = _get_board()
    assert counter["n"] == 1

    results: list = []
    barrier = threading.Barrier(8)

    def _worker():
        barrier.wait()
        results.append(_get_board())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    assert all(r is warm for r in results), "warm-cache callers recomputed"
    assert counter["n"] == 1, "diagnostics recomputed under concurrent warm-cache load"


def test_cache_miss_return_shape_preserved(tmp_path, monkeypatch):
    """A cache-miss payload carries the exact documented top-level keys."""
    _reset_cache()
    conn = _make_board_db(tmp_path, monkeypatch)
    kb.create_task(conn, title="live one", assignee="x")
    conn.close()

    payload = _get_board()
    assert set(payload.keys()) == {
        "columns",
        "done_window",
        "tenants",
        "assignees",
        "latest_event_id",
        "now",
    }
    assert isinstance(payload["columns"], list)
    assert set(payload["done_window"].keys()) == {"limit", "since"}


def _setup_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME (and Path.home) at a temp root WITHOUT pinning a
    fixed db_path, so board reads/writes flow through the real resolution
    chain (env → ``current`` file → ``default``). Unlike ``_make_board_db``
    this lets ``get_board(board=None)`` and ``kb.connect(board=...)`` agree on
    where each board's DB lives — required for a board-switch test that
    asserts on payload *content*, not just cache identity."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def test_switching_active_board_bypasses_cache_for_none_board(tmp_path, monkeypatch):
    """A ``board=None`` request keys on the *resolved* active board, so moving
    the ``current`` pointer between two calls returns each board's own data
    instead of the first board's cached payload (the cross-board pollution the
    high-severity review flagged)."""
    _reset_cache()
    _setup_hermes_home(tmp_path, monkeypatch)

    # Seed the default board through the resolution chain.
    conn_default = kb.connect(board="default")
    kb.create_task(conn_default, title="default task", assignee="x")
    conn_default.close()

    # Seed a second board.
    kb.create_board("other")
    conn_other = kb.connect(board="other")
    kb.create_task(conn_other, title="other task", assignee="y")
    conn_other.close()

    kb.set_current_board("default")
    first = _get_board(board=None)
    assert any(
        t["title"] == "default task"
        for col in first["columns"]
        for t in col["tasks"]
    )

    kb.set_current_board("other")
    second = _get_board(board=None)
    assert any(
        t["title"] == "other task"
        for col in second["columns"]
        for t in col["tasks"]
    )
    # A distinct board resolved to a distinct cache key — not the stale hit.
    assert second is not first
    assert not any(
        t["title"] == "default task"
        for col in second["columns"]
        for t in col["tasks"]
    )



# ---------------------------------------------------------------------------
# Write-side invalidation (follow-up to #89): a dashboard mutation within the
# TTL window must drop the stale entry so the next board read reflects the
# write instead of replaying the pre-write payload.
# ---------------------------------------------------------------------------


def _titles(payload) -> set[str]:
    return {
        t["title"]
        for col in payload["columns"]
        for t in col["tasks"]
    }


def _freeze_clock(monkeypatch):
    """Pin time.monotonic so the TTL never expires on its own — any refresh
    across a mutation must therefore come from write invalidation, not the
    2.5s timer. Returns the mutable clock dict."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(plugin_api.time, "monotonic", lambda: clock["t"])
    return clock


def test_create_within_ttl_invalidates_stale_board(tmp_path, monkeypatch):
    _reset_cache()
    _setup_hermes_home(tmp_path, monkeypatch)
    kb.set_current_board("default")
    conn = kb.connect(board="default")
    kb.create_task(conn, title="first task", assignee="x")
    conn.close()

    _freeze_clock(monkeypatch)

    warm = plugin_api.get_board(
        tenant=None, assignee=None, include_archived=False, board=None,
        workflow_template_id=None, current_step_key=None,
        done_limit=plugin_api._DONE_LIMIT_DEFAULT, done_since=None,
    )
    assert _titles(warm) == {"first task"}

    # Mutate through the dashboard route while the cache entry is still live
    # (clock frozen → not expired). Without invalidation the next read would
    # replay ``warm`` and never show "second task".
    plugin_api.create_task(
        plugin_api.CreateTaskBody(title="second task", assignee="x"),
        board=None,
    )

    after = plugin_api.get_board(
        tenant=None, assignee=None, include_archived=False, board=None,
        workflow_template_id=None, current_step_key=None,
        done_limit=plugin_api._DONE_LIMIT_DEFAULT, done_since=None,
    )
    assert after is not warm
    assert _titles(after) == {"first task", "second task"}


def test_delete_within_ttl_invalidates_stale_board(tmp_path, monkeypatch):
    _reset_cache()
    _setup_hermes_home(tmp_path, monkeypatch)
    kb.set_current_board("default")
    conn = kb.connect(board="default")
    tid = kb.create_task(conn, title="doomed", assignee="x")
    kb.create_task(conn, title="survivor", assignee="x")
    conn.close()

    _freeze_clock(monkeypatch)

    warm = plugin_api.get_board(
        tenant=None, assignee=None, include_archived=False, board=None,
        workflow_template_id=None, current_step_key=None,
        done_limit=plugin_api._DONE_LIMIT_DEFAULT, done_since=None,
    )
    assert _titles(warm) == {"doomed", "survivor"}

    plugin_api.delete_task(tid, board=None)

    after = plugin_api.get_board(
        tenant=None, assignee=None, include_archived=False, board=None,
        workflow_template_id=None, current_step_key=None,
        done_limit=plugin_api._DONE_LIMIT_DEFAULT, done_since=None,
    )
    assert _titles(after) == {"survivor"}


def test_patch_within_ttl_invalidates_stale_board(tmp_path, monkeypatch):
    _reset_cache()
    _setup_hermes_home(tmp_path, monkeypatch)
    kb.set_current_board("default")
    conn = kb.connect(board="default")
    tid = kb.create_task(conn, title="old title", assignee="x")
    conn.close()

    _freeze_clock(monkeypatch)

    warm = plugin_api.get_board(
        tenant=None, assignee=None, include_archived=False, board=None,
        workflow_template_id=None, current_step_key=None,
        done_limit=plugin_api._DONE_LIMIT_DEFAULT, done_since=None,
    )
    assert _titles(warm) == {"old title"}

    plugin_api.update_task(
        tid, plugin_api.UpdateTaskBody(title="new title"), board=None,
    )

    after = plugin_api.get_board(
        tenant=None, assignee=None, include_archived=False, board=None,
        workflow_template_id=None, current_step_key=None,
        done_limit=plugin_api._DONE_LIMIT_DEFAULT, done_since=None,
    )
    assert _titles(after) == {"new title"}


def test_invalidate_helper_clears_all_entries(tmp_path, monkeypatch):
    _reset_cache()
    _setup_hermes_home(tmp_path, monkeypatch)
    kb.set_current_board("default")
    conn = kb.connect(board="default")
    kb.create_task(conn, title="a", assignee="alice")
    kb.create_task(conn, title="b", assignee="bob")
    conn.close()

    # Two distinct cache keys.
    _get_board(assignee="alice")
    _get_board(assignee="bob")
    assert len(plugin_api._BOARD_CACHE) == 2

    plugin_api._invalidate_board_cache()
    assert plugin_api._BOARD_CACHE == {}


def test_dispatch_non_dryrun_invalidates_board_cache(tmp_path, monkeypatch):
    """A real (``dry_run=False``) dispatch nudge from the toolbar drops the
    cache so the post-nudge ``loadBoard()`` shows the dispatched state instead
    of replaying the pre-dispatch payload within the TTL. A ``dry_run=True``
    preview must NOT invalidate (it mutates nothing)."""
    _reset_cache()
    _setup_hermes_home(tmp_path, monkeypatch)
    kb.set_current_board("default")
    conn = kb.connect(board="default")
    kb.create_task(conn, title="pending task", assignee="x")
    conn.close()

    _freeze_clock(monkeypatch)

    # Warm the cache.
    _get_board(board=None)
    assert len(plugin_api._BOARD_CACHE) == 1

    # dispatch_once is exercised for real behaviour is out of scope here; we
    # only assert the cache-invalidation contract of the endpoint, so stub it.
    from dataclasses import dataclass

    @dataclass
    class _FakeResult:
        spawned: int = 0

    monkeypatch.setattr(
        plugin_api.kanban_db, "dispatch_once",
        lambda conn, dry_run, max_spawn, board: _FakeResult(),
    )

    # dry_run preview must leave the cache intact.
    plugin_api.dispatch(dry_run=True, max_n=8, board=None)
    assert len(plugin_api._BOARD_CACHE) == 1

    # A real nudge must clear it.
    plugin_api.dispatch(dry_run=False, max_n=8, board=None)
    assert plugin_api._BOARD_CACHE == {}


def test_inflight_fill_does_not_repopulate_after_concurrent_invalidation(tmp_path, monkeypatch):
    """A slow board fill that started before a dashboard write must NOT store
    its pre-write payload after the write's invalidation runs. The generation
    guard makes such a fill return its payload to its own caller but decline to
    cache it, so the next read recomputes instead of serving stale data for the
    rest of the TTL window."""
    _reset_cache()
    _setup_hermes_home(tmp_path, monkeypatch)
    kb.set_current_board("default")
    conn = kb.connect(board="default")
    kb.create_task(conn, title="original", assignee="x")
    conn.close()

    _freeze_clock(monkeypatch)

    real_compute = plugin_api._compute_board

    def _compute_then_invalidate(conn, **kwargs):
        # Simulate a dashboard mutation committing + invalidating WHILE this
        # slow board fill is in flight (i.e. after the caller snapshotted the
        # generation at the cache miss, before it stores the result).
        payload = real_compute(conn, **kwargs)
        plugin_api._invalidate_board_cache()
        return payload

    monkeypatch.setattr(plugin_api, "_compute_board", _compute_then_invalidate)

    result = _get_board(board=None)
    # The caller still gets a real payload (as fresh as it could have been).
    assert _titles(result) == {"original"}
    # But it must NOT have been cached — the mid-flight invalidation bumped the
    # generation, so the store was skipped and the cache stays empty.
    assert plugin_api._BOARD_CACHE == {}

