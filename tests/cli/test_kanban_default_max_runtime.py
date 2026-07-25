"""Default runtime cap for kanban tasks (kanban.default_max_runtime_seconds).

Regression context: ``enforce_max_runtime`` only reclaims rows where
``max_runtime_seconds IS NOT NULL``. Tasks created without an explicit
``--max-runtime`` stored NULL, so that path never reclaimed them. On a real
board 91% of one lane's runs (879/970 over 30 days) had no cap at all.

This is a BACKSTOP behind ``dispatch_stale_timeout_seconds`` (which reclaims
on a stale heartbeat and so spares long healthy runs), not a replacement for
it. These tests pin the precedence contract:

  explicit value > config default > uncapped
"""
from __future__ import annotations

import pytest


@pytest.fixture
def kb_conn(monkeypatch, tmp_path):
    """A real kanban DB in a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    yield kb, conn
    conn.close()


def _stored_cap(kb, conn, task_id):
    row = conn.execute(
        "SELECT max_runtime_seconds FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    return row["max_runtime_seconds"]


class TestDefaultMaxRuntime:
    def test_explicit_value_wins_over_config_default(self, kb_conn, monkeypatch):
        kb, conn = kb_conn
        monkeypatch.setattr(kb, "_default_max_runtime_seconds", lambda: 5400)

        tid = kb.create_task(
            conn, title="explicit", assignee="dev", max_runtime_seconds=600
        )
        assert _stored_cap(kb, conn, tid) == 600

    def test_config_default_applied_when_unset(self, kb_conn, monkeypatch):
        kb, conn = kb_conn
        monkeypatch.setattr(kb, "_default_max_runtime_seconds", lambda: 5400)

        tid = kb.create_task(conn, title="implicit", assignee="dev")
        assert _stored_cap(kb, conn, tid) == 5400

    def test_no_config_default_stays_uncapped(self, kb_conn, monkeypatch):
        """Absent config must preserve the old behavior, not invent a limit."""
        kb, conn = kb_conn
        monkeypatch.setattr(kb, "_default_max_runtime_seconds", lambda: None)

        tid = kb.create_task(conn, title="uncapped", assignee="dev")
        assert _stored_cap(kb, conn, tid) is None

    def test_explicit_value_survives_when_no_config_default(self, kb_conn, monkeypatch):
        kb, conn = kb_conn
        monkeypatch.setattr(kb, "_default_max_runtime_seconds", lambda: None)

        tid = kb.create_task(
            conn, title="explicit-only", assignee="dev", max_runtime_seconds=120
        )
        assert _stored_cap(kb, conn, tid) == 120


class TestDefaultMaxRuntimeResolver:
    """The config reader itself: it must never raise into create_task()."""

    def _resolve(self, monkeypatch, cfg):
        from hermes_cli import kanban_db as kb
        import hermes_cli.config as hc
        monkeypatch.setattr(hc, "load_config_readonly", lambda: cfg)
        return kb._default_max_runtime_seconds()

    def test_reads_configured_value(self, monkeypatch):
        got = self._resolve(monkeypatch, {"kanban": {"default_max_runtime_seconds": 5400}})
        assert got == 5400

    def test_missing_key_is_none(self, monkeypatch):
        assert self._resolve(monkeypatch, {"kanban": {}}) is None

    def test_missing_kanban_section_is_none(self, monkeypatch):
        assert self._resolve(monkeypatch, {}) is None

    def test_zero_means_opt_out_not_instant_kill(self, monkeypatch):
        """0 must disable the cap, NOT set a 0-second limit that kills every worker."""
        assert self._resolve(monkeypatch, {"kanban": {"default_max_runtime_seconds": 0}}) is None

    def test_negative_is_opt_out(self, monkeypatch):
        assert self._resolve(monkeypatch, {"kanban": {"default_max_runtime_seconds": -1}}) is None

    def test_string_value_is_coerced(self, monkeypatch):
        assert self._resolve(monkeypatch, {"kanban": {"default_max_runtime_seconds": "5400"}}) == 5400

    def test_garbage_value_falls_back_to_none(self, monkeypatch):
        """A malformed config must not break task creation."""
        assert self._resolve(monkeypatch, {"kanban": {"default_max_runtime_seconds": "ninety"}}) is None

    def test_config_load_failure_falls_back_to_none(self, monkeypatch):
        from hermes_cli import kanban_db as kb
        import hermes_cli.config as hc

        def boom():
            raise RuntimeError("no config here")

        monkeypatch.setattr(hc, "load_config_readonly", boom)
        assert kb._default_max_runtime_seconds() is None

    def test_resolver_does_not_mutate_the_shared_config(self, monkeypatch):
        """It reads the CACHED dict (no deepcopy), so it must never write to it.

        ``load_config_readonly`` hands back the live cache; mutating it would
        corrupt config for every later caller in the process. This pins the
        read-only contract that made the accessor switch safe.
        """
        from hermes_cli import kanban_db as kb
        import hermes_cli.config as hc

        cfg = {"kanban": {"default_max_runtime_seconds": 5400}}
        import copy
        before = copy.deepcopy(cfg)
        monkeypatch.setattr(hc, "load_config_readonly", lambda: cfg)

        assert kb._default_max_runtime_seconds() == 5400
        assert cfg == before, "resolver mutated the shared config cache"


class TestTimeoutEnforcementBehavior:
    def test_detector_reclaims_capped_but_not_uncapped_tasks(self, kb_conn, monkeypatch):
        """Exercise the reclaim behavior instead of pinning SQL source text."""
        kb, conn = kb_conn
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
        monkeypatch.setattr(kb, "_default_max_runtime_seconds", lambda: None)

        capped = kb.create_task(
            conn, title="capped", assignee="worker", max_runtime_seconds=1
        )
        uncapped = kb.create_task(conn, title="uncapped", assignee="worker")

        for tid in (capped, uncapped):
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, 999999)
            old_started = int(kb.time.time()) - 30
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET started_at = ? "
                    "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                    (old_started, tid),
                )

        timed_out = kb.enforce_max_runtime(
            conn, signal_fn=lambda _pid, _sig: None
        )

        assert timed_out == [capped]
        assert kb.get_task(conn, capped).status == "ready"
        assert kb.get_task(conn, uncapped).status == "running"
