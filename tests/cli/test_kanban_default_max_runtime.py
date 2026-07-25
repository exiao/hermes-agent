"""Default runtime cap for kanban tasks (kanban.default_max_runtime_seconds).

Regression context: ``enforce_max_runtime`` only reclaims rows where
``max_runtime_seconds IS NOT NULL``. Tasks created without an explicit
``--max-runtime`` stored NULL, so nothing ever reclaimed them. On Eric's board
91% of dev runs (879/970) had no cap, and one run (t_7c4131f0) burned 214.7
hours of Modal sandbox time (~$82) before anything noticed.

These tests pin the precedence contract:
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
        monkeypatch.setattr(hc, "load_config", lambda: cfg)
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

        monkeypatch.setattr(hc, "load_config", boom)
        assert kb._default_max_runtime_seconds() is None


class TestTimeoutEnforcementStillGatesOnNull:
    """Guard the assumption this whole fix rests on."""

    def test_detector_query_requires_non_null_cap(self):
        import inspect
        from hermes_cli import kanban_db as kb

        src = inspect.getsource(kb.enforce_max_runtime)
        assert "max_runtime_seconds IS NOT NULL" in src, (
            "enforce_max_runtime no longer filters on a non-NULL cap; "
            "the default-cap fix may be unnecessary or the reclaim path changed."
        )
