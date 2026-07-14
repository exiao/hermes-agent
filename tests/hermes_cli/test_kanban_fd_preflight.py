"""Tests for the FD-headroom preflight in ``hermes_cli.kanban_db.connect``.

Layer 2 corruption guard (2026-07-13 FD-leak incident): before opening the
shared kanban DB, ``connect()`` refuses when the process OR the host is within
``kanban.fd_headroom`` file descriptors of an FD ceiling. Opening a WAL board
under FD starvation is the documented index-desync corruption path; a refused
connect is self-healing because the dispatcher / notifier / claim paths already
skip a failed tick and retry.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an initialized kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Neutralize any ambient env override so tests control the threshold.
    monkeypatch.delenv("HERMES_KANBAN_FD_HEADROOM", raising=False)
    kb.init_db()
    return home


# --------------------------------------------------------------------------
# _resolve_fd_headroom
# --------------------------------------------------------------------------


def test_headroom_default_when_unset(kanban_home, monkeypatch):
    # No env override; config load returns the default section → the DEFAULT.
    assert kb._resolve_fd_headroom() == kb.DEFAULT_FD_HEADROOM


def test_headroom_env_override_wins(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_FD_HEADROOM", "999")
    assert kb._resolve_fd_headroom() == 999


def test_headroom_env_invalid_falls_through(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_FD_HEADROOM", "not-an-int")
    # Falls through to config/default, never raises.
    assert kb._resolve_fd_headroom() == kb.DEFAULT_FD_HEADROOM


def test_headroom_config_value(kanban_home, monkeypatch):
    import hermes_cli.config as config

    monkeypatch.setattr(
        config, "load_config_readonly", lambda *a, **k: {"kanban": {"fd_headroom": 128}}
    )
    assert kb._resolve_fd_headroom() == 128


# --------------------------------------------------------------------------
# _assert_fd_headroom trip logic
# --------------------------------------------------------------------------


def test_preflight_trips_on_low_per_proc(monkeypatch):
    """Per-process headroom below threshold → KanbanFDPressureError."""
    monkeypatch.setattr(kb, "_resolve_fd_headroom", lambda: 64)
    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: 10)
    monkeypatch.setattr(kb, "_system_fd_headroom", lambda: None)
    with pytest.raises(kb.KanbanFDPressureError) as exc:
        kb._assert_fd_headroom()
    assert "per-process" in str(exc.value)


def test_preflight_trips_on_low_system_even_when_proc_fine(monkeypatch):
    """System-wide is the binding limit; per-proc healthy must not mask it."""
    monkeypatch.setattr(kb, "_resolve_fd_headroom", lambda: 64)
    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: 50_000)  # plenty
    monkeypatch.setattr(kb, "_system_fd_headroom", lambda: 5)     # near kern.maxfiles
    with pytest.raises(kb.KanbanFDPressureError) as exc:
        kb._assert_fd_headroom()
    assert "system-wide" in str(exc.value)


def test_preflight_ok_when_healthy(monkeypatch):
    monkeypatch.setattr(kb, "_resolve_fd_headroom", lambda: 64)
    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: 5000)
    monkeypatch.setattr(kb, "_system_fd_headroom", lambda: 20_000)
    kb._assert_fd_headroom()  # no raise


def test_preflight_noop_when_unknowable(monkeypatch):
    """Neither headroom computable → never block a connect we can't justify."""
    monkeypatch.setattr(kb, "_resolve_fd_headroom", lambda: 64)
    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: None)
    monkeypatch.setattr(kb, "_system_fd_headroom", lambda: None)
    kb._assert_fd_headroom()  # no raise


def test_preflight_disabled_when_threshold_zero(monkeypatch):
    monkeypatch.setattr(kb, "_resolve_fd_headroom", lambda: 0)
    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: 1)  # would trip if checked
    monkeypatch.setattr(kb, "_system_fd_headroom", lambda: 1)
    kb._assert_fd_headroom()  # no raise — check disabled


# --------------------------------------------------------------------------
# _count_open_fds fail-closed on EMFILE
# --------------------------------------------------------------------------


def test_count_open_fds_fails_closed_on_emfile(monkeypatch):
    """opendir()-ing the fd table under EMFILE means we're already starved:
    return a huge count so the preflight refuses (fail closed), not None."""

    def _boom(_path):
        raise OSError(24, "Too many open files")  # EMFILE

    monkeypatch.setattr(os, "listdir", _boom)
    assert kb._count_open_fds() == (1 << 30)


def test_count_open_fds_none_on_other_oserror(monkeypatch):
    def _boom(_path):
        raise OSError(13, "Permission denied")  # EACCES

    monkeypatch.setattr(os, "listdir", _boom)
    assert kb._count_open_fds() is None


# --------------------------------------------------------------------------
# connect() end-to-end: refuse cleanly, and a tick survives the refusal
# --------------------------------------------------------------------------


def test_connect_refuses_and_writes_no_sidecar(kanban_home, monkeypatch):
    """Under simulated pressure connect() raises before opening the DB —
    no fresh WAL/SHM sidecar is created by the refused open."""
    db_path = kb.kanban_db_path()
    wal = db_path.with_name(db_path.name + "-wal")
    shm = db_path.with_name(db_path.name + "-shm")
    # Clear any leftover sidecars from init so we can assert refusal is clean.
    for p in (wal, shm):
        if p.exists():
            p.unlink()
    # Also clear the per-process init cache so connect() would take the full
    # init path if the preflight didn't stop it first.
    kb._INITIALIZED_PATHS.clear()

    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: 1)
    monkeypatch.setattr(kb, "_system_fd_headroom", lambda: None)
    monkeypatch.setenv("HERMES_KANBAN_FD_HEADROOM", "64")

    with pytest.raises(kb.KanbanFDPressureError):
        kb.connect()

    assert not wal.exists()
    assert not shm.exists()


def test_connect_succeeds_before_and_after_pressure(kanban_home, monkeypatch):
    """Fail-before/pass-after: connect works healthy, refuses under pressure,
    then works again once pressure clears (self-healing)."""
    # Healthy: connect works.
    conn = kb.connect()
    conn.close()

    # Under pressure: refuse.
    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: 1)
    monkeypatch.setattr(kb, "_system_fd_headroom", lambda: None)
    monkeypatch.setenv("HERMES_KANBAN_FD_HEADROOM", "64")
    with pytest.raises(kb.KanbanFDPressureError):
        kb.connect()

    # Pressure clears: connect works again with no lingering damage.
    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: 5000)
    conn = kb.connect()
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def test_dispatcher_style_tick_survives_refusal(kanban_home, monkeypatch):
    """A caller that wraps connect() in `except Exception` and skips the tick
    (the dispatcher/notifier pattern) survives the refusal without crashing."""
    monkeypatch.setattr(kb, "_proc_fd_headroom", lambda: 1)
    monkeypatch.setattr(kb, "_system_fd_headroom", lambda: None)
    monkeypatch.setenv("HERMES_KANBAN_FD_HEADROOM", "64")

    tick_ran = False
    skipped = False
    try:
        conn = kb.connect()  # would corrupt if it opened under pressure
        tick_ran = True
        conn.close()
    except Exception:
        skipped = True  # dispatcher logs a warning and retries next tick

    assert skipped and not tick_ran
