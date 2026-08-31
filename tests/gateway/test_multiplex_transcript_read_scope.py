"""Profile-routed chats must READ their transcript from their own store.

A multiplexed gateway runs each turn inside ``_profile_runtime_scope``, so a
routed profile's messages are WRITTEN to ``profiles/<name>/state.db`` (#88532
fixed the write side). The gateway then reloaded history OUTSIDE that scope,
which resolves the ROOT ``state.db`` — a file that holds the session row and
zero messages. Every turn in a routed chat therefore replayed an empty
history: the agent answered "this session starts fresh, I have no earlier
chat history" while its full transcript sat intact in the profile store.

These tests pin the read path to the same scope the write path uses.
"""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig
from gateway.session import SessionStore
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def multiplex_homes(tmp_path, monkeypatch):
    """Root home plus a routed ``manager`` profile home, HERMES_HOME on root."""
    import hermes_state

    root = tmp_path / "hermes"
    profile = root / "profiles" / "manager"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    # Same rationale as test_multiplex_session_db_profile_scope: restore the
    # import-time constant so path resolution goes through get_hermes_home().
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    return root, profile


def _make_store(root: Path) -> SessionStore:
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=root / "sessions", config=GatewayConfig())
    store._loaded = True
    return store


def _write_conversation(store: SessionStore, session_id: str) -> None:
    store._db.create_session(session_id, "signal")
    store.append_to_transcript(session_id, {"role": "user", "content": "first ask"})
    store.append_to_transcript(
        session_id, {"role": "assistant", "content": "first answer"}
    )


def test_read_outside_scope_sees_nothing_written_under_scope(multiplex_homes):
    """The bug, stated directly: same session id, two different stores."""
    root, profile = multiplex_homes
    store = _make_store(root)
    session_id = "20260830_185308_54b749db"

    token = set_hermes_home_override(str(profile))
    try:
        _write_conversation(store, session_id)
    finally:
        reset_hermes_home_override(token)

    # Unscoped read: root state.db, which never received the messages.
    assert store.load_transcript(session_id) == []

    # Scoped read: the profile's own store has the whole conversation.
    token = set_hermes_home_override(str(profile))
    try:
        scoped = store.load_transcript(session_id)
    finally:
        reset_hermes_home_override(token)
    assert [m["role"] for m in scoped] == ["user", "assistant"]
    assert scoped[0]["content"] == "first ask"


@pytest.mark.asyncio
async def test_load_transcript_for_source_follows_the_routed_profile(multiplex_homes):
    """``_load_transcript_for_source`` reads the profile store under multiplexing."""
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore

    root, profile = multiplex_homes
    store = _make_store(root)
    session_id = "20260830_185308_54b749db"

    token = set_hermes_home_override(str(profile))
    try:
        _write_conversation(store, session_id)
    finally:
        reset_hermes_home_override(token)

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)

    class _Cfg:
        multiplex_profiles = True

    runner.config = _Cfg()
    source = object()
    with patch.object(
        GatewayRunner, "_resolve_profile_home_for_source", return_value=profile
    ):
        history = await runner._load_transcript_for_source(session_id, source)

    assert [m["role"] for m in history] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_single_profile_gateway_is_a_pass_through(multiplex_homes):
    """Multiplexing off: same store as before, scope never entered."""
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore

    root, _profile = multiplex_homes
    store = _make_store(root)
    session_id = "20260830_162623_f5e0851f"
    _write_conversation(store, session_id)

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)

    class _Cfg:
        multiplex_profiles = False

    runner.config = _Cfg()
    with patch.object(
        GatewayRunner, "_resolve_profile_home_for_source"
    ) as resolve:
        history = await runner._load_transcript_for_source(session_id, object())

    resolve.assert_not_called()
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_profile_store_is_the_only_writer(multiplex_homes):
    """Sanity: the root store holds no message rows for a routed session."""
    root, profile = multiplex_homes
    store = _make_store(root)
    session_id = "20260830_224239_2628b785"

    token = set_hermes_home_override(str(profile))
    try:
        _write_conversation(store, session_id)
    finally:
        reset_hermes_home_override(token)

    def _message_count(db_path: Path) -> int:
        if not db_path.exists():
            return 0
        conn = sqlite3.connect(str(db_path))
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ? "
                "AND role IN ('user', 'assistant')",
                (session_id,),
            ).fetchone()[0]
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    assert _message_count(profile / "state.db") == 2
    assert _message_count(root / "state.db") == 0
