"""Regression coverage for #88532.

A multiplexed gateway serves every profile from one process.  ``SessionStore``
used to bind a single ``SessionDB`` during ``__init__``, freezing it to the
process's own root home, so a named profile's sessions were physically written
to the root ``state.db`` even though ``_profile_runtime_scope`` had already
redirected ``get_hermes_home()`` for that turn.  The rows carried the correct
``profile_name``, which is why the only visible symptom was the desktop listing
a profile's session under the default bot: the desktop reads
``profiles/<name>/state.db``, which never received the write.

These tests pin the handle to the *active* scope rather than to construction
time.  ``test_write_under_profile_scope_lands_in_profile_store`` is the one
that reproduces the report; it fails against the pre-fix code with the session
row sitting in the root store.
"""

import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import AsyncSessionStore, SessionStore, SessionSource
from gateway.platforms.base import MessageEvent
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def multiplex_homes(tmp_path, monkeypatch):
    """A root home plus a named profile home, with HERMES_HOME on the root.

    Mirrors the reported layout: one gateway process launched under the root
    home, serving a ``fitness`` profile whose store lives under
    ``profiles/fitness``.
    """
    import hermes_state

    root = tmp_path / "hermes"
    profile = root / "profiles" / "fitness"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))

    # The suite-wide fixture in conftest re-points ``hermes_state.DEFAULT_DB_PATH``
    # at a fake home, which trips the deliberate escape hatch in
    # ``_default_db_path()``: a re-pointed constant wins over everything,
    # including the context-local override.  That is correct for tests that
    # want one fixed DB, but it would pin every lookup here to a single path
    # and make these assertions vacuous.  Restore the import-time snapshot so
    # the hatch is closed and resolution goes through ``get_hermes_home()``,
    # which is what production does.  ``HERMES_HOME`` above still keeps that
    # resolution inside ``tmp_path``, so no real store is ever opened.
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    return root, profile


def _make_store(root: Path) -> SessionStore:
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(sessions_dir=root / "sessions", config=GatewayConfig())
    store._loaded = True
    return store


def _session_ids(db_path: Path) -> set:
    """Read session ids straight out of a state.db, or empty if absent."""
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT id FROM sessions").fetchall()
    except sqlite3.OperationalError:
        # No sessions table: nothing was ever written here.
        return set()
    finally:
        conn.close()
    return {r[0] for r in rows}


def test_store_uses_root_db_when_no_profile_scope_is_active(multiplex_homes):
    """Single-profile gateways are unaffected: no scope, same path as before."""
    root, _profile = multiplex_homes
    store = _make_store(root)

    assert Path(store._db.db_path) == root / "state.db"


def test_db_handle_follows_the_active_profile_scope(multiplex_homes):
    """The handle is resolved per access, not frozen at construction."""
    root, profile = multiplex_homes
    store = _make_store(root)

    # Constructed outside any scope, exactly as the gateway constructs it.
    assert Path(store._db.db_path) == root / "state.db"

    token = set_hermes_home_override(str(profile))
    try:
        assert Path(store._db.db_path) == profile / "state.db"
    finally:
        reset_hermes_home_override(token)

    # And the scope is restored once the turn's scope exits.
    assert Path(store._db.db_path) == root / "state.db"


def test_write_under_profile_scope_lands_in_profile_store(multiplex_homes):
    """The reported bug: the row must be in the profile's own file.

    This is the assertion the issue makes by hand with ``sqlite3``: the
    session for profile ``fitness`` belongs in ``profiles/fitness/state.db``
    and must NOT be in the root store.
    """
    root, profile = multiplex_homes
    store = _make_store(root)

    token = set_hermes_home_override(str(profile))
    try:
        store._db.create_session("20260817_233028_542fda58", "feishu")
    finally:
        reset_hermes_home_override(token)

    assert _session_ids(profile / "state.db") == {"20260817_233028_542fda58"}
    assert _session_ids(root / "state.db") == set()


def test_handles_are_cached_per_path(multiplex_homes):
    """One handle per profile: no reopen per message, no sharing across profiles."""
    root, profile = multiplex_homes
    store = _make_store(root)

    root_first = store._db
    root_second = store._db
    assert root_first is root_second

    token = set_hermes_home_override(str(profile))
    try:
        profile_first = store._db
        profile_second = store._db
    finally:
        reset_hermes_home_override(token)

    assert profile_first is profile_second
    assert profile_first is not root_first


def test_explicitly_pinned_handle_still_wins(multiplex_homes):
    """``store._db = ...`` remains authoritative for every subsequent read.

    Guardrail rather than a bug reproduction: a large number of existing
    tests install a fake handle or disable the DB this way, and the property
    must not quietly resolve past a deliberate assignment.
    """
    root, profile = multiplex_homes
    store = _make_store(root)

    sentinel = object()
    store._db = sentinel
    token = set_hermes_home_override(str(profile))
    try:
        assert store._db is sentinel
    finally:
        reset_hermes_home_override(token)

    # Disabling the DB (the JSONL-fallback path) must survive scope changes.
    store._db = None
    token = set_hermes_home_override(str(profile))
    try:
        assert store._db is None
    finally:
        reset_hermes_home_override(token)


def test_close_all_db_handles_sweeps_every_profile_handle(multiplex_homes):
    """Teardown must release every cached per-profile handle, not just the
    one the tearing-down task's own scope resolves.

    Follow-up hardening for the per-path cache: ``gateway/run.py``'s
    teardown path closes ``store._db`` (root scope only); the sweep closes
    the rest so secondary profiles' WAL locks are released before a
    ``--replace`` restart reopens their stores.
    """
    root, profile = multiplex_homes
    store = _make_store(root)

    root_db = store._db
    token = set_hermes_home_override(str(profile))
    try:
        profile_db = store._db
    finally:
        reset_hermes_home_override(token)
    assert root_db is not profile_db

    store.close_all_db_handles()

    # Both handles are closed (connection released) and the cache is empty,
    # so the next access opens a fresh handle rather than a dead one.
    assert root_db._conn is None
    assert profile_db._conn is None
    assert store._db_handles == {}
    fresh = store._db
    assert fresh is not root_db
    assert fresh._conn is not None
    fresh.close()


def test_runner_session_db_follows_the_active_profile_scope(multiplex_homes):
    """GatewayRunner._session_db is the same frozen-handle class of bug.

    /resume, /title, /history and session search run inside
    ``_profile_runtime_scope`` on a multiplexed gateway and must read the
    serving profile's state.db.  Exercise the property on a bare runner shell
    (full construction wires adapters and is irrelevant to the seam under
    test).
    """
    import threading

    from gateway.run import GatewayRunner, _SESSION_DB_UNPINNED

    root, profile = multiplex_homes
    runner = object.__new__(GatewayRunner)
    runner._session_db_pinned = _SESSION_DB_UNPINNED
    runner._session_db_handles = {}
    runner._session_db_handles_lock = threading.Lock()

    root_db = runner._session_db
    assert Path(root_db._db.db_path) == root / "state.db"

    token = set_hermes_home_override(str(profile))
    try:
        profile_db = runner._session_db
        assert Path(profile_db._db.db_path) == profile / "state.db"
        # Cached per path: same wrapper identity on re-access.
        assert runner._session_db is profile_db
    finally:
        reset_hermes_home_override(token)

    assert runner._session_db is root_db

    # Pinning (how suites install fakes / disable the DB) wins across scopes.
    runner._session_db = None
    token = set_hermes_home_override(str(profile))
    try:
        assert runner._session_db is None
    finally:
        reset_hermes_home_override(token)
    runner._session_db_pinned = _SESSION_DB_UNPINNED

    runner.close_all_session_db_handles()
    assert runner._session_db_handles == {}
    assert root_db._db._conn is None
    assert profile_db._db._conn is None


@pytest.mark.asyncio
async def test_inbound_turn_uses_real_profile_scoped_session_store(multiplex_homes, monkeypatch):
    """The complete inbound path loads and persists against the routed store.

    This reproduces the reported split: the runner is constructed outside a
    profile scope, then a real multiplexed turn resolves an existing session,
    loads its prior messages, and persists the next turn.  Both the read and
    write assertions inspect the two actual state.db files, so a test double at
    ``_handle_message_with_agent_inner`` cannot make this pass.
    """
    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home

    root, profile = multiplex_homes
    store = _make_store(root)
    store.config.multiplex_profiles = True
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-88532",
        chat_type="dm",
        user_id="user-88532",
    )

    # Seed the profile database through the same scope-aware store that the
    # gateway uses.  The root database must remain empty.
    token = set_hermes_home_override(str(profile))
    try:
        entry = store.get_or_create_session(source)
        store.append_to_transcript(
            entry.session_id,
            {"role": "user", "content": "Earlier message"},
        )
        store.append_to_transcript(
            entry.session_id,
            {"role": "assistant", "content": "Earlier answer"},
        )
    finally:
        reset_hermes_home_override(token)

    runner = object.__new__(GatewayRunner)
    runner.config = store.config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db_pinned = None
    runner._resolve_profile_home_for_source = lambda _source: profile
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._is_telegram_topic_lane = lambda _source: False
    runner._set_session_env = lambda _context: None
    runner._clear_session_env = lambda _tokens: None
    runner._pinned_session_context_prompt = lambda _context, _redact, _key: ""
    runner._mark_durable_active_turn = AsyncMock(return_value=False)
    runner._is_session_run_current = lambda _key, _generation: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._adapter_for_source = lambda _source: None
    runner._bind_adapter_run_generation = lambda *_args: None
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._drain_watch_notifications = AsyncMock()
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._clear_restart_failure_count = AsyncMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "New answer",
            "messages": [
                {"role": "user", "content": "Earlier message"},
                {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "Current message"},
                {"role": "assistant", "content": "New answer"},
            ],
            "history_offset": 2,
            "tools": [],
            "api_calls": 1,
            "last_prompt_tokens": 0,
            "agent_persisted": False,
        }
    )
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})

    event = MessageEvent(text="Current message", source=source, message_id="msg-88532")
    result = await runner._handle_message_with_agent(event, source, "quick-88532", 1)

    assert result == "New answer"
    assert [
        (message["role"], message["content"])
        for message in runner._run_agent.call_args.kwargs["history"]
    ] == [
        ("user", "Earlier message"),
        ("assistant", "Earlier answer"),
    ]
    token = set_hermes_home_override(str(profile))
    try:
        profile_messages = store.load_transcript(entry.session_id)
    finally:
        reset_hermes_home_override(token)
    assert [message["content"] for message in profile_messages if "content" in message] == [
        "Earlier message",
        "Earlier answer",
        "Current message",
        "New answer",
    ]
    assert _session_ids(profile / "state.db") == {entry.session_id}
    assert _session_ids(root / "state.db") == set()
    assert Path(get_hermes_home()) == root


def test_inbound_turn_is_a_pass_through_without_multiplexing(multiplex_homes):
    """Single-profile gateways never enter the scope."""
    import asyncio

    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home

    root, profile = multiplex_homes

    class _Cfg:
        multiplex_profiles = False

    runner = object.__new__(GatewayRunner)
    runner.config = _Cfg()

    def _boom(source):
        raise AssertionError("profile home must not be resolved when multiplexing is off")

    runner._resolve_profile_home_for_source = _boom

    seen = {}

    async def _inner(event, source, _quick_key, run_generation):
        seen["home"] = Path(get_hermes_home())
        return "ok"

    runner._handle_message_with_agent_inner = _inner

    assert asyncio.run(
        runner._handle_message_with_agent(object(), object(), "k", 1)
    ) == "ok"
    assert seen["home"] == root
