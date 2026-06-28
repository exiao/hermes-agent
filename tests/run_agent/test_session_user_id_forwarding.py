"""Regression: gateway session rows must carry the platform user_id.

Two create sites previously wrote user_id=NULL:
  - run_agent.py:_ensure_db_session (hardcoded user_id=None)
  - agent/conversation_compression.py rotation create_session (never forwarded)

list_unlinked_telegram_sessions_for_user filters
``WHERE source='telegram' AND user_id = ?``, so NULL rows are undiscoverable
and a returning Telegram user can't resume a prior unlinked DM session.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def _make_agent(session_db, *, user_id, platform, session_id):
    # Stub get_model_context_length so ContextCompressor init never probes the
    # OpenRouter endpoint / fetches model metadata. Without this, a cold CI
    # worker (no warm ~/.hermes cache) makes a live network call here, adding
    # long timeouts or failing under socket-blocking test setups. We only care
    # about the session-row user_id, not the real context window.
    with (
        patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=200_000,
        ),
    ):
        from run_agent import AIAgent

        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            platform=platform,
            user_id=user_id,
            skip_context_files=True,
            skip_memory=True,
        )


def test_ensure_db_session_persists_user_id_for_telegram_discovery():
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "state.db")
        agent = _make_agent(
            db, user_id="user-123", platform="telegram", session_id="tg-session"
        )

        agent._ensure_db_session()

        # Row carries the user_id (not NULL).
        rows = db._conn.execute(
            "SELECT user_id FROM sessions WHERE id = ?", ("tg-session",)
        ).fetchall()
        assert rows and rows[0]["user_id"] == "user-123"

        # And it is therefore discoverable by the unlinked-session lookup.
        found = db.list_unlinked_telegram_sessions_for_user(
            chat_id="chat-1", user_id="user-123"
        )
        assert any(r["id"] == "tg-session" for r in found)


def test_compression_rotation_forwards_user_id():
    """The real compress_context rotation path must carry user_id onto the child.

    Drives the actual ``agent/conversation_compression.compress_context``
    rotation branch (in_place=False) with a stubbed compressor, so the fix at
    the rotation ``create_session`` call site is genuinely exercised — not
    simulated.
    """
    from agent.conversation_compression import compress_context
    from hermes_state import SessionDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db = SessionDB(db_path=Path(tmpdir) / "state.db")
        agent = _make_agent(
            db, user_id="user-456", platform="telegram", session_id="tg-parent"
        )
        agent._ensure_db_session()

        # Force the legacy rotation path (the in-place path keeps one id and
        # never re-creates a row, so user_id forwarding is moot there).
        agent.compression_in_place = False

        # Stub the compressor: return a short, valid compacted transcript.
        # Carries the few attributes the post-compaction bookkeeping reads;
        # the rotation create_session (the fix site) runs before that block.
        class _StubCompressor:
            _last_compress_aborted = False
            _last_summary_error = None
            compression_count = 1

            def compress(self, messages, **kwargs):
                return [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}]

        agent.context_compressor = _StubCompressor()
        old_session_id = agent.session_id

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        compress_context(agent, messages, system_message="sys")

        # The session id rotated to a fresh child.
        assert agent.session_id != old_session_id

        rows = db._conn.execute(
            "SELECT user_id FROM sessions WHERE id = ?", (agent.session_id,)
        ).fetchall()
        assert rows and rows[0]["user_id"] == "user-456"

        # Child is discoverable for the returning Telegram user.
        found = db.list_unlinked_telegram_sessions_for_user(
            chat_id="chat-1", user_id="user-456"
        )
        assert any(r["id"] == agent.session_id for r in found)
