"""W13 — compactor safety for shape-mirrored children.

These tests verify two invariants that keep the cache anchor from being
wiped on child-side compaction:

1. `_build_child_agent` bumps the child compactor's `protect_first_n` to
   cover the full mirrored prefix so the summarizer can never touch it.

2. `_compress_context` reuses `_frozen_system_prompt` instead of
   rebuilding — otherwise the child would emit its *own* baseline system
   string on turn N+1, evicting the Anthropic cache entry.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ── Fixture: a fake snapshot that mimics AIAgent.capture_delegation_snapshot
def _fake_snapshot(n_prefix_msgs=10):
    return {
        "system": "PARENT_SYSTEM_VERBATIM",
        "tools": [],
        "model": "claude-sonnet-4-20250514",
        "messages_prefix": [
            {"role": "user", "content": f"msg {i}"}
            if i % 2 == 0
            else {"role": "assistant", "content": f"reply {i}"}
            for i in range(n_prefix_msgs)
        ],
    }


class TestProtectFirstNBump(unittest.TestCase):
    """_build_child_agent must bump the child compactor's head-protection."""

    def _run_build_with_snapshot(self, *, prefix_len, initial_protect_n=3):
        """Invoke _build_child_agent with a mocked AIAgent constructor and
        return the child (a MagicMock) after the snapshot-attach block runs.
        """
        from tools.delegate_tool import _build_child_agent

        # Build the mock child returned by the AIAgent constructor.
        child = MagicMock()
        compactor = MagicMock()
        compactor.protect_first_n = initial_protect_n
        child.context_compressor = compactor
        # Parent-auth surface that _build_child_agent inspects.
        child._active_children = []
        child.valid_tool_names = set()
        child.tools = []

        parent = MagicMock()
        parent._delegation_shape_snapshot = _fake_snapshot(prefix_len)
        parent.model = "claude-sonnet-4-20250514"
        parent.base_url = ""
        parent.api_key = "test-key"
        parent.provider = "anthropic"
        parent.api_mode = None
        parent.acp_command = None
        parent.acp_args = []
        parent.max_tokens = 4096
        parent.reasoning_config = None
        parent.prefill_messages = None
        parent.quiet_mode = True
        parent.platform = "cli"
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent.enabled_toolsets = []
        parent.valid_tool_names = set()
        parent._active_children = []
        parent._active_children_lock = None
        parent._session_db = None
        parent.session_id = "parent-sess"
        parent._delegate_depth = 0
        parent._print_fn = None

        with patch("run_agent.AIAgent", return_value=child), \
             patch("tools.delegate_tool._build_child_progress_callback",
                   return_value=None), \
             patch("tools.delegate_tool._resolve_child_credential_pool",
                   return_value=None):
            result = _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=5,
                parent_agent=parent,
            )
        return result, compactor

    def test_protect_first_n_bumped_to_cover_prefix_plus_goal(self):
        """A 10-message prefix must bump protect_first_n to at least 11."""
        child, compactor = self._run_build_with_snapshot(prefix_len=10)
        self.assertGreaterEqual(compactor.protect_first_n, 11,
            "protect_first_n must cover all 10 prefix msgs + the goal")

    def test_protect_first_n_never_shrinks(self):
        """If the compactor already protects more than the prefix, keep it."""
        child, compactor = self._run_build_with_snapshot(
            prefix_len=2, initial_protect_n=50)
        self.assertEqual(compactor.protect_first_n, 50,
            "protect_first_n must not shrink below its prior value")

    def test_no_snapshot_means_no_compactor_mutation(self):
        """Without a delegation snapshot, don't touch the child compactor."""
        from tools.delegate_tool import _build_child_agent

        child = MagicMock()
        compactor = MagicMock()
        compactor.protect_first_n = 3
        child.context_compressor = compactor
        child._active_children = []
        child.valid_tool_names = set()
        child.tools = []

        parent = MagicMock()
        parent._delegation_shape_snapshot = None  # ← no snapshot
        parent.model = "claude-sonnet-4-20250514"
        parent.base_url = ""
        parent.api_key = "test-key"
        parent.provider = "anthropic"
        parent.api_mode = None
        parent.acp_command = None
        parent.acp_args = []
        parent.max_tokens = 4096
        parent.reasoning_config = None
        parent.prefill_messages = None
        parent.quiet_mode = True
        parent.platform = "cli"
        parent.providers_allowed = None
        parent.providers_ignored = None
        parent.providers_order = None
        parent.provider_sort = None
        parent.enabled_toolsets = []
        parent.valid_tool_names = set()
        parent._active_children = []
        parent._active_children_lock = None
        parent._session_db = None
        parent.session_id = "parent-sess"
        parent._delegate_depth = 0
        parent._print_fn = None

        with patch("run_agent.AIAgent", return_value=child), \
             patch("tools.delegate_tool._build_child_progress_callback",
                   return_value=None), \
             patch("tools.delegate_tool._resolve_child_credential_pool",
                   return_value=None):
            _build_child_agent(
                task_index=0,
                goal="test",
                context=None,
                toolsets=None,
                model=None,
                max_iterations=5,
                parent_agent=parent,
            )
        # Unchanged.
        self.assertEqual(compactor.protect_first_n, 3)


class TestFrozenSystemPromptOnCompaction(unittest.TestCase):
    """_compress_context must reuse _frozen_system_prompt when set."""

    def test_frozen_prompt_reused_after_compaction(self):
        """After compression, _cached_system_prompt must equal the frozen value."""
        from run_agent import AIAgent

        # Build a thin AIAgent-like object that exposes just the methods
        # _compress_context needs.  We don't want to spin up a full agent —
        # we want to assert the branch behavior.
        agent = MagicMock(spec=AIAgent)
        agent._frozen_system_prompt = "PARENT_SYSTEM_VERBATIM"
        agent._cached_system_prompt = None
        agent._memory_manager = None
        agent._session_db = None
        agent.session_id = "s1"
        agent.model = "claude-sonnet-4-20250514"
        agent.log_prefix = ""
        agent._todo_store = MagicMock()
        agent._todo_store.format_for_injection.return_value = ""
        agent.context_compressor = MagicMock()
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "kept"}
        ]
        agent.context_compressor.compression_count = 0
        agent.context_compressor.last_prompt_tokens = 0
        agent.context_compressor.last_completion_tokens = 0
        agent.flush_memories = MagicMock()
        agent.commit_memory_session = MagicMock()
        agent._invalidate_system_prompt = MagicMock()
        agent._build_system_prompt = MagicMock(return_value="REBUILT_STRING")
        agent._session_id = "s1"
        agent.session_log_file = Path("/tmp/fake.json")
        agent.logs_dir = Path("/tmp")

        # Call the real _compress_context on the mock — we need the real
        # method, bound to our mock state.
        compressed, new_prompt = AIAgent._compress_context(
            agent,
            messages=[{"role": "user", "content": "m"}],
            system_message=None,
            approx_tokens=1000,
        )

        # Frozen prompt must be used as the new system prompt.
        self.assertEqual(new_prompt, "PARENT_SYSTEM_VERBATIM")
        self.assertEqual(agent._cached_system_prompt, "PARENT_SYSTEM_VERBATIM")
        # _build_system_prompt must NOT have been called — the whole point
        # of the shape mirror is that the frozen value is authoritative.
        agent._build_system_prompt.assert_not_called()

    def test_no_frozen_prompt_means_rebuild_happens(self):
        """Without a frozen prompt, the original rebuild path still runs."""
        from run_agent import AIAgent

        agent = MagicMock(spec=AIAgent)
        agent._frozen_system_prompt = None  # ← not a child
        agent._cached_system_prompt = None
        agent._memory_manager = None
        agent._session_db = None
        agent.session_id = "s1"
        agent.model = "claude-sonnet-4-20250514"
        agent.log_prefix = ""
        agent._todo_store = MagicMock()
        agent._todo_store.format_for_injection.return_value = ""
        agent.context_compressor = MagicMock()
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "kept"}
        ]
        agent.context_compressor.compression_count = 0
        agent.context_compressor.last_prompt_tokens = 0
        agent.context_compressor.last_completion_tokens = 0
        agent.flush_memories = MagicMock()
        agent.commit_memory_session = MagicMock()
        agent._invalidate_system_prompt = MagicMock()
        agent._build_system_prompt = MagicMock(return_value="REBUILT_STRING")
        agent.session_log_file = Path("/tmp/fake.json")
        agent.logs_dir = Path("/tmp")

        compressed, new_prompt = AIAgent._compress_context(
            agent,
            messages=[{"role": "user", "content": "m"}],
            system_message="parent_msg",
            approx_tokens=1000,
        )
        # Standard path: rebuild was invoked.
        agent._build_system_prompt.assert_called_once_with("parent_msg")
        self.assertEqual(new_prompt, "REBUILT_STRING")


if __name__ == "__main__":
    unittest.main()
