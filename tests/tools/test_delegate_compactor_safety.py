"""Compaction-path safety for delegated children.

`_compress_context` (now a forwarder to
`agent.conversation_compression.compress_context`) must rebuild the system
prompt via `_build_system_prompt` on every compaction so a child emits a
coherent baseline system string on its next turn.

Note: the earlier "delegation shape mirror" invariants (child-compactor
`protect_first_n` bumping and `_frozen_system_prompt` reuse) were removed
from the source in the upstream sync that landed on live-config
(`tools/delegate_tool.py` no longer touches the child compactor;
`run_agent.py` no longer defines `capture_delegation_snapshot` or
`_frozen_system_prompt`; `compress_context` unconditionally rebuilds the
prompt). The tests that asserted those invariants were orphaned and have
been removed. A sibling cleanup is tracked for
`tests/tools/test_delegate_shape_mirror.py`.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class TestCompressContextRebuildsPrompt(unittest.TestCase):
    """_compress_context must rebuild the system prompt on compaction."""

    def test_rebuild_happens_on_compaction(self):
        """Compaction invokes _build_system_prompt and returns its result."""
        from run_agent import AIAgent

        agent = MagicMock(spec=AIAgent)
        agent.compression_enabled = True
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

        agent._build_system_prompt.assert_called_once_with("parent_msg")
        self.assertEqual(new_prompt, "REBUILT_STRING")


if __name__ == "__main__":
    unittest.main()
