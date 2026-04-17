#!/usr/bin/env python3
"""
Shape-mirror delegation tests (W10).

Verifies the contract between AIAgent.capture_delegation_snapshot (parent side)
and tools.delegate_tool._build_child_agent / _run_single_child (child side)
that underpins Anthropic prompt-cache reuse across the parent→child boundary.

The contract:
  1. Parent's `capture_delegation_snapshot(messages)` returns the fields needed
     to reconstruct the parent's next API request: system, tools, messages_prefix,
     model, plus runtime knobs.  messages_prefix and tools are deep-copied so
     later parent mutation can't bleed into the child's replay.
  2. `_build_child_agent` attaches that snapshot to the child as
     `_parent_shape_snapshot` and freezes the child's system to parent's via
     `_frozen_system_prompt`.
  3. `_run_single_child` replays the snapshot's `messages_prefix` via
     `run_conversation(conversation_history=...)` so the child's first API
     call byte-matches what the parent would have sent.

Run with: python -m pytest tests/tools/test_delegate_shape_mirror.py -v
"""

import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _build_child_agent,
    _run_single_child,
)


def _make_mock_parent_with_snapshot(snapshot=None, depth=0):
    """Mock parent carrying a delegation shape snapshot."""
    parent = MagicMock()
    parent.base_url = "https://api.anthropic.com"
    parent.api_key = "sk-ant-test"
    parent.provider = "anthropic"
    parent.api_mode = "anthropic_messages"
    parent.model = "claude-sonnet-4-20250514"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = None
    parent.valid_tool_names = []
    parent.reasoning_config = None
    parent.acp_command = None
    parent.acp_args = []
    parent._delegation_shape_snapshot = snapshot
    return parent


class TestCaptureDelegationSnapshot(unittest.TestCase):
    """AIAgent.capture_delegation_snapshot — parent-side contract."""

    def _make_fake_agent(self, *, system="SYS", ephemeral=None, tools=None,
                         messages=None, model="claude-sonnet-4",
                         reasoning_config=None):
        """Build a minimal object with the attributes capture reads.

        We import the unbound method from AIAgent and call it with our fake
        `self` so we don't need a real AIAgent construction (which pulls in
        network clients, tool registries, etc.).
        """
        from run_agent import AIAgent
        fake = SimpleNamespace(
            _cached_system_prompt=system,
            ephemeral_system_prompt=ephemeral,
            tools=tools or [],
            model=model,
            max_tokens=4096,
            reasoning_config=reasoning_config,
            api_mode="anthropic_messages",
            provider="anthropic",
            base_url="https://api.anthropic.com",
        )
        # Bind method with our fake self.
        return AIAgent.capture_delegation_snapshot(fake, messages or [])

    def test_returns_dict_with_required_keys(self):
        snap = self._make_fake_agent()
        required = {
            "model", "system", "tools", "messages_prefix",
            "max_tokens", "reasoning_config", "api_mode",
            "provider", "base_url",
        }
        self.assertTrue(required.issubset(set(snap.keys())),
                        f"Missing keys: {required - set(snap.keys())}")

    def test_system_is_cached_prompt_when_no_ephemeral(self):
        snap = self._make_fake_agent(system="BASE_SYSTEM", ephemeral=None)
        self.assertEqual(snap["system"], "BASE_SYSTEM")

    def test_system_concatenates_ephemeral_overlay(self):
        """Parent's effective system == cached + ephemeral; child must match."""
        snap = self._make_fake_agent(system="BASE", ephemeral="EPHEMERAL_OVERLAY")
        self.assertIn("BASE", snap["system"])
        self.assertIn("EPHEMERAL_OVERLAY", snap["system"])
        # Overlay comes after base (matters for cache boundary placement).
        self.assertLess(snap["system"].index("BASE"),
                        snap["system"].index("EPHEMERAL_OVERLAY"))

    def test_messages_prefix_is_deep_copy(self):
        """Later parent mutation must not bleed into child's replay."""
        original = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        snap = self._make_fake_agent(messages=original)
        # Mutate the original; snapshot must be unaffected.
        original[0]["content"][0]["text"] = "MUTATED"
        original.append({"role": "assistant", "content": "injected"})
        self.assertEqual(snap["messages_prefix"][0]["content"][0]["text"], "hi")
        self.assertEqual(len(snap["messages_prefix"]), 1)

    def test_tools_is_deep_copy(self):
        original = [{"name": "foo", "description": "orig"}]
        snap = self._make_fake_agent(tools=original)
        original[0]["description"] = "MUTATED"
        self.assertEqual(snap["tools"][0]["description"], "orig")

    def test_empty_messages_yields_empty_prefix(self):
        snap = self._make_fake_agent(messages=[])
        self.assertEqual(snap["messages_prefix"], [])

    def test_system_empty_string_when_nothing_cached(self):
        """If the parent hasn't cached a system yet, snapshot.system is ''."""
        from run_agent import AIAgent
        fake = SimpleNamespace(
            _cached_system_prompt=None,
            ephemeral_system_prompt=None,
            tools=[],
            model="m",
            max_tokens=None,
            reasoning_config=None,
            api_mode=None,
            provider=None,
            base_url=None,
        )
        snap = AIAgent.capture_delegation_snapshot(fake, [])
        self.assertEqual(snap["system"], "")


class TestBuildChildAttachesSnapshot(unittest.TestCase):
    """_build_child_agent must forward snapshot onto the child."""

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_snapshot_attached_when_parent_has_one(self, MockAgent, mock_cfg):
        mock_cfg.return_value = {"max_iterations": 50, "reasoning_effort": ""}
        child_mock = MagicMock()
        MockAgent.return_value = child_mock

        snapshot = {
            "model": "claude-sonnet-4",
            "system": "PARENT_SYSTEM_VERBATIM",
            "tools": [{"name": "foo"}],
            "messages_prefix": [
                {"role": "user", "content": [{"type": "text", "text": "turn 1"}]},
                {"role": "assistant", "content": "response"},
            ],
            "max_tokens": 4096,
            "reasoning_config": None,
            "api_mode": "anthropic_messages",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
        }
        parent = _make_mock_parent_with_snapshot(snapshot=snapshot)

        child = _build_child_agent(
            task_index=0, goal="finish work", context=None, toolsets=None,
            model=None, max_iterations=50, parent_agent=parent,
        )

        # Child carries the snapshot and a frozen system matching parent.
        self.assertIs(child._parent_shape_snapshot, snapshot)
        self.assertEqual(child._frozen_system_prompt, "PARENT_SYSTEM_VERBATIM")

    @patch("tools.delegate_tool._load_config")
    @patch("run_agent.AIAgent")
    def test_no_snapshot_no_attachment(self, MockAgent, mock_cfg):
        """When parent has no snapshot, child must not carry spurious fields."""
        mock_cfg.return_value = {"max_iterations": 50, "reasoning_effort": ""}
        child_mock = MagicMock(spec_set=[
            "_print_fn", "_delegate_depth", "_credential_pool",
            "_delegate_saved_tool_names",
        ])
        MockAgent.return_value = child_mock
        parent = _make_mock_parent_with_snapshot(snapshot=None)

        child = _build_child_agent(
            task_index=0, goal="task", context=None, toolsets=None,
            model=None, max_iterations=50, parent_agent=parent,
        )

        # spec_set means any unexpected attribute access (like
        # child._parent_shape_snapshot = ...) would have raised.  If we got
        # here, the shape-attach block correctly no-op'd.
        self.assertFalse(hasattr(child, "_parent_shape_snapshot"))


class TestRunSingleChildReplaysPrefix(unittest.TestCase):
    """_run_single_child must pass messages_prefix as conversation_history."""

    def test_replays_prefix_when_snapshot_present(self):
        prefix = [
            {"role": "user", "content": [{"type": "text", "text": "turn 1"}]},
            {"role": "assistant", "content": "response 1"},
        ]
        snapshot = {
            "system": "SYS",
            "messages_prefix": prefix,
            "tools": [],
            "model": "m",
        }
        child = MagicMock()
        child._parent_shape_snapshot = snapshot
        child._credential_pool = None
        child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
        }
        child.get_activity_summary.return_value = {
            "current_tool": None,
            "api_call_count": 1,
            "max_iterations": 50,
            "last_activity_desc": "",
        }

        parent = MagicMock()
        parent._touch_activity = MagicMock()

        _run_single_child(task_index=0, goal="do the thing",
                          child=child, parent_agent=parent)

        child.run_conversation.assert_called_once()
        call_kwargs = child.run_conversation.call_args.kwargs
        self.assertEqual(call_kwargs["user_message"], "do the thing")
        self.assertEqual(call_kwargs["conversation_history"], prefix)

    def test_falls_back_to_bare_run_when_no_snapshot(self):
        """Backward-compat: no snapshot → no conversation_history arg."""
        child = MagicMock()
        # Ensure the attribute lookup returns None (the dispatch branch).
        child._parent_shape_snapshot = None
        child._credential_pool = None
        child.run_conversation.return_value = {
            "final_response": "ok",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
        }
        child.get_activity_summary.return_value = {
            "current_tool": None, "api_call_count": 1,
            "max_iterations": 50, "last_activity_desc": "",
        }

        parent = MagicMock()
        parent._touch_activity = MagicMock()

        _run_single_child(task_index=0, goal="work",
                          child=child, parent_agent=parent)

        child.run_conversation.assert_called_once()
        call_kwargs = child.run_conversation.call_args.kwargs
        self.assertEqual(call_kwargs["user_message"], "work")
        self.assertNotIn("conversation_history", call_kwargs)


class TestShapeMirrorEndToEnd(unittest.TestCase):
    """The headline contract: child's replayed prefix == parent's snapshot."""

    def test_prefix_byte_equal_after_full_round_trip(self):
        """Parent captures → _build_child_agent forwards → _run_single_child
        replays — the messages_prefix received by child.run_conversation
        must be byte-equal to what the parent captured."""
        from run_agent import AIAgent

        parent_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "initial user turn",
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "assistant answer"},
            ]},
        ]
        fake_parent_self = SimpleNamespace(
            _cached_system_prompt="PARENT_SYSTEM",
            ephemeral_system_prompt=None,
            tools=[{"name": "terminal", "description": "run cmd"}],
            model="claude-sonnet-4",
            max_tokens=4096,
            reasoning_config=None,
            api_mode="anthropic_messages",
            provider="anthropic",
            base_url="https://api.anthropic.com",
        )
        snapshot = AIAgent.capture_delegation_snapshot(
            fake_parent_self, parent_messages)
        captured_prefix = copy.deepcopy(snapshot["messages_prefix"])

        # Now simulate the delegate_tool side: child receives snapshot.
        with patch("tools.delegate_tool._load_config",
                   return_value={"max_iterations": 50, "reasoning_effort": ""}):
            with patch("run_agent.AIAgent") as MockAgent:
                child_mock = MagicMock()
                MockAgent.return_value = child_mock
                parent = _make_mock_parent_with_snapshot(snapshot=snapshot)
                child = _build_child_agent(
                    task_index=0, goal="continue work", context=None,
                    toolsets=None, model=None, max_iterations=50,
                    parent_agent=parent,
                )
                # Now route through _run_single_child.
                child._credential_pool = None
                child.run_conversation.return_value = {
                    "final_response": "x", "completed": True,
                    "interrupted": False, "api_calls": 1,
                }
                child.get_activity_summary.return_value = {
                    "current_tool": None, "api_call_count": 1,
                    "max_iterations": 50, "last_activity_desc": "",
                }
                _run_single_child(
                    task_index=0, goal="continue work",
                    child=child, parent_agent=parent,
                )

        # The prefix handed to run_conversation must equal what parent captured.
        call_kwargs = child.run_conversation.call_args.kwargs
        self.assertEqual(call_kwargs["conversation_history"], captured_prefix)
        # And child's system is parent's verbatim.
        self.assertEqual(child._frozen_system_prompt, "PARENT_SYSTEM")


if __name__ == "__main__":
    unittest.main()
