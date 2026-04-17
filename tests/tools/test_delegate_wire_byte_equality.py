#!/usr/bin/env python3
"""
W11 — integration byte-equality test for shape-mirror delegation.

Unit tests in test_delegate_shape_mirror.py pin the contract between
AIAgent.capture_delegation_snapshot and tools.delegate_tool.  This file goes
one layer deeper: it feeds both the "what parent would have sent" branch and
the "what child actually sends via the mirror" branch through the *real*
wire-payload builder (agent.anthropic_adapter.build_anthropic_kwargs) and
asserts that the resulting Anthropic request kwargs are byte-identical across
the prefix that the prompt cache keys on.

If any future change drifts the two paths apart — a subtle re-ordering of
tools, a new cache marker placement, a system-prompt transform applied on
one side but not the other — this test catches it before it burns Max quota
on a cache miss.

Prefix equality is defined as identity of:
  - `system` (string or list-of-blocks)
  - `tools` (list, including order)
  - the first N messages of `messages`, where N == len(parent snapshot prefix)
Everything AFTER the prefix (the new user goal, assistant response, etc.)
may differ — that's where the cache miss and child divergence begin.

Run:  python -m pytest tests/tools/test_delegate_wire_byte_equality.py -v

See ~/.hermes/plans/hermes-patches/delegation-wire-byte-equality-test.md
"""

import copy
import unittest
from types import SimpleNamespace

from agent.anthropic_adapter import build_anthropic_kwargs


def _parent_would_send(*, system, tools, messages_prefix, next_user_goal,
                       model, max_tokens, reasoning_config):
    """Simulate what the parent's *next* turn would look like if it had
    chosen to run the work itself instead of delegating.

    The parent, if not delegating, would:
      - keep the same system string
      - keep the same tools
      - append the new user message (the goal) to messages_prefix
      - call build_anthropic_kwargs with all of that
    """
    messages = copy.deepcopy(messages_prefix) + [
        {"role": "user", "content": next_user_goal},
    ]
    # Parent's system prompt enters build_anthropic_kwargs as a message of
    # role "system" (that's how convert_messages_to_anthropic extracts it).
    # capture_delegation_snapshot's `system` field IS that string.  We have
    # to re-inject it as a system message for build_anthropic_kwargs to
    # find it.
    if system:
        messages = [{"role": "system", "content": system}] + messages
    return build_anthropic_kwargs(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        reasoning_config=reasoning_config,
    )


def _child_actually_sends(*, snapshot, next_user_goal):
    """Simulate what the child sends on its very first API call under the
    shape-mirror regime.

    In the real code path:
      1. _build_child_agent attaches snapshot as _parent_shape_snapshot and
         writes snapshot['system'] into child._frozen_system_prompt.
      2. _run_single_child invokes child.run_conversation(
             user_message=goal,
             conversation_history=snapshot['messages_prefix'],
         ).
      3. Inside run_conversation, _cached_system_prompt is set from
         _frozen_system_prompt (run_agent.py line ~8462), and the message
         list is [*conversation_history, {user: goal}].
      4. That list plus the frozen system prompt goes into
         build_anthropic_kwargs via _build_api_kwargs.

    This helper reproduces steps 3+4 directly against build_anthropic_kwargs
    so we exercise the same wire builder the live code uses.
    """
    system = snapshot["system"]
    tools = snapshot["tools"]
    messages = copy.deepcopy(snapshot["messages_prefix"]) + [
        {"role": "user", "content": next_user_goal},
    ]
    if system:
        messages = [{"role": "system", "content": system}] + messages
    return build_anthropic_kwargs(
        model=snapshot["model"],
        messages=messages,
        tools=tools,
        max_tokens=snapshot["max_tokens"],
        reasoning_config=snapshot["reasoning_config"],
    )


def _capture_via_real_method(*, system, ephemeral=None, tools=None,
                             messages=None, model="claude-sonnet-4-20250514",
                             max_tokens=4096, reasoning_config=None):
    """Drive AIAgent.capture_delegation_snapshot with a SimpleNamespace self.

    Uses the real method (not a copy) so if capture_delegation_snapshot ever
    changes shape, this test catches it.
    """
    from run_agent import AIAgent
    fake = SimpleNamespace(
        _cached_system_prompt=system,
        ephemeral_system_prompt=ephemeral,
        tools=tools or [],
        model=model,
        max_tokens=max_tokens,
        reasoning_config=reasoning_config,
        api_mode="anthropic_messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
    )
    return AIAgent.capture_delegation_snapshot(fake, messages or [])


class TestWireBytePrefixEquality(unittest.TestCase):
    """Parent's what-I-would-have-sent ≡ child's what-I-do-send, up to the
    boundary of the new user goal."""

    def test_simple_text_conversation_prefix_equal(self):
        """Basic case: one user turn, one assistant turn, then delegate."""
        parent_system = "You are a helpful assistant. Keep answers concise."
        parent_tools = [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Execute a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            },
        ]
        parent_messages = [
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4."},
        ]
        goal = "Now compute the integral of x from 0 to 1."

        # --- Parent side: capture, then reconstruct wire payload ---
        snapshot = _capture_via_real_method(
            system=parent_system, tools=parent_tools,
            messages=parent_messages,
        )
        parent_kwargs = _parent_would_send(
            system=parent_system,
            tools=parent_tools,
            messages_prefix=parent_messages,
            next_user_goal=goal,
            model=snapshot["model"],
            max_tokens=snapshot["max_tokens"],
            reasoning_config=snapshot["reasoning_config"],
        )

        # --- Child side: replay via snapshot ---
        child_kwargs = _child_actually_sends(snapshot=snapshot,
                                             next_user_goal=goal)

        # Byte-equality on the cache-critical prefix fields.
        self.assertEqual(parent_kwargs.get("system"),
                         child_kwargs.get("system"),
                         "system prompt drift between parent and child")
        self.assertEqual(parent_kwargs.get("tools"),
                         child_kwargs.get("tools"),
                         "tools list drift between parent and child")
        self.assertEqual(parent_kwargs.get("model"),
                         child_kwargs.get("model"),
                         "model drift between parent and child")

        # The prefix of messages (everything before the trailing user goal)
        # must be identical.  The goal message itself is identical by
        # construction because both paths append the same goal.
        p_msgs = parent_kwargs["messages"]
        c_msgs = child_kwargs["messages"]
        self.assertEqual(len(p_msgs), len(c_msgs))
        for i, (p, c) in enumerate(zip(p_msgs, c_msgs)):
            self.assertEqual(p, c, f"message {i} drift: {p!r} vs {c!r}")

    def test_multi_turn_prefix_with_tool_use_blocks(self):
        """Longer prefix with a tool_use → tool_result pair.

        This is the shape that matters in practice: most delegations happen
        *after* the parent has done some tool work.  The cache prefix is
        only interesting if it contains real work.
        """
        parent_system = "You are an SRE copilot."
        parent_tools = [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            },
        ]
        parent_messages = [
            {"role": "user", "content": "Check if nginx is running."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "toolu_01abc",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"cmd": "systemctl status nginx"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_01abc",
                "content": "active (running)",
            },
            {"role": "assistant", "content": "nginx is active."},
        ]
        goal = "Now delegate: investigate the last 500 lines of the access log."

        snapshot = _capture_via_real_method(
            system=parent_system, tools=parent_tools,
            messages=parent_messages,
        )
        parent_kwargs = _parent_would_send(
            system=parent_system,
            tools=parent_tools,
            messages_prefix=parent_messages,
            next_user_goal=goal,
            model=snapshot["model"],
            max_tokens=snapshot["max_tokens"],
            reasoning_config=snapshot["reasoning_config"],
        )
        child_kwargs = _child_actually_sends(snapshot=snapshot,
                                             next_user_goal=goal)

        self.assertEqual(parent_kwargs.get("system"),
                         child_kwargs.get("system"))
        self.assertEqual(parent_kwargs.get("tools"),
                         child_kwargs.get("tools"))
        self.assertEqual(parent_kwargs["messages"],
                         child_kwargs["messages"])

    def test_cache_control_markers_round_trip_identically(self):
        """cache_control markers on system blocks MUST survive the snapshot
        round-trip unchanged — they are the reason the mirror exists."""
        parent_system_blocks = [
            {"type": "text", "text": "You are Hermes."},
            {
                "type": "text",
                "text": "Long secondary context that is worth caching.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            },
        ]
        # capture_delegation_snapshot reads _cached_system_prompt, which in
        # production is a string (built at turn start).  cache_control on
        # the system side is applied by the adapter during convert_messages_.
        # We test that pre-existing cache_control markers on message content
        # survive the round-trip.
        parent_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "turn 1",
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "answer 1"},
            ]},
        ]
        parent_system = "You are Hermes."  # cached_system_prompt is a string

        snapshot = _capture_via_real_method(
            system=parent_system,
            tools=[],
            messages=parent_messages,
        )
        goal = "next goal"
        parent_kwargs = _parent_would_send(
            system=parent_system,
            tools=[],
            messages_prefix=parent_messages,
            next_user_goal=goal,
            model=snapshot["model"],
            max_tokens=snapshot["max_tokens"],
            reasoning_config=snapshot["reasoning_config"],
        )
        child_kwargs = _child_actually_sends(snapshot=snapshot,
                                             next_user_goal=goal)

        # Find the user message with the cache_control marker on both sides.
        def _find_marked(msgs):
            for m in msgs:
                c = m.get("content")
                if isinstance(c, list):
                    for blk in c:
                        if isinstance(blk, dict) and blk.get("cache_control"):
                            return blk
            return None

        p_marked = _find_marked(parent_kwargs["messages"])
        c_marked = _find_marked(child_kwargs["messages"])
        self.assertIsNotNone(p_marked, "parent lost cache_control marker")
        self.assertIsNotNone(c_marked, "child lost cache_control marker")
        self.assertEqual(p_marked, c_marked,
                         "cache_control marker drifted across mirror")

    def test_ephemeral_overlay_mirrored_in_system(self):
        """When parent has ephemeral_system_prompt, child's system must
        include it too — otherwise system strings drift and cache misses."""
        snapshot = _capture_via_real_method(
            system="BASE_SYSTEM",
            ephemeral="ACTIVE_EPHEMERAL_OVERLAY",
            tools=[],
            messages=[],
        )
        self.assertIn("BASE_SYSTEM", snapshot["system"])
        self.assertIn("ACTIVE_EPHEMERAL_OVERLAY", snapshot["system"])

        goal = "go"
        # Parent's wire payload uses cached + overlay (that's what it sends).
        effective_parent_system = (
            "BASE_SYSTEM" + "\n\n" + "ACTIVE_EPHEMERAL_OVERLAY"
        ).strip()
        parent_kwargs = _parent_would_send(
            system=effective_parent_system,
            tools=[],
            messages_prefix=[],
            next_user_goal=goal,
            model=snapshot["model"],
            max_tokens=snapshot["max_tokens"],
            reasoning_config=snapshot["reasoning_config"],
        )
        child_kwargs = _child_actually_sends(snapshot=snapshot,
                                             next_user_goal=goal)

        self.assertEqual(parent_kwargs.get("system"),
                         child_kwargs.get("system"),
                         "ephemeral overlay not mirrored — cache will miss")

    def test_tools_deep_copy_isolates_parent_mutation(self):
        """If the parent mutates its tools AFTER capture (e.g. toolset
        rotation between turns), the snapshot's deep copy must protect the
        child's prefix.  Concretely: the child's wire `tools` must match
        the parent's AT-THE-TIME-OF-CAPTURE tools, not post-capture."""
        tools_at_capture = [
            {
                "type": "function",
                "function": {"name": "foo",
                             "description": "original",
                             "parameters": {"type": "object"}},
            },
        ]
        snapshot = _capture_via_real_method(
            system="S", tools=tools_at_capture, messages=[],
        )
        # Parent mutates its in-memory tools AFTER snapshotting.  In
        # production this could be a toolset swap.
        tools_at_capture[0]["function"]["description"] = "MUTATED"
        tools_at_capture.append(
            {"type": "function",
             "function": {"name": "bar",
                          "description": "added later",
                          "parameters": {"type": "object"}}}
        )

        # What the parent WOULD HAVE sent at capture time (pristine tools):
        pristine_tools = [
            {
                "type": "function",
                "function": {"name": "foo",
                             "description": "original",
                             "parameters": {"type": "object"}},
            },
        ]
        goal = "x"
        parent_kwargs = _parent_would_send(
            system="S", tools=pristine_tools, messages_prefix=[],
            next_user_goal=goal, model=snapshot["model"],
            max_tokens=snapshot["max_tokens"],
            reasoning_config=snapshot["reasoning_config"],
        )
        child_kwargs = _child_actually_sends(snapshot=snapshot,
                                             next_user_goal=goal)

        self.assertEqual(parent_kwargs["tools"], child_kwargs["tools"],
                         "child saw post-capture mutated tools — deep copy failed")


class TestWirePayloadPrefixExactBytes(unittest.TestCase):
    """Tighter: the serialized JSON prefix must match byte-for-byte.

    Kwargs equality is checked above with ==, but Anthropic's cache keys on
    the exact JSON bytes (well, canonical representation).  If two dicts
    compare equal via Python == their JSON serializations may still differ
    (e.g. key ordering isn't preserved by ==, but is by json.dumps).
    We use sort_keys=True to emulate a canonical ordering check.
    """

    def test_canonical_json_prefix_byte_equal(self):
        import json
        parent_system = "You are Hermes."
        parent_tools = [
            {"type": "function", "function": {
                "name": "tool_a", "description": "A",
                "parameters": {"type": "object"}}},
            {"type": "function", "function": {
                "name": "tool_b", "description": "B",
                "parameters": {"type": "object"}}},
        ]
        parent_messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        goal = "the new task"
        snapshot = _capture_via_real_method(
            system=parent_system, tools=parent_tools,
            messages=parent_messages,
        )
        parent_kwargs = _parent_would_send(
            system=parent_system, tools=parent_tools,
            messages_prefix=parent_messages, next_user_goal=goal,
            model=snapshot["model"], max_tokens=snapshot["max_tokens"],
            reasoning_config=snapshot["reasoning_config"],
        )
        child_kwargs = _child_actually_sends(snapshot=snapshot,
                                             next_user_goal=goal)

        # Project onto the cache-relevant prefix only.
        def _cache_prefix(kw, prefix_len):
            return {
                "model": kw.get("model"),
                "system": kw.get("system"),
                "tools": kw.get("tools"),
                "messages_prefix": kw["messages"][:prefix_len],
            }

        prefix_len = len(parent_messages)
        p_bytes = json.dumps(_cache_prefix(parent_kwargs, prefix_len),
                             sort_keys=True).encode("utf-8")
        c_bytes = json.dumps(_cache_prefix(child_kwargs, prefix_len),
                             sort_keys=True).encode("utf-8")
        self.assertEqual(p_bytes, c_bytes,
                         "canonical JSON bytes diverged — cache will miss")
        # Extra: if bytes equal, hashes equal (belt and suspenders).
        import hashlib
        self.assertEqual(hashlib.sha256(p_bytes).hexdigest(),
                         hashlib.sha256(c_bytes).hexdigest())


if __name__ == "__main__":
    unittest.main()
