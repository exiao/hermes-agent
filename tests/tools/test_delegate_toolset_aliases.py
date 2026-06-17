#!/usr/bin/env python3
"""Tests for delegate_task toolset-alias normalization.

Verifies that common wrong toolset names (display names, mcp_-prefixed names,
and observed typos) are mapped to their canonical registry keys instead of
being silently dropped, and that unknown names are dropped with a warning.

Run with:  python -m pytest tests/tools/test_delegate_toolset_aliases.py -v
"""

import logging
import unittest
from unittest.mock import patch

from tools.delegate_tool import _normalize_toolset_names
from tools.registry import ToolRegistry


class TestToolsetAliasNormalization(unittest.TestCase):
    def test_real_world_wrong_names_map_to_canonical(self):
        # Names taken from a session-history audit of actual mistakes.
        cases = {
            "mcp_terminal": "terminal",
            "ShellExec": "terminal",
            "shell": "terminal",
            "SessionSearch": "session_search",
            "mcp_session_search": "session_search",
            "mcp_memory": "memory",
            "code": "code_execution",
            "filesystem": "file",
            "web_search": "search",
            "websearch": "search",
        }
        for wrong, canonical in cases.items():
            self.assertEqual(
                _normalize_toolset_names([wrong]),
                [canonical],
                f"{wrong!r} should normalize to {canonical!r}",
            )

    def test_mcp_prefix_strip_for_valid_remainder(self):
        # A leading mcp_ on an otherwise-valid toolset is stripped.
        self.assertEqual(_normalize_toolset_names(["mcp_web"]), ["web"])
        self.assertEqual(_normalize_toolset_names(["mcp_vision"]), ["vision"])

    def test_registry_backed_toolsets_are_accepted_case_insensitively(self):
        reg = ToolRegistry()
        reg.register(
            name="minimax_generate",
            toolset="mcp-MiniMax",
            schema={"description": "Generate via MiniMax"},
            handler=lambda _args: "{}",
        )
        reg.register_toolset_alias("MiniMax", "mcp-MiniMax")

        with patch("tools.registry.registry", reg):
            self.assertEqual(
                _normalize_toolset_names(["mcp-minimax", "MiniMax", "minimax"]),
                ["mcp-MiniMax"],
            )

    def test_canonical_names_pass_through_unchanged(self):
        names = ["terminal", "file", "web", "session_search", "memory"]
        self.assertEqual(_normalize_toolset_names(names), names)

    def test_unknown_name_is_dropped_and_warns(self):
        with self.assertLogs("tools.delegate_tool", level="WARNING") as cm:
            result = _normalize_toolset_names(["terminal", "totally_bogus"])
        self.assertEqual(result, ["terminal"])
        self.assertTrue(
            any("totally_bogus" in line for line in cm.output),
            "expected a warning naming the unknown toolset",
        )

    def test_order_preserved_and_deduped(self):
        # ShellExec -> terminal, then an explicit terminal: collapse to one,
        # preserving first-seen order relative to file.
        self.assertEqual(
            _normalize_toolset_names(["ShellExec", "file", "terminal"]),
            ["terminal", "file"],
        )

    def test_empty_and_none(self):
        self.assertEqual(_normalize_toolset_names([]), [])
        self.assertEqual(_normalize_toolset_names(None), None)


if __name__ == "__main__":
    unittest.main()
