#!/usr/bin/env python3
"""
W12 — tests for shape-mirror cache-hit telemetry.

The shape-mirror regime (W6-W11) is invisible unless we measure it.  W12
adds per-task and aggregate cache accounting to delegate_task's JSON output
AND to the logger so gateway operators can see hit ratios live.

These tests verify:
  - per-task tokens dict carries cache_read / cache_write / cache_hit_ratio
  - aggregate cache_summary field is present and math adds up
  - zero-delegation (empty results) case doesn't divide by zero
  - mock children don't crash the telemetry path (same pathology that
    caused the W10 isinstance gate)

See ~/.hermes/plans/hermes-patches/delegation-cache-telemetry.md
"""

import json
import unittest
from unittest.mock import MagicMock, patch


class TestPerTaskCacheTelemetry(unittest.TestCase):
    """_run_single_child emits cache_read, cache_write, cache_hit_ratio."""

    def _make_child_with_tokens(self, *, input_t, output_t,
                                cache_read, cache_write):
        """Build a mock child agent that looks like it finished a run."""
        child = MagicMock()
        child.session_prompt_tokens = input_t
        child.session_completion_tokens = output_t
        child.session_cache_read_tokens = cache_read
        child.session_cache_write_tokens = cache_write
        child.model = "claude-sonnet-4-20250514"
        child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 2,
            "messages": [],
        }
        # Prevent the isinstance(dict) gate from triggering on MagicMock.
        # The snapshot attach only happens via _build_child_agent; here we
        # construct the child directly so leave _parent_shape_snapshot un-set.
        child._parent_shape_snapshot = None
        child._credential_pool = None
        child._delegate_saved_tool_names = []
        child.tool_progress_callback = None
        child.get_activity_summary.return_value = {
            "current_tool": None, "api_call_count": 0, "max_iterations": 5,
            "last_activity_desc": "",
        }
        return child

    def _run(self, child, parent=None):
        from tools.delegate_tool import _run_single_child
        if parent is None:
            parent = MagicMock()
            parent._active_children = []
            parent._active_children_lock = None
            parent._touch_activity = lambda *a, **k: None
        return _run_single_child(
            task_index=0,
            goal="test",
            child=child,
            parent_agent=parent,
        )

    def test_tokens_dict_has_cache_fields(self):
        child = self._make_child_with_tokens(
            input_t=1000, output_t=500,
            cache_read=8000, cache_write=200,
        )
        entry = self._run(child)

        self.assertEqual(entry["status"], "completed")
        self.assertIn("tokens", entry)
        toks = entry["tokens"]
        self.assertEqual(toks["input"], 1000)
        self.assertEqual(toks["output"], 500)
        self.assertEqual(toks["cache_read"], 8000)
        self.assertEqual(toks["cache_write"], 200)
        # total prompt = input + cache_read = 1000 + 8000 = 9000
        # hit ratio = 8000 / 9000 ≈ 0.8889
        self.assertAlmostEqual(toks["cache_hit_ratio"], 8000 / 9000, places=3)

    def test_zero_prompt_tokens_hit_ratio_is_zero(self):
        child = self._make_child_with_tokens(
            input_t=0, output_t=0, cache_read=0, cache_write=0,
        )
        entry = self._run(child)
        self.assertEqual(entry["tokens"]["cache_hit_ratio"], 0.0)

    def test_mock_child_token_fields_fall_back_to_zero(self):
        """A MagicMock child that hasn't had its token attrs set should
        coerce to zero rather than leaking a Mock object into the JSON."""
        child = MagicMock()
        # Do NOT set session_* attributes → they auto-create as MagicMock.
        child.run_conversation.return_value = {
            "final_response": "done",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }
        child._parent_shape_snapshot = None
        child._credential_pool = None
        child._delegate_saved_tool_names = []
        child.tool_progress_callback = None
        child.model = "claude-sonnet-4-20250514"  # real string, not mock
        child.get_activity_summary.return_value = {
            "current_tool": None, "api_call_count": 0, "max_iterations": 5,
            "last_activity_desc": "",
        }

        entry = self._run(child)
        toks = entry["tokens"]
        # All four numeric fields must be real numbers, not MagicMocks.
        for k in ("input", "output", "cache_read", "cache_write",
                  "cache_hit_ratio"):
            self.assertIsInstance(toks[k], (int, float),
                                  f"{k} leaked a non-numeric")
        self.assertEqual(toks["cache_hit_ratio"], 0.0)

    def test_tokens_dict_is_json_serializable(self):
        """Sanity: the entry dict must round-trip through json.dumps."""
        child = self._make_child_with_tokens(
            input_t=100, output_t=50, cache_read=900, cache_write=0,
        )
        entry = self._run(child)
        # Must not raise.
        encoded = json.dumps(entry)
        self.assertIn("cache_hit_ratio", encoded)


class TestAggregateCacheSummary(unittest.TestCase):
    """Aggregate cache_summary math in delegate_task JSON payload."""

    def test_aggregate_math_sums_across_tasks(self):
        """Three child results with known tokens → sum them and check ratio."""
        # We test the aggregation logic directly via a lightweight
        # reconstruction rather than spinning up three real mock agents.
        # The aggregation is pure arithmetic on entry dicts.
        entries = [
            {"tokens": {"input": 100, "output": 50,
                        "cache_read": 900, "cache_write": 0,
                        "cache_hit_ratio": 0.9}},
            {"tokens": {"input": 200, "output": 100,
                        "cache_read": 800, "cache_write": 0,
                        "cache_hit_ratio": 0.8}},
            {"tokens": {"input": 50, "output": 10,
                        "cache_read": 0, "cache_write": 150,
                        "cache_hit_ratio": 0.0}},
        ]
        agg_read = sum(e["tokens"]["cache_read"] for e in entries)
        agg_write = sum(e["tokens"]["cache_write"] for e in entries)
        agg_input = sum(e["tokens"]["input"] for e in entries)
        total = agg_input + agg_read
        ratio = agg_read / total if total else 0.0

        self.assertEqual(agg_read, 1700)
        self.assertEqual(agg_write, 150)
        self.assertEqual(agg_input, 350)
        # 1700 / (350 + 1700) = 1700/2050 ≈ 0.8293
        self.assertAlmostEqual(ratio, 1700 / 2050, places=4)

    def test_delegate_task_output_has_cache_summary(self):
        """End-to-end: delegate_task JSON must contain cache_summary with
        correct aggregated fields and hit ratio."""
        from tools.delegate_tool import delegate_task

        # Build a parent agent that has just enough for delegate_task to
        # run without exploding.  We patch _run_single_child to return
        # canned entries so we bypass all the pool/credential plumbing.
        parent = MagicMock()
        parent._delegate_depth = 0
        parent.session_id = "test-session"
        parent._active_children = []
        parent._active_children_lock = None

        canned_entries = [
            {"task_index": 0, "status": "completed", "summary": "ok",
             "api_calls": 1, "duration_seconds": 0.1, "model": "claude",
             "exit_reason": "completed",
             "tokens": {"input": 100, "output": 50,
                        "cache_read": 900, "cache_write": 0,
                        "cache_hit_ratio": 0.9},
             "tool_trace": []},
            {"task_index": 1, "status": "completed", "summary": "ok",
             "api_calls": 1, "duration_seconds": 0.1, "model": "claude",
             "exit_reason": "completed",
             "tokens": {"input": 200, "output": 100,
                        "cache_read": 800, "cache_write": 0,
                        "cache_hit_ratio": 0.8},
             "tool_trace": []},
        ]

        call_counter = {"n": 0}
        def _fake_run_single_child(*args, **kwargs):
            entry = canned_entries[call_counter["n"]]
            call_counter["n"] += 1
            return entry

        with patch("tools.delegate_tool._run_single_child",
                   side_effect=_fake_run_single_child):
            out = delegate_task(
                tasks=[{"goal": "a"}, {"goal": "b"}],
                parent_agent=parent,
            )

        parsed = json.loads(out)
        self.assertIn("cache_summary", parsed)
        summary = parsed["cache_summary"]
        self.assertEqual(summary["input_tokens"], 300)
        self.assertEqual(summary["output_tokens"], 150)
        self.assertEqual(summary["cache_read_tokens"], 1700)
        self.assertEqual(summary["cache_write_tokens"], 0)
        # 1700 / (300 + 1700) = 0.85
        self.assertAlmostEqual(summary["cache_hit_ratio"], 0.85, places=4)

    def test_cache_summary_empty_results_no_divide_by_zero(self):
        """If all children fail with no token data, cache_hit_ratio must
        still be a real number (0.0), not NaN or a ZeroDivisionError."""
        from tools.delegate_tool import delegate_task

        parent = MagicMock()
        parent._delegate_depth = 0
        parent.session_id = "test-session"
        parent._active_children = []
        parent._active_children_lock = None

        def _fake_run_single_child(*args, **kwargs):
            return {
                "task_index": 0, "status": "error",
                "summary": None, "error": "boom",
                "api_calls": 0, "duration_seconds": 0.1,
            }

        with patch("tools.delegate_tool._run_single_child",
                   side_effect=_fake_run_single_child):
            out = delegate_task(
                tasks=[{"goal": "a"}],
                parent_agent=parent,
            )
        parsed = json.loads(out)
        self.assertIn("cache_summary", parsed)
        self.assertEqual(parsed["cache_summary"]["cache_hit_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
