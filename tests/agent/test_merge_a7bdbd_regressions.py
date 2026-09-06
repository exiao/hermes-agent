"""Regressions from the 1849-commit upstream merge (a7bdbd7625).

Two independent bugs shipped together and both broke normal tool use:

  * ``get_read_block_error`` looped over a ``hermes_dirs`` local that had moved
    into ``_denied_path_set()``. That loop runs on EVERY read, not only a denied
    one, so every ``read_file`` and ``search_files`` call raised
    ``NameError: name 'hermes_dirs' is not defined``. 29 occurrences in a day.
  * The plugin hook "still running" guard treated ANY in-flight invocation as
    stuck, so two tool calls in parallel made the second fail closed with
    "pre_tool_call plugin callback timed out or is still running". 17
    occurrences, and the logs never recorded an actual timeout, which is the
    tell that it was concurrency and not a slow plugin.

Both are tested here rather than in the two existing files because the shared
cause is the merge, and a future merge should fail on this one file.
"""
from __future__ import annotations

import threading
import time

import pytest

from agent.file_safety import get_read_block_error
from hermes_cli.plugins import PluginManager


class TestReadsSurviveTheDenylistRefactor:
    """A read of an ordinary file must not raise."""

    def test_reading_an_ordinary_file_returns_no_error(self, tmp_path):
        target = tmp_path / "notes.md"
        target.write_text("hello")
        # The NameError fired here, before any deny decision was reached.
        assert get_read_block_error(str(target)) is None

    def test_reading_a_missing_file_returns_no_error(self, tmp_path):
        assert get_read_block_error(str(tmp_path / "nope.md")) is None

    def test_the_browser_profile_deny_still_applies(self, tmp_path, monkeypatch):
        """The deny this code exists for must survive the move into the cache."""
        import agent.file_safety as fs

        home = tmp_path / "hermes-home"
        (home / "browser-profile" / "Default").mkdir(parents=True)
        cookies = home / "browser-profile" / "Default" / "Cookies"
        cookies.write_text("sqlite")

        monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
        monkeypatch.setattr(fs, "_hermes_root_path", lambda: home)
        fs._DENYLIST_CACHE.clear()

        err = get_read_block_error(str(cookies))
        assert err is not None
        assert "browser" in err.lower()

        allowed = tmp_path / "elsewhere.txt"
        allowed.write_text("x")
        assert get_read_block_error(str(allowed)) is None


class TestParallelToolCallsAreNotBlocked:
    """Concurrency is not a stuck hook."""

    def test_a_second_call_runs_while_the_first_is_still_working(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 5.0
        )

        release = threading.Event()
        entered = threading.Event()
        calls = []
        lock = threading.Lock()

        def slow_first_then_fast(**_kwargs):
            with lock:
                calls.append(1)
                first = len(calls) == 1
            if first:
                # Only the FIRST invocation is slow. The second must be judged
                # on its own merits, not blocked because the first is in
                # flight. Keeping both slow would time the second one out for
                # real and test nothing.
                entered.set()
                release.wait(timeout=10.0)
            return None

        mgr = PluginManager()
        mgr._hooks["pre_tool_call"] = [slow_first_then_fast]

        def first():
            mgr.invoke_hook("pre_tool_call", tool_name="terminal")

        thread = threading.Thread(target=first, daemon=True)
        thread.start()
        assert entered.wait(timeout=5.0), "first invocation never started"

        # The first callback is mid-flight and well inside its timeout. The
        # second must be allowed to run, not fail closed.
        second = mgr.invoke_hook("pre_tool_call", tool_name="terminal")
        blocked = [
            r for r in second if isinstance(r, dict) and r.get("action") == "block"
        ]
        assert not blocked, (
            "a parallel tool call was blocked while the first hook was still "
            f"healthily running: {second}"
        )
        assert len(calls) == 2, "the second invocation never reached the callback"

        release.set()
        thread.join(timeout=5.0)

    def test_a_genuinely_hung_hook_still_fails_closed(self, monkeypatch):
        """The guard's real job must survive the fix."""
        from hermes_cli.plugins import _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE

        monkeypatch.setattr(
            "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.1
        )

        hold = threading.Event()

        def hung(**_kwargs):
            hold.wait(timeout=10.0)
            return None

        mgr = PluginManager()
        mgr._hooks["pre_tool_call"] = [hung]

        first = mgr.invoke_hook("pre_tool_call", tool_name="terminal")
        assert any(
            isinstance(r, dict)
            and r.get("message") == _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE
            for r in first
        ), "a hook that outran its timeout should block"

        # And the abandoned thread must not be joined by later fires.
        t0 = time.monotonic()
        second = mgr.invoke_hook("pre_tool_call", tool_name="terminal")
        assert time.monotonic() - t0 < 5.0
        assert any(
            isinstance(r, dict)
            and r.get("message") == _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE
            for r in second
        )
        hold.set()

    def test_finished_invocations_stop_counting_as_in_flight(self, monkeypatch):
        """A completed call must clear its slot, or the map leaks and blocks."""
        monkeypatch.setattr(
            "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 5.0
        )
        mgr = PluginManager()
        mgr._hooks["pre_tool_call"] = [lambda **_kw: None]

        for _ in range(3):
            assert mgr.invoke_hook("pre_tool_call", tool_name="terminal") == []

        assert not mgr._hook_running_callbacks, (
            f"in-flight map leaked entries: {mgr._hook_running_callbacks}"
        )
