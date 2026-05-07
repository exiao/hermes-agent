"""Tests for config-driven tool friendly names in gateway progress messages."""

import queue


def _build_progress_callback(display_config=None, progress_queue=None):
    """Build a minimal progress_callback closure matching gateway/run.py logic.

    This mirrors the relevant subset of GatewayRunner._handle_incoming_message
    so we can test the friendly-name mapping without spinning up a full gateway.
    """
    if display_config is None:
        display_config = {}
    if progress_queue is None:
        progress_queue = queue.Queue()

    _tool_friendly_names = display_config.get("tool_friendly_names")
    if not isinstance(_tool_friendly_names, dict):
        _tool_friendly_names = {}

    _tool_show_preview = set(display_config.get("tool_show_preview") or [])

    def progress_callback(tool_name, preview=None, args=None):
        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(tool_name, default="⚙️")
        display_name = _tool_friendly_names.get(tool_name, tool_name or "")
        _has_friendly = tool_name in _tool_friendly_names
        _suppress_preview = _has_friendly and tool_name not in _tool_show_preview

        if _suppress_preview:
            msg = f"{emoji} {display_name}..."
        elif preview:
            msg = f'{emoji} {display_name}: "{preview}"'
        else:
            msg = f"{emoji} {display_name}..."
        progress_queue.put(msg)
        return msg

    return progress_callback, progress_queue


class TestToolFriendlyNames:
    """Friendly name mapping in progress messages."""

    def test_no_config_uses_raw_name(self):
        """Without tool_friendly_names config, raw tool name is shown."""
        cb, q = _build_progress_callback(display_config={})
        msg = cb("WebSearch", preview="AAPL earnings")
        assert "WebSearch" in msg
        assert "Searching the web" not in msg

    def test_friendly_name_replaces_raw(self):
        """With mapping, friendly name replaces raw tool name."""
        config = {"tool_friendly_names": {"WebSearch": "Searching the web"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("WebSearch", preview="AAPL earnings")
        assert "Searching the web" in msg
        assert "WebSearch" not in msg

    def test_unmapped_tool_falls_through(self):
        """Tools not in the mapping dict keep their raw name."""
        config = {"tool_friendly_names": {"WebSearch": "Searching the web"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal")
        assert "terminal" in msg

    def test_empty_friendly_names_dict(self):
        """Empty dict = same as no config."""
        config = {"tool_friendly_names": {}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("read_file", preview="notes.md")
        assert "read_file" in msg

    def test_no_preview_with_friendly_name(self):
        """Friendly name works for the no-preview branch too."""
        config = {"tool_friendly_names": {"terminal": "Looking up data"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal")
        assert "Looking up data..." in msg

    def test_display_config_none_handled(self):
        """If display_config is None (not a dict), no crash."""
        cb, q = _build_progress_callback(display_config=None)
        msg = cb("WebSearch", preview="test")
        assert "WebSearch" in msg

    def test_messages_queued(self):
        """Progress messages land in the queue."""
        config = {"tool_friendly_names": {"WebSearch": "Searching the web"}}
        cb, q = _build_progress_callback(display_config=config)
        cb("WebSearch", preview="AAPL")
        assert not q.empty()
        queued = q.get_nowait()
        assert "Searching the web" in queued

    def test_non_dict_friendly_names_ignored(self):
        """If tool_friendly_names is a non-dict truthy value, treat as empty."""
        config = {"tool_friendly_names": "oops"}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("WebSearch", preview="test")
        assert "WebSearch" in msg

    # --- Preview suppression tests ---

    def test_friendly_name_suppresses_preview_by_default(self):
        """Tools with a friendly name hide their preview (raw args are noise)."""
        config = {"tool_friendly_names": {"terminal": "Looking up data"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="scripts/serper.sh 'AAPL'")
        assert "Looking up data..." in msg
        assert "serper" not in msg

    def test_show_preview_opts_tool_back_in(self):
        """tool_show_preview list re-enables preview for specific tools."""
        config = {
            "tool_friendly_names": {"WebSearch": "Searching the web"},
            "tool_show_preview": ["WebSearch"],
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("WebSearch", preview="AAPL earnings")
        assert "Searching the web" in msg
        assert "AAPL earnings" in msg

    def test_unmapped_tool_still_shows_preview(self):
        """Tools without a friendly name always show preview (no suppression)."""
        config = {"tool_friendly_names": {"terminal": "Looking up data"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("WebSearch", preview="AAPL earnings")
        assert "WebSearch" in msg
        assert "AAPL earnings" in msg

    def test_show_preview_empty_list(self):
        """Empty tool_show_preview = all friendly-named tools suppress preview."""
        config = {
            "tool_friendly_names": {"terminal": "Looking up data"},
            "tool_show_preview": [],
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="/root/.hermes/scripts/foo.sh")
        assert "Looking up data..." in msg
        assert "/root" not in msg
