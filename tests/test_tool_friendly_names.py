"""Tests for config-driven tool display and rewrite rules in gateway progress messages."""

import queue


def _build_progress_callback(display_config=None, progress_queue=None,
                             hide_code_in_progress=False, supports_code_blocks=False):
    """Build a minimal progress_callback closure matching gateway/run.py logic.

    This mirrors the relevant subset of GatewayRunner._handle_incoming_message
    so we can test the display mapping without spinning up a full gateway.

    hide_code_in_progress mirrors the per-platform display setting; when False
    (default) a markdown-capable adapter (supports_code_blocks=True) renders a
    terminal command as a ```bash block. When True, code-bearing tools never
    echo their command/code.
    """
    import re as _re_mod

    if display_config is None:
        display_config = {}
    if progress_queue is None:
        progress_queue = queue.Queue()

    _tool_display = display_config.get("tool_display")
    if not isinstance(_tool_display, dict):
        _tool_display = {}

    _rewrite_rules_raw = display_config.get("tool_display_rewrite")
    if not isinstance(_rewrite_rules_raw, dict):
        _rewrite_rules_raw = {}
    _rewrite_compiled = []
    for _pat, _repl in _rewrite_rules_raw.items():
        try:
            _rewrite_compiled.append((_re_mod.compile(_pat), str(_repl)))
        except _re_mod.error:
            pass

    # Legacy compat
    _legacy_friendly = display_config.get("tool_friendly_names")
    if isinstance(_legacy_friendly, dict) and not _tool_display:
        _tool_display = _legacy_friendly
    _legacy_show = display_config.get("tool_show_preview")
    _legacy_show_set = set(_legacy_show) if isinstance(_legacy_show, list) else set()

    def _resolve_tool_display(tool_name, preview, args):
        cmd_text = preview or ""
        if not cmd_text and args:
            cmd_text = args.get("command", args.get("code", ""))

        for pattern, replacement in _rewrite_compiled:
            m = pattern.search(cmd_text)
            if m:
                _r = _re_mod.sub(r"\$(\d+)", r"\\\1", replacement)
                return pattern.sub(_r, m.group(0))

        template = _tool_display.get(tool_name)
        if template is not None:
            if "{preview}" in template:
                return template.replace("{preview}", preview or "")
            elif tool_name in _legacy_show_set and preview:
                return f'{template}: "{preview}"'
            else:
                return template + "..."

        return None

    _CODE_BEARING_TOOLS = {"terminal", "execute_code", "process"}

    def progress_callback(tool_name, preview=None, args=None):
        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(tool_name, default="⚙️")

        _resolved = _resolve_tool_display(tool_name, preview, args)
        _is_code_bearing = tool_name in _CODE_BEARING_TOOLS

        _bash_block = None
        if not hide_code_in_progress:
            if (
                supports_code_blocks
                and tool_name == "terminal"
                and isinstance(args, dict)
                and isinstance(args.get("command"), str)
                and args["command"].strip()
            ):
                _bash_block = f"```bash\n{args['command'].rstrip()}\n```"

        if _bash_block is not None:
            msg = _bash_block
        elif _resolved:
            msg = f"{emoji} {_resolved}"
        elif _is_code_bearing and hide_code_in_progress:
            # Never echo raw command/code — generic label only.
            msg = f"{emoji} {tool_name}..."
        elif preview:
            msg = f'{emoji} {tool_name}: "{preview}"'
        else:
            msg = f"{emoji} {tool_name}..."

        progress_queue.put(msg)
        return msg

    return progress_callback, progress_queue


class TestToolDisplay:
    """tool_display config — static friendly names with {preview} placeholder."""

    def test_no_config_uses_raw_name(self):
        """Without tool_display config, raw tool name is shown."""
        cb, q = _build_progress_callback(display_config={})
        msg = cb("WebSearch", preview="AAPL earnings")
        assert "WebSearch" in msg
        assert "AAPL earnings" in msg

    def test_display_name_replaces_raw(self):
        """With mapping, display name replaces raw tool name."""
        config = {"tool_display": {"WebSearch": "Searching the web: {preview}"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("WebSearch", preview="AAPL earnings")
        assert "Searching the web: AAPL earnings" in msg
        assert "WebSearch" not in msg

    def test_no_preview_placeholder_suppresses_args(self):
        """Display name without {preview} suppresses the raw preview."""
        config = {"tool_display": {"terminal": "Looking up data"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="scripts/serper.sh 'AAPL'")
        assert "Looking up data..." in msg
        assert "serper" not in msg

    def test_unmapped_tool_falls_through(self):
        """Unmapped NON-code tools keep their raw name + preview."""
        config = {"tool_display": {"WebSearch": "Searching the web: {preview}"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("read_file", preview="notes.md")
        assert "read_file" in msg
        assert "notes.md" in msg

    def test_unmapped_terminal_never_echoes_command(self):
        """With hide_code_in_progress, an unmapped `terminal` shows a generic label."""
        config = {"tool_display": {"WebSearch": "Searching the web: {preview}"}}
        cb, q = _build_progress_callback(display_config=config, hide_code_in_progress=True,
                                         supports_code_blocks=True)
        msg = cb("terminal", preview="ls -la /root/.hermes", args={"command": "ls -la /root/.hermes"})
        assert "terminal" in msg
        assert "ls -la" not in msg
        assert "/root/.hermes" not in msg

    def test_terminal_secret_heredoc_never_leaks(self):
        """The real leak case: with hide_code_in_progress, a heredoc must not appear."""
        secret_cmd = (
            "python3 - <<'PY'\n"
            "import os\n"
            "s=os.environ.get('BLOOMBOT_WHATSAPP_SECRET')\n"
            "open('/root/.hermes/.env')\n"
            "PY"
        )
        cb, q = _build_progress_callback(display_config={}, hide_code_in_progress=True,
                                         supports_code_blocks=True)
        msg = cb("terminal", args={"command": secret_cmd})
        for leak in ("BLOOMBOT_WHATSAPP_SECRET", "/root/.hermes/.env", "python3", "import os"):
            assert leak not in msg

    def test_unmapped_execute_code_never_echoes_code(self):
        """With hide_code_in_progress, `execute_code` never echoes its code preview."""
        cb, q = _build_progress_callback(display_config={}, hide_code_in_progress=True)
        msg = cb("execute_code", preview="print(open('/etc/passwd').read())")
        assert "passwd" not in msg
        assert "print(" not in msg

    def test_process_stdin_never_leaks(self):
        """`process` submit/write stdin is part of the terminal surface: with
        hide_code_in_progress, a command piped into a background shell must not
        appear in the progress bubble."""
        cb, q = _build_progress_callback(display_config={}, hide_code_in_progress=True)
        msg = cb("process", preview='submit "cat ~/.hermes/.env"')
        assert ".env" not in msg
        assert "cat " not in msg

    def test_process_stdin_shown_when_hide_off(self):
        """Default (hide off): process previews still flow through as before."""
        cb, q = _build_progress_callback(display_config={}, hide_code_in_progress=False)
        msg = cb("process", preview='submit "echo hi"')
        assert "echo hi" in msg

    def test_default_keeps_bash_block_for_code_block_platforms(self):
        """Default (hide off): a markdown adapter still renders the full ```bash block."""
        cb, q = _build_progress_callback(display_config={}, hide_code_in_progress=False,
                                         supports_code_blocks=True)
        msg = cb("terminal", args={"command": "ls -la /root/.hermes"})
        assert "```bash" in msg
        assert "ls -la /root/.hermes" in msg

    def test_default_plain_platform_unaffected(self):
        """Default (hide off) on a non-code-block platform keeps the old preview line."""
        cb, q = _build_progress_callback(display_config={}, hide_code_in_progress=False,
                                         supports_code_blocks=False)
        msg = cb("terminal", preview="ls -la")
        assert "ls -la" in msg

    def test_empty_display_dict(self):
        """Empty dict = same as no config."""
        config = {"tool_display": {}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("read_file", preview="notes.md")
        assert "read_file" in msg
        assert "notes.md" in msg

    def test_preview_placeholder_with_no_preview(self):
        """{preview} with no actual preview = empty string."""
        config = {"tool_display": {"WebSearch": "Searching the web: {preview}"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("WebSearch")
        assert "Searching the web: " in msg

    def test_non_dict_display_config_handled(self):
        """If tool_display is not a dict, no crash."""
        config = {"tool_display": "oops"}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("WebSearch", preview="test")
        assert "WebSearch" in msg

    def test_messages_queued(self):
        """Progress messages land in the queue."""
        config = {"tool_display": {"WebSearch": "Searching the web: {preview}"}}
        cb, q = _build_progress_callback(display_config=config)
        cb("WebSearch", preview="AAPL")
        assert not q.empty()
        queued = q.get_nowait()
        assert "Searching the web: AAPL" in queued


class TestToolDisplayRewrite:
    """tool_display_rewrite — regex pattern → human-readable replacement."""

    def test_rewrite_matches_and_replaces(self):
        """Rewrite rule captures group and produces human-friendly message."""
        config = {
            "tool_display_rewrite": {"bloom earnings (.+)": "Looking up earnings data: $1"}
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="bloom earnings AAPL")
        assert "Looking up earnings data: AAPL" in msg
        assert "terminal" not in msg.replace("⚙️", "").strip().split(" ", 1)[-1][:8]

    def test_first_match_wins(self):
        """Multiple rules — first match takes priority."""
        config = {
            "tool_display_rewrite": {
                "bloom earnings (.+)": "Looking up earnings data: $1",
                "bloom (.+)": "Bloom generic: $1",
            }
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="bloom earnings TSLA")
        assert "Looking up earnings data: TSLA" in msg
        assert "Bloom generic" not in msg

    def test_no_match_falls_through_to_tool_display(self):
        """When no rewrite matches, tool_display is used."""
        config = {
            "tool_display": {"terminal": "Looking up data"},
            "tool_display_rewrite": {"bloom earnings (.+)": "Looking up earnings data: $1"},
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="curl https://api.example.com")
        assert "Looking up data..." in msg
        assert "curl" not in msg

    def test_rewrite_uses_args_command_fallback(self):
        """When preview is None, rewrite checks args.command."""
        config = {
            "tool_display_rewrite": {"bloom fundamentals (.+)": "Looking up fundamentals: $1"}
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", args={"command": "bloom fundamentals NVDA"})
        assert "Looking up fundamentals: NVDA" in msg

    def test_invalid_regex_skipped(self):
        """Invalid regex patterns are silently skipped."""
        config = {
            "tool_display_rewrite": {
                "[invalid(": "should not crash",
                "bloom price (.+)": "Checking price: $1",
            }
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="bloom price GOOG")
        assert "Checking price: GOOG" in msg

    def test_non_dict_rewrite_config_handled(self):
        """If tool_display_rewrite is not a dict, no crash."""
        config = {"tool_display_rewrite": ["bad", "config"]}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="bloom earnings AAPL")
        # Falls through to raw name
        assert "terminal" in msg

    def test_rewrite_takes_priority_over_tool_display(self):
        """Rewrite rules run before tool_display static names."""
        config = {
            "tool_display": {"terminal": "Looking up data"},
            "tool_display_rewrite": {"bloom news (.+)": "Looking up news: $1"},
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="bloom news MSFT")
        assert "Looking up news: MSFT" in msg
        assert "Looking up data" not in msg

    def test_multiple_capture_groups(self):
        """Multiple capture groups work with $1, $2."""
        config = {
            "tool_display_rewrite": {"bloom (\\w+) (\\w+)": "$1 for $2"}
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="bloom earnings AAPL")
        assert "earnings for AAPL" in msg


class TestLegacyCompat:
    """Legacy config keys (tool_friendly_names, tool_show_preview) still work."""

    def test_legacy_friendly_names_used_when_no_tool_display(self):
        """Old tool_friendly_names key works as fallback."""
        config = {"tool_friendly_names": {"terminal": "Looking up data"}}
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal", preview="some command")
        assert "Looking up data..." in msg
        assert "some command" not in msg

    def test_tool_display_takes_priority_over_legacy(self):
        """New tool_display overrides legacy tool_friendly_names."""
        config = {
            "tool_display": {"terminal": "New name"},
            "tool_friendly_names": {"terminal": "Old name"},
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("terminal")
        assert "New name..." in msg
        assert "Old name" not in msg

    def test_legacy_show_preview_preserved(self):
        """Legacy tool_show_preview still shows preview for mapped tools."""
        config = {
            "tool_friendly_names": {"WebSearch": "Searching the web"},
            "tool_show_preview": ["WebSearch"],
        }
        cb, q = _build_progress_callback(display_config=config)
        msg = cb("WebSearch", preview="AAPL earnings")
        assert "Searching the web" in msg
        assert "AAPL earnings" in msg
