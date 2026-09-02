"""
Tests for MEDIA tag extraction from tool results.

Verifies that MEDIA tags (e.g., from TTS tool) are only extracted from
messages in the CURRENT turn, not from the full conversation history.
This prevents voice messages from accumulating and being sent multiple
times per reply. (Regression test for #160)

Also covers #34608: a stale MEDIA: path emitted by an execute_code /
make_image tool several turns earlier must not leak onto a later
text-only reply, even when the path-based dedup set fails to capture it.
"""

import json
import re
import sys
import types
from unittest.mock import MagicMock

import pytest


def extract_media_tags_fixed(result_messages, history_len):
    """
    Extract MEDIA tags from tool results, but ONLY from new messages
    (those added after history_len). This is the fixed behavior.
    
    Args:
        result_messages: Full list of messages including history + new
        history_len: Length of history before this turn
        
    Returns:
        Tuple of (media_tags list, has_voice_directive bool)
    """
    media_tags = []
    has_voice_directive = False
    
    # Only process new messages from this turn
    new_messages = result_messages[history_len:] if len(result_messages) > history_len else []
    
    for msg in new_messages:
        if msg.get("role") == "tool" or msg.get("role") == "function":
            content = msg.get("content", "")
            if "MEDIA:" in content:
                for match in re.finditer(r'MEDIA:(\S+)', content):
                    path = match.group(1).strip().rstrip('",}')
                    if path:
                        media_tags.append(f"MEDIA:{path}")
                if "[[audio_as_voice]]" in content:
                    has_voice_directive = True
    
    return media_tags, has_voice_directive


def extract_media_tags_production(result_messages, history_len, history_media_paths):
    """Mirror of the production scan in gateway/run.py after the #34608 fix.

    Primary guard: scope the scan to the current turn via ``history_len``
    slicing (matching how ``agent_history`` is passed as
    ``conversation_history`` into ``run_conversation``). Secondary guard:
    path-based dedup against ``history_media_paths`` (the #160 compression-safe
    fallback, also used when compression shrinks the list below history_len).
    """
    media_tags = []
    has_voice_directive = False

    if len(result_messages) >= history_len and history_len:
        scan_msgs = result_messages[history_len:]
    else:
        scan_msgs = result_messages

    for msg in scan_msgs:
        if msg.get("role") == "tool" or msg.get("role") == "function":
            content = msg.get("content", "")
            if "MEDIA:" in content:
                for match in re.finditer(r'MEDIA:(\S+)', content):
                    path = match.group(1).strip().rstrip('",}')
                    if path and path not in history_media_paths:
                        media_tags.append(f"MEDIA:{path}")
                if "[[audio_as_voice]]" in content:
                    has_voice_directive = True

    return media_tags, has_voice_directive


def extract_media_tags_broken(result_messages):
    """
    The BROKEN behavior: extract MEDIA tags from ALL messages including history.
    This causes TTS voice messages to accumulate and be re-sent on every reply.
    """
    media_tags = []
    has_voice_directive = False
    
    for msg in result_messages:
        if msg.get("role") == "tool" or msg.get("role") == "function":
            content = msg.get("content", "")
            if "MEDIA:" in content:
                for match in re.finditer(r'MEDIA:(\S+)', content):
                    path = match.group(1).strip().rstrip('",}')
                    if path:
                        media_tags.append(f"MEDIA:{path}")
                if "[[audio_as_voice]]" in content:
                    has_voice_directive = True
    
    return media_tags, has_voice_directive


class TestMediaExtraction:
    """Tests for MEDIA tag extraction from tool results."""

    def test_repairs_explicit_computer_use_media_path_from_json_result(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_0123456789abcdef0123456789abcdef.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        response = (
            "Here is the screenshot.\n"
            f"MEDIA:/Users/Alice/AppData/Local/hermes/cache/images/{capture_name}"
        )
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "capture", "function": {"name": "computer_use"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "capture",
                "content": json.dumps({"screenshot_path": canonical}),
            },
        ]

        repaired = _repair_explicit_computer_use_media_paths(response, messages)

        assert repaired == f"Here is the screenshot.\nMEDIA:{canonical}"

    def test_repairs_explicit_path_from_multimodal_text_summary(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_fedcba9876543210fedcba9876543210.jpg"
        canonical = rf"D:\Hermes Data\cache\images\{capture_name}"
        response = f'MEDIA:"/Users/Alice/Hermes Data/cache/images/{capture_name}"'
        messages = [
            {
                "role": "tool",
                "name": "computer_use",
                "tool_call_id": "capture",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "capture mode=screen 1920x1080\n"
                            f"  (shareable screenshot saved to {canonical})"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,AAAA"},
                    },
                ],
            }
        ]

        repaired = _repair_explicit_computer_use_media_paths(response, messages)

        assert repaired == f'MEDIA:"{canonical}"'

    def test_does_not_auto_attach_computer_use_capture(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        messages = [
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": canonical}),
            }
        ]

        assert (
            _repair_explicit_computer_use_media_paths("Done.", messages)
            == "Done."
        )

    def test_does_not_rewrite_unmatched_or_previous_turn_capture(self):
        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        old_name = "computer_use_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.png"
        current_name = "computer_use_cccccccccccccccccccccccccccccccc.png"
        old_canonical = rf"C:\cache\images\{old_name}"
        current_canonical = rf"C:\cache\images\{current_name}"
        history = [
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": old_canonical}),
            },
        ]
        current_turn = [
            {"role": "user", "content": "Send the current screenshot."},
            {
                "role": "tool",
                "name": "computer_use",
                "content": json.dumps({"screenshot_path": current_canonical}),
            },
        ]
        response = f"MEDIA:/Users/Alice/cache/images/{old_name}"

        repaired = _repair_explicit_computer_use_media_paths(
            response,
            history + current_turn,
            history_offset=len(history),
        )

        assert repaired == response

    def test_malformed_json_result_fails_closed(self):
        """Truncated JSON must not repair to a doubled-backslash artifact.

        JSON escaping doubles backslashes; regex-scanning the raw string
        would yield ``C:\\\\Users\\\\...`` — a path that exists nowhere. When
        json.loads fails, the helper must yield nothing rather than rewrite
        the response to a corrupted path.
        """
        import json as _json

        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_dddddddddddddddddddddddddddddddd.png"
        canonical = rf"C:\Users\Alice\AppData\Local\hermes\cache\images\{capture_name}"
        payload = _json.dumps(
            {
                "summary": f"capture\n  (shareable screenshot saved to {canonical})",
                "screenshot_path": canonical,
            }
        )
        truncated = payload[:-5]  # starts with '{' but no longer parses
        response = f"MEDIA:/Users/Alice/AppData/Local/hermes/cache/images/{capture_name}"
        messages = [
            {"role": "tool", "name": "computer_use", "content": truncated},
        ]

        assert (
            _repair_explicit_computer_use_media_paths(response, messages)
            == response
        )

    def test_compression_fallback_slices_from_last_user_message(self):
        """When compression shrinks messages below history_offset, the repair
        recovers the current turn from the last user message — and fails
        closed (no rewrite) when no user message remains."""
        import json as _json

        from gateway.media_repair import (
            repair_explicit_computer_use_media_paths as _repair_explicit_computer_use_media_paths,
        )

        capture_name = "computer_use_eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee.png"
        canonical = rf"C:\cache\images\{capture_name}"
        response = f"MEDIA:/cache/images/{capture_name}"
        current_turn = [
            {"role": "user", "content": "Send the screenshot."},
            {
                "role": "tool",
                "name": "computer_use",
                "content": _json.dumps({"screenshot_path": canonical}),
            },
        ]

        # history_offset larger than the (compressed) message list forces the
        # fallback branch; the last-user slice still finds this turn's result.
        repaired = _repair_explicit_computer_use_media_paths(
            response, current_turn, history_offset=10
        )
        assert repaired == f"MEDIA:{canonical}"

        # No user message at all -> fail closed, nothing rewritten.
        no_user = [current_turn[1]]
        assert (
            _repair_explicit_computer_use_media_paths(
                response, no_user, history_offset=10
            )
            == response
        )

    def test_gateway_auto_append_ignores_media_examples_in_skill_docs(self):
        """Skill/documentation examples must not be appended as real attachments."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "How should I format gateway media?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_skill", "function": {"name": "skill_view"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_skill",
                "content": """
Recommended pattern:
```text
MEDIA:/absolute/path/to/image.png
```
Second message:
```text
caption
```
""",
            },
            {"role": "assistant", "content": "Use a standalone media message."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == []
        assert voice is False

    def test_gateway_auto_append_uses_deliverable_final_reply_tags_only(self):
        from gateway.run import _append_missing_auto_media_tags

        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_file", "function": {"name": "send_file"}}]},
            {"role": "tool", "tool_call_id": "call_file", "content": "MEDIA:/tmp/report.md"},
        ]
        response = '{"file":"MEDIA:/tmp/report.md"}'

        assert _append_missing_auto_media_tags(response, messages) == (
            '{"file":"MEDIA:/tmp/report.md"}\nMEDIA:/tmp/report.md'
        )

    def test_gateway_auto_append_counts_inline_deliverable_reply_tags(self):
        from gateway.run import _append_missing_auto_media_tags

        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_file", "function": {"name": "send_file"}}]},
            {"role": "tool", "tool_call_id": "call_file", "content": "MEDIA:/tmp/report.md"},
        ]

        assert _append_missing_auto_media_tags(
            'Already sent MEDIA:"/tmp/report.md"',
            messages,
        ) == 'Already sent MEDIA:"/tmp/report.md"'

    def test_history_scan_uses_send_file_standalone_tag_rules(self):
        from gateway.run import _collect_history_media_paths

        history = [
            {"role": "assistant", "tool_calls": [{"id": "call_file", "function": {"name": "send_file"}}]},
            {
                "role": "tool",
                "tool_call_id": "call_file",
                "content": "Caption has MEDIA:/tmp/example.md\nMEDIA:/tmp/report.md",
            },
        ]

        assert _collect_history_media_paths(history) == {"/tmp/report.md"}

    def test_gateway_auto_append_keeps_real_tts_media_tag(self):
        """TTS tool media tags are still auto-appended when the model omits them."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "Say this as audio"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_tts", "function": {"name": "text_to_speech"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_tts",
                "content": '{"success": true, "media_tag": "[[audio_as_voice]]\\nMEDIA:/tmp/voice.ogg"}',
            },
            {"role": "assistant", "content": "Done."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/tmp/voice.ogg"]
        assert voice is True

    def test_gateway_auto_append_keeps_send_file_media_tag(self):
        """send_file media tags are auto-appended even when the model forgets to
        echo them in the final reply (the original silent-drop bug). Covers a
        code-file extension (.py) the old hardcoded tool-result matcher omitted.
        """
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "send me the compile prompt file"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_sf", "function": {"name": "send_file"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_sf",
                "content": "prompts.py\nMEDIA:/repo/pipeline/prompts.py",
            },
            {"role": "assistant", "content": "Sent."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/repo/pipeline/prompts.py"]
        assert voice is False

    def test_gateway_auto_append_send_file_markdown_and_json(self):
        """send_file delivers .md/.json — extensions the pre-fix tool-result
        matcher dropped, so a batch of doc/data files vanished silently.
        """
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "send the skill and coverage files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "function": {"name": "send_file"}},
                    {"id": "b", "function": {"name": "send_file"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "MEDIA:/s/SKILL.md"},
            {"role": "tool", "tool_call_id": "b", "content": "MEDIA:/w/coverage.json"},
            {"role": "assistant", "content": "Both sent."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/s/SKILL.md", "MEDIA:/w/coverage.json"]
        assert voice is False

    def test_gateway_auto_append_send_file_ignores_caption_media_examples(self):
        """send_file captions are prepended before the validated MEDIA line;
        only the returned file_path tag should be eligible for auto-append.
        """
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "send the report"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_sf", "function": {"name": "send_file"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_sf",
                "content": (
                    "Caption example MEDIA:/private/secret.md\n"
                    "MEDIA:/safe/report.md"
                ),
            },
            {"role": "assistant", "content": "Sent."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/safe/report.md"]
        assert voice is False

    def test_gateway_auto_append_send_file_paths_with_spaces(self):
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "send the report"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_sf", "function": {"name": "send_file"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_sf",
                "content": "MEDIA:/tmp/My Folder/report.md",
            },
            {"role": "assistant", "content": "Sent."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/tmp/My Folder/report.md"]
        assert voice is False

    def test_gateway_auto_append_adds_unechoed_send_file_tags_from_partial_batch(self):
        from gateway.run import _append_missing_auto_media_tags

        messages = [
            {"role": "user", "content": "send both files"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "a", "function": {"name": "send_file"}},
                    {"id": "b", "function": {"name": "send_file"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "MEDIA:/tmp/first.md"},
            {"role": "tool", "tool_call_id": "b", "content": "MEDIA:/tmp/second.md"},
        ]

        response = _append_missing_auto_media_tags(
            "Here is the first file\nMEDIA:/tmp/first.md",
            messages,
            history_offset=0,
        )

        assert response.count("MEDIA:/tmp/first.md") == 1
        assert response.endswith("MEDIA:/tmp/second.md")

    def test_current_turn_send_file_can_resend_historical_path(self):
        from gateway.run import _collect_auto_append_media_tags

        history = [
            {"role": "user", "content": "send report"},
            {"role": "assistant", "tool_calls": [{"id": "old", "function": {"name": "send_file"}}]},
            {"role": "tool", "tool_call_id": "old", "content": "MEDIA:/tmp/report.md"},
            {"role": "assistant", "content": "Sent."},
        ]
        current = [
            {"role": "user", "content": "send report again"},
            {"role": "assistant", "tool_calls": [{"id": "new", "function": {"name": "send_file"}}]},
            {"role": "tool", "tool_call_id": "new", "content": "MEDIA:/tmp/report.md"},
        ]

        tags, voice = _collect_auto_append_media_tags(
            history + current,
            history_offset=len(history),
            history_media_paths={"/tmp/report.md"},
        )

        assert tags == ["MEDIA:/tmp/report.md"]
        assert voice is False

    def test_compression_fallback_current_send_file_can_resend_historical_path(self):
        """Compression fallback scans all returned messages, but a fresh
        current-turn send_file tool result must still resend a path that also
        exists in history.
        """
        from gateway.run import _collect_auto_append_media_tags

        compressed_messages = [
            {"role": "system", "content": "Earlier context was compressed."},
            {"role": "user", "content": "send report again"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "new", "function": {"name": "send_file"}}],
            },
            {"role": "tool", "tool_call_id": "new", "content": "MEDIA:/tmp/report.md"},
        ]

        tags, voice = _collect_auto_append_media_tags(
            compressed_messages,
            history_offset=50,
            history_media_paths={"/tmp/report.md"},
            history_tool_call_ids={"old"},
        )

        assert tags == ["MEDIA:/tmp/report.md"]
        assert voice is False

    def test_quoted_final_response_media_tags_are_not_duplicated(self):
        from gateway.run import _append_missing_auto_media_tags

        messages = [
            {"role": "user", "content": "send report"},
            {"role": "assistant", "tool_calls": [{"id": "sf", "function": {"name": "send_file"}}]},
            {"role": "tool", "tool_call_id": "sf", "content": "MEDIA:/tmp/report.md"},
        ]

        response = _append_missing_auto_media_tags(
            'Already sent MEDIA:"/tmp/report.md"',
            messages,
            history_offset=0,
        )

        assert response == 'Already sent MEDIA:"/tmp/report.md"'

    def test_windows_media_echo_paths_are_deduped_across_separators(self):
        from gateway.run import _append_missing_auto_media_tags

        messages = [
            {"role": "user", "content": "send report"},
            {"role": "assistant", "tool_calls": [{"id": "sf", "function": {"name": "send_file"}}]},
            {"role": "tool", "tool_call_id": "sf", "content": r"MEDIA:C:\Users\me\report.md"},
        ]

        response = _append_missing_auto_media_tags(
            "Already sent MEDIA:C:/Users/me/report.md",
            messages,
            history_offset=0,
        )

        assert response == "Already sent MEDIA:C:/Users/me/report.md"

    def test_auto_append_updates_result_for_queued_first_response(self):
        from gateway.run import _append_missing_auto_media_tags_to_result

        result = {
            "final_response": "Sent.",
            "messages": [
                {"role": "user", "content": "send report"},
                {"role": "assistant", "tool_calls": [{"id": "sf", "function": {"name": "send_file"}}]},
                {"role": "tool", "tool_call_id": "sf", "content": "MEDIA:/tmp/report.md"},
            ],
        }

        response = _append_missing_auto_media_tags_to_result(result, history_offset=0)

        assert response == "Sent.\nMEDIA:/tmp/report.md"
        assert result["final_response"] == "Sent.\nMEDIA:/tmp/report.md"

    def test_append_missing_auto_media_tags_to_result_handles_none_messages(self):
        from gateway.run import _append_missing_auto_media_tags_to_result

        result = {"final_response": "Sent.", "messages": None}

        response = _append_missing_auto_media_tags_to_result(result, history_offset=0)

        assert response == "Sent."
        assert result["final_response"] == "Sent."

    @pytest.mark.asyncio
    async def test_background_task_auto_appends_send_file_media(self, monkeypatch, tmp_path):
        from gateway.config import Platform
        from gateway.platforms.base import BasePlatformAdapter
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        media_path = tmp_path / "report.md"
        media_path.write_text("hello", encoding="utf-8")

        class FakeAgent:
            def __init__(self, **kwargs):
                pass

            def run_conversation(self, user_message, task_id=None):
                return {
                    "final_response": "Sent.",
                    "messages": [
                        {"role": "assistant", "tool_calls": [{"id": "sf", "function": {"name": "send_file"}}]},
                        {"role": "tool", "tool_call_id": "sf", "content": f"MEDIA:{media_path}"},
                    ],
                }

        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        class Adapter:
            name = "stub"
            extract_media = staticmethod(BasePlatformAdapter.extract_media)
            extract_images = staticmethod(lambda content: ([], content))

            def __init__(self):
                self.sent = []
                self.documents = []

            async def send(self, chat_id, content, metadata=None, **kwargs):
                self.sent.append(content)

            async def send_image(self, **kwargs):
                raise AssertionError("unexpected image URL delivery")

            async def send_voice(self, **kwargs):
                raise AssertionError("unexpected voice delivery")

            async def send_video(self, **kwargs):
                raise AssertionError("unexpected video delivery")

            async def send_image_file(self, **kwargs):
                raise AssertionError("unexpected image file delivery")

            async def send_document(self, chat_id, file_path, metadata=None, **kwargs):
                self.documents.append(file_path)

        async def run_inline(fn):
            return fn()

        adapter = Adapter()
        source = SessionSource(platform=Platform.DISCORD, chat_id="chat", chat_type="dm", user_id="user")
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = {Platform.DISCORD: adapter}
        runner._provider_routing = {}
        runner._fallback_model = None
        runner._service_tier = None
        runner._session_db = None
        runner._thread_metadata_for_source = lambda source, event_message_id=None: None
        runner._resolve_session_agent_runtime = lambda **kwargs: ("model", {"api_key": "key"})
        runner._resolve_session_reasoning_config = lambda **kwargs: None
        runner._load_service_tier = lambda: None
        runner._resolve_turn_agent_config = lambda prompt, model, runtime: {"model": model, "runtime": runtime}
        runner._run_in_executor_with_context = run_inline
        runner._cleanup_agent_resources = lambda agent: None
        monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"agent": {}})

        await runner._run_background_task("send report", source, "task-1")

        assert any(text.startswith("✅ Background task complete") for text in adapter.sent)
        assert adapter.documents == [str(media_path)]

    def test_send_file_ignores_subdirectory_context_media_hints(self):
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "send report"},
            {"role": "assistant", "tool_calls": [{"id": "sf", "function": {"name": "send_file"}}]},
            {
                "role": "tool",
                "tool_call_id": "sf",
                "content": (
                    "MEDIA:/safe/report.md\n\n"
                    "[Subdirectory context discovered: docs/AGENTS.md]\n"
                    "Example:\nMEDIA:/private/example.md"
                ),
            },
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)

        assert tags == ["MEDIA:/safe/report.md"]
        assert voice is False

    def test_history_media_paths_use_widened_shared_matcher(self):
        from gateway.run import _collect_history_media_paths

        history = [
            {"role": "tool", "content": "MEDIA:/repo/pipeline/prompts.py"},
            {"role": "tool", "content": "MEDIA:/tmp/My Folder/report.md"},
        ]

        assert _collect_history_media_paths(history) == {
            "/repo/pipeline/prompts.py",
            "/tmp/My Folder/report.md",
        }

    def test_gateway_auto_append_image_generate_json_path(self):
        """image_generate returns a local path in JSON (no MEDIA: tag); it is
        auto-appended so delivery doesn't depend on the model restating it."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "Make me a cat"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_img", "function": {"name": "image_generate"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_img",
                "content": '{"success": true, "image": "/tmp/gen/cat.png", "agent_visible_image": "/tmp/gen/cat.png"}',
            },
            {"role": "assistant", "content": "Here's your cat."},
        ]

        tags, voice = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/tmp/gen/cat.png"]
        assert voice is False

    def test_gateway_auto_append_image_generate_prefers_host_path(self):
        """When host and sandbox paths differ, the host-deliverable path wins."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {"role": "user", "content": "Make me a dog"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_img", "function": {"name": "image_generate"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_img",
                "content": '{"success": true, "host_image": "/host/dog.jpg", "image": "/host/dog.jpg", "agent_visible_image": "/sandbox/dog.jpg"}',
            },
        ]

        tags, _ = _collect_auto_append_media_tags(messages, history_offset=0)
        assert tags == ["MEDIA:/host/dog.jpg"]

    def test_gateway_auto_append_image_generate_failure_and_url_ignored(self):
        """Failed generations and remote URLs are not auto-delivered."""
        from gateway.run import _collect_auto_append_media_tags

        def _img_msgs(content):
            return [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c", "function": {"name": "image_generate"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "c", "content": content},
            ]

        # Failed generation
        tags, _ = _collect_auto_append_media_tags(
            _img_msgs('{"success": false, "image": null, "error": "boom"}'),
            history_offset=0,
        )
        assert tags == []

        # Remote URL is not a local file path
        tags, _ = _collect_auto_append_media_tags(
            _img_msgs('{"success": true, "image": "https://fal.media/x/cat.png"}'),
            history_offset=0,
        )
        assert tags == []

    def test_gateway_auto_append_image_generate_dedupes_history(self):
        """A generated image path already in history is not re-sent."""
        from gateway.run import _collect_auto_append_media_tags

        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c", "function": {"name": "image_generate"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": '{"success": true, "image": "/tmp/gen/cat.png"}',
            },
        ]

        tags, _ = _collect_auto_append_media_tags(
            messages, history_offset=0, history_media_paths={"/tmp/gen/cat.png"}
        )
        assert tags == []

    def test_collect_history_media_paths_includes_image_generate_json(self):
        """Regression for #46627: the history media-path collector must pick up
        image_generate JSON-payload paths (no MEDIA: tag), not just MEDIA:
        text tags. Otherwise, after a compression boundary the auto-append
        fallback rescans full history, finds the generated path absent from
        the dedup set, and re-emits the same MEDIA tag every turn.
        """
        from gateway.run import _collect_history_media_paths

        history = [
            {"role": "user", "content": "make a cat"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": '{"success": true, "image": "/tmp/gen/cat.png"}',
            },
            # A separate MEDIA: text tag from another tool, to confirm both shapes.
            {
                "role": "tool",
                "tool_call_id": "d",
                "content": "Saved MEDIA:/tmp/voice/note.ogg done",
            },
        ]
        paths = _collect_history_media_paths(history)
        assert "/tmp/gen/cat.png" in paths  # JSON-payload path (the bug)
        assert "/tmp/voice/note.ogg" in paths  # MEDIA: text path (already worked)

    def test_non_streaming_dedup_excludes_current_turn_tool_output(self):
        from gateway.platforms.base import BasePlatformAdapter

        old_path = "/tmp/gen/old.png"
        current_path = "/tmp/gen/current.png"
        transcript = [
            {"role": "user", "content": "make the old image"},
            {"role": "assistant", "content": f"MEDIA:{old_path}"},
            {"role": "user", "content": "make a new image"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "current", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "current",
                "content": f'{{"success": true, "image": "{current_path}"}}',
            },
            {"role": "assistant", "content": f"MEDIA:{current_path}"},
        ]
        adapter = MagicMock()
        adapter._session_store.peek_session_id.return_value = "session-id"
        adapter._session_store.load_transcript.return_value = transcript

        paths = BasePlatformAdapter._history_media_paths_for_session(
            adapter, "session-key"
        )
        assert paths == {old_path}

    @pytest.mark.parametrize(
        "current_path",
        ["/tmp/tts/current.ogg", "/tmp/tts/already-delivered.ogg"],
    )
    def test_non_streaming_dedup_scopes_tts_paths_to_prior_turns(
        self, current_path
    ):
        from gateway.platforms.base import BasePlatformAdapter

        old_path = "/tmp/tts/already-delivered.ogg"
        transcript = [
            {"role": "user", "content": "say the old message"},
            {"role": "assistant", "content": f"MEDIA:{old_path}"},
            {"role": "user", "content": "say the current message"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "tts", "function": {"name": "text_to_speech"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tts",
                "content": f"[[audio_as_voice]]\\nMEDIA:{current_path}",
            },
            {
                "role": "assistant",
                "content": f"[[audio_as_voice]]\\nMEDIA:{current_path}",
            },
        ]
        adapter = MagicMock()
        adapter._session_store.peek_session_id.return_value = "session-id"
        adapter._session_store.load_transcript.return_value = transcript

        paths = BasePlatformAdapter._history_media_paths_for_session(
            adapter, "session-key"
        )
        assert paths == {old_path}

    def test_image_generate_not_reemitted_after_compression(self):
        """End-to-end of the #46627 fix: collect history paths, then the
        compression-fallback rescan (history_offset stale) must dedup the
        generated image against them — no re-emission."""
        from gateway.run import (
            _collect_auto_append_media_tags,
            _collect_history_media_paths,
        )

        history = [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c", "function": {"name": "image_generate"}}],
            },
            {
                "role": "tool",
                "tool_call_id": "c",
                "content": '{"success": true, "image": "/tmp/gen/dog.png"}',
            },
        ]
        history_paths = _collect_history_media_paths(history)

        # Simulate the post-compression fallback: history_offset is stale
        # (larger than the shrunken message list), so the collector rescans
        # the full list. With the dedup set populated, the already-delivered
        # image must NOT be re-emitted.
        tags, _ = _collect_auto_append_media_tags(
            history, history_offset=9999, history_media_paths=history_paths
        )
        assert tags == [], f"generated image re-emitted after compression: {tags}"


    def test_media_tags_not_extracted_from_history(self):
        """MEDIA tags from previous turns should NOT be extracted again."""
        # Simulate conversation history with a TTS call from a previous turn
        history = [
            {"role": "user", "content": "Say hello as audio"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "text_to_speech"}}]},
            {"role": "tool", "tool_call_id": "1", "content": '{"success": true, "media_tag": "[[audio_as_voice]]\\nMEDIA:/path/to/audio1.ogg"}'},
            {"role": "assistant", "content": "I've said hello for you!"},
        ]
        
        # New turn: user asks a simple question
        new_messages = [
            {"role": "user", "content": "What time is it?"},
            {"role": "assistant", "content": "It's 3:30 AM."},
        ]
        
        all_messages = history + new_messages
        history_len = len(history)
        
        # Fixed behavior: should extract NO media tags (none in new messages)
        tags, voice_directive = extract_media_tags_fixed(all_messages, history_len)
        assert tags == [], "Fixed extraction should not find tags in history"
        assert voice_directive is False
        
        # Broken behavior: would incorrectly extract the old media tag
        broken_tags, broken_voice = extract_media_tags_broken(all_messages)
        assert len(broken_tags) == 1, "Broken extraction finds tags in history"
        assert "audio1.ogg" in broken_tags[0]
    
    
    


class TestStaleToolMediaLeak:
    """Regression tests for #34608.

    A MEDIA: path emitted by an execute_code / make_image tool several turns
    earlier remains in the full conversation message list. A later text-only
    reply (zero MEDIA directives) must NOT attach that stale image.

    The production code previously relied solely on path-based dedup against
    paths reconstructed from the replayable transcript. When that
    reconstruction does not byte-match the in-memory tool content (timestamp
    stripping, observed-context withholding, compression rewrites), the stale
    path is absent from the dedup set and leaks. Turn-scoped slicing closes
    this class of bug deterministically.
    """

    def test_stale_execute_code_media_not_attached_to_text_only_reply(self):
        """The exact #34608 scenario: make_image cover from an earlier turn."""
        # Prior turn generated an image via execute_code stdout.
        history = [
            {"role": "user", "content": "Make a cover image"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "function": {"name": "execute_code"}}]},
            {"role": "tool", "tool_call_id": "1",
             "content": "Generating cover...\nMEDIA:/tmp/seosmi_cover.png\nDone."},
            {"role": "assistant", "content": "Here is your cover."},
        ]
        # Current turn: plain text status update, zero MEDIA directives.
        new_messages = [
            {"role": "user", "content": "What skill version am I on?"},
            {"role": "assistant", "content": "You're on v0.15.1."},
        ]
        all_messages = history + new_messages
        history_len = len(history)

        # Simulate the dedup set FAILING to capture the stale path (the real
        # #34608 condition: replayable-history reconstruction diverged from
        # the in-memory tool content, so the path is not in the set).
        history_media_paths = set()

        tags, voice = extract_media_tags_production(
            all_messages, history_len, history_media_paths
        )
        assert tags == [], (
            "Stale tool MEDIA from a prior turn must not leak onto a "
            f"later text-only reply, got {tags}"
        )
        assert voice is False

        # The pre-fix production behaviour (scan everything, dedup only) would
        # have leaked the stale path when the dedup set missed it.
        broken_tags, _ = extract_media_tags_broken(all_messages)
        assert any("seosmi_cover.png" in t for t in broken_tags), (
            "Sanity: the unscoped scan does surface the stale path"
        )


    def test_compression_shrink_falls_back_to_path_dedup(self):
        """When the list is shorter than history_len (mid-run compression),
        fall back to scanning everything with path-based dedup so the #160
        compression-safe guarantee is preserved."""
        # Post-compression list is shorter than the original history length.
        compressed_messages = [
            {"role": "user", "content": "summary so far..."},
            {"role": "tool", "tool_call_id": "7",
             "content": "MEDIA:/tmp/old_from_history.png"},
            {"role": "assistant", "content": "ok"},
        ]
        original_history_len = 12  # larger than the compressed list
        # The old path IS captured in the dedup set here (history scan ran
        # before compression), so it must still be excluded.
        history_media_paths = {"/tmp/old_from_history.png"}
        tags, _ = extract_media_tags_production(
            compressed_messages, original_history_len, history_media_paths
        )
        assert tags == [], (
            "On the compression fallback path, path-dedup must still exclude "
            f"known-old media, got {tags}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
