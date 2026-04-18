"""Tests for the send_file tool and expanded extract_media() regex.

Covers: send_file_tool validation, MEDIA: tag generation, extract_media()
matching for newly-added file extensions (documents, config, code, archives),
and extract_local_files() expanded extension support.

Based on hermes-patches/send-file-attachment.md.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.send_file_tool import send_file_tool, MAX_FILE_SIZE, SEND_FILE_SCHEMA
from tools.registry import registry
from agent.anthropic_adapter import convert_tools_to_anthropic
from gateway.platforms.base import BasePlatformAdapter

# Trigger registration (idempotent)
import tools.send_file_tool  # noqa: F401


# ===========================================================================
# send_file_tool — validation
# ===========================================================================

class TestSendFileToolValidation:
    """Core validation: existence, type, size, error messages."""

    def test_nonexistent_file_returns_error(self):
        result = send_file_tool("/nonexistent/path/file.txt")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower() or "Not found" in parsed["error"]

    def test_directory_returns_error(self, tmp_path):
        result = send_file_tool(str(tmp_path))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "directory" in parsed["error"].lower() or "Not a file" in parsed["error"]

    def test_empty_file_returns_error(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = send_file_tool(str(f))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "empty" in parsed["error"].lower()

    def test_oversized_file_returns_error(self, tmp_path):
        f = tmp_path / "big.bin"
        f.write_text("x")  # Create the file
        # Mock the size check
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = MAX_FILE_SIZE + 1
            mock_stat.return_value.st_mode = 0o100644
            result = send_file_tool(str(f))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "too large" in parsed["error"].lower() or "100 MB" in parsed["error"]

    def test_valid_file_returns_media_tag(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Hello")
        result = send_file_tool(str(f))
        assert result.startswith("MEDIA:")
        assert str(f.resolve()) in result

    def test_tilde_expansion(self):
        """~/file paths are expanded to absolute."""
        # Create a temp file in home dir
        home = Path.home()
        test_file = home / ".hermes_test_send_file_tmp"
        try:
            test_file.write_text("test content")
            result = send_file_tool("~/.hermes_test_send_file_tmp")
            assert result.startswith("MEDIA:")
            assert str(test_file) in result
        finally:
            test_file.unlink(missing_ok=True)

    def test_caption_prepended(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        result = send_file_tool(str(f), caption="Here's the config")
        assert result.startswith("Here's the config")
        assert "\nMEDIA:" in result

    def test_no_caption_no_prefix(self, tmp_path):
        f = tmp_path / "data.yaml"
        f.write_text("key: value")
        result = send_file_tool(str(f))
        assert result.startswith("MEDIA:")


# ===========================================================================
# extract_media() — expanded extension matching
# ===========================================================================

class TestExtractMediaExpandedExtensions:
    """Verify extract_media() matches MEDIA: tags for all newly-added extensions."""

    @pytest.mark.parametrize("ext", [
        # Documents (newly added)
        "pdf", "txt", "md", "csv", "rtf", "doc", "docx", "xls", "xlsx", "pptx",
        # Config/data (newly added)
        "json", "yaml", "yml", "toml", "xml", "ini", "cfg", "conf",
        # Code (newly added)
        "py", "js", "ts", "sh", "rb", "go", "rs", "java", "c", "cpp", "h", "hpp", "sql", "html", "css",
        # Archives (newly added)
        "zip", "tar", "gz",
        # Logs (newly added)
        "log",
        # Original media (regression)
        "png", "jpg", "jpeg", "gif", "webp",
        "mp4", "mov", "avi", "mkv", "webm",
        "ogg", "opus", "mp3", "wav", "m4a",
    ])
    def test_extract_media_matches_extension(self, ext):
        content = f"Here is the file\nMEDIA:/tmp/test_file.{ext}"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) >= 1, f"extract_media() did not match .{ext}"
        path = media[0][0]
        assert path.endswith(f".{ext}"), f"Extracted path {path!r} doesn't end with .{ext}"

    def test_extract_media_cleans_tag_from_text(self):
        content = "Here is your config\nMEDIA:/tmp/config.yaml"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert "MEDIA:" not in cleaned
        assert "Here is your config" in cleaned

    def test_extract_media_multiple_files(self):
        content = (
            "Files attached:\n"
            "MEDIA:/tmp/report.pdf\n"
            "MEDIA:/tmp/data.json\n"
            "MEDIA:/tmp/image.png"
        )
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 3
        exts = {Path(m[0]).suffix for m in media}
        assert exts == {".pdf", ".json", ".png"}

    def test_extract_media_quoted_path(self):
        content = 'MEDIA:"/tmp/my file.pdf"'
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) >= 1

    def test_extract_media_tilde_path(self):
        content = "MEDIA:~/documents/report.md"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) >= 1
        assert media[0][0].endswith("report.md")

    def test_voice_directive_preserved(self):
        content = "[[audio_as_voice]]\nMEDIA:/tmp/speech.ogg"
        media, cleaned = BasePlatformAdapter.extract_media(content)
        assert len(media) == 1
        assert media[0][1] is True  # is_voice flag


# ===========================================================================
# extract_local_files() — expanded extension support
# ===========================================================================

class TestExtractLocalFilesExpanded:
    """Verify extract_local_files() detects bare paths with new extensions."""

    def _extract_with_mock(self, content, existing_paths=None):
        """Run extract_local_files with os.path.isfile mocked."""
        if existing_paths is None:
            # All paths exist
            with patch("os.path.isfile", return_value=True):
                return BasePlatformAdapter.extract_local_files(content)
        else:
            expanded = {str(Path(p).expanduser().resolve()) for p in existing_paths}
            with patch("os.path.isfile", side_effect=lambda p: p in expanded):
                return BasePlatformAdapter.extract_local_files(content)

    @pytest.mark.parametrize("ext", [
        "pdf", "txt", "md", "csv", "json", "yaml", "yml",
        "py", "js", "ts", "sh", "html", "css", "zip", "log",
    ])
    def test_bare_path_detected(self, ext):
        content = f"Check the file /tmp/data.{ext} for details"
        files, cleaned = self._extract_with_mock(content)
        assert any(f.endswith(f".{ext}") for f in files), (
            f"extract_local_files() missed .{ext}"
        )

    def test_original_image_extensions_still_work(self):
        content = "See /tmp/photo.png and /tmp/video.mp4"
        files, cleaned = self._extract_with_mock(content)
        assert len(files) == 2

    def test_code_block_paths_ignored(self):
        content = "```\n/tmp/data.json\n```"
        files, cleaned = self._extract_with_mock(content)
        assert len(files) == 0

    def test_inline_code_paths_ignored(self):
        content = "Run `cat /tmp/data.yaml` to see it"
        files, cleaned = self._extract_with_mock(content)
        assert len(files) == 0

    def test_url_paths_ignored(self):
        content = "Download from https://example.com/data.json"
        files, cleaned = self._extract_with_mock(content)
        assert len(files) == 0


# ===========================================================================
# Schema shape — regression tests for the double-wrap bug
# ===========================================================================
#
# HISTORY: SEND_FILE_SCHEMA was originally authored wrapped in an outer
# {"type": "function", "function": {...}} envelope — the only tool in the
# codebase to do so. The registry's get_definitions() adds that envelope
# itself, producing a double-wrapped schema where parameters sat two levels
# deep. The Anthropic adapter's convert_tools_to_anthropic() did
# fn.get("parameters", ...) on the outer dict and fell back to empty
# properties — so the model never saw file_path. See:
#   ~/.hermes/plans/hermes-patches/send-file-schema-unwrap.md
#
# These tests lock in the flat shape convention so future edits can't
# silently reintroduce the bug.
# ===========================================================================

class TestSendFileSchemaShape:
    """Regression tests for the schema double-wrap bug."""

    def test_schema_is_flat_not_wrapped(self):
        """SEND_FILE_SCHEMA must be flat — no outer {type:function,function:{}} envelope."""
        # The flat form has 'name', 'description', 'parameters' at top level.
        assert "name" in SEND_FILE_SCHEMA, (
            "SEND_FILE_SCHEMA missing top-level 'name' — likely double-wrapped. "
            "See hermes-patches/send-file-schema-unwrap.md"
        )
        assert "parameters" in SEND_FILE_SCHEMA, (
            "SEND_FILE_SCHEMA missing top-level 'parameters' — likely double-wrapped."
        )
        # Negative: must NOT have the outer function envelope.
        assert "function" not in SEND_FILE_SCHEMA, (
            "SEND_FILE_SCHEMA has outer 'function' key — this is the double-wrap bug. "
            "Unwrap to match every other tool in tools/*.py"
        )
        assert SEND_FILE_SCHEMA.get("type") != "function", (
            "SEND_FILE_SCHEMA has top-level type='function' — this is the outer envelope "
            "that causes the double-wrap bug."
        )

    def test_required_file_path_declared(self):
        """file_path must be declared required — otherwise model can call with empty args."""
        params = SEND_FILE_SCHEMA["parameters"]
        assert params["type"] == "object"
        assert "file_path" in params["properties"]
        assert params["properties"]["file_path"]["type"] == "string"
        assert "file_path" in params["required"]

    def test_registry_exposes_file_path_in_openai_format(self):
        """registry.get_definitions() must surface file_path in the OpenAI-format schema."""
        defs = registry.get_definitions({"send_file"})
        assert len(defs) == 1, "send_file not registered"
        entry = defs[0]
        # OpenAI format: {"type": "function", "function": {name, description, parameters}}
        assert entry["type"] == "function"
        fn = entry["function"]
        assert fn["name"] == "send_file"
        props = fn["parameters"]["properties"]
        assert "file_path" in props, (
            "file_path missing from OpenAI-format schema — registry not emitting params "
            "correctly. Check that SEND_FILE_SCHEMA is flat (see send-file-schema-unwrap.md)."
        )
        assert "caption" in props
        assert "file_path" in fn["parameters"]["required"]

    def test_anthropic_adapter_surfaces_file_path(self):
        """After convert_tools_to_anthropic(), file_path must be in input_schema.properties.

        This is the exact path that broke: registry -> get_definitions ->
        convert_tools_to_anthropic -> model. If any step drops the parameters,
        the model sees {"properties": {}} and calls the tool with empty args.
        """
        oai_defs = registry.get_definitions({"send_file"})
        ant_defs = convert_tools_to_anthropic(oai_defs)
        assert len(ant_defs) == 1
        tool = ant_defs[0]
        assert tool["name"] == "send_file"
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "file_path" in schema["properties"], (
            "file_path missing from Anthropic input_schema — this is the exact bug "
            "that caused send_file to be called with empty args. "
            "See hermes-patches/send-file-schema-unwrap.md"
        )
        assert schema["properties"]["file_path"]["type"] == "string"
        assert "file_path" in schema["required"]

    def test_anthropic_schema_has_no_empty_properties_fallback(self):
        """Guard against the empty-properties fallback in convert_tools_to_anthropic().

        The converter defaults to {"type":"object","properties":{}} when it
        can't find 'parameters'. A passing test that hits that fallback is
        silent failure. Assert we don't.
        """
        oai_defs = registry.get_definitions({"send_file"})
        ant_defs = convert_tools_to_anthropic(oai_defs)
        schema = ant_defs[0]["input_schema"]
        assert schema["properties"] != {}, (
            "Anthropic input_schema has empty properties — converter hit its "
            "fallback. Schema body is not reaching the model."
        )

    def test_all_tool_schemas_are_flat(self):
        """Every registered tool's raw schema must be flat (no outer function envelope).

        This catches the double-wrap bug for ANY tool, not just send_file.
        If a new tool lands with the wrong shape, this test fails loudly.
        """
        for name in registry.get_all_tool_names():
            schema = registry.get_schema(name)
            assert schema is not None, f"Tool {name} has no schema"
            # Flat form: name, description, parameters at top level.
            # Wrapped form (bug): {"type": "function", "function": {...}}
            assert schema.get("type") != "function" or "function" not in schema, (
                f"Tool {name!r} has wrapped schema ({{type:function, function:{{...}}}}). "
                f"Unwrap to flat form — see tools/send_message_tool.py for reference. "
                f"Refs: hermes-patches/send-file-schema-unwrap.md"
            )
            # Parameters must be at top level (or absent for zero-arg tools).
            if "parameters" not in schema:
                # Some tools legitimately have no params — but then properties
                # should be absent or empty at the top level too.
                continue
            assert isinstance(schema["parameters"], dict), (
                f"Tool {name!r} 'parameters' is not a dict"
            )


class TestSendFileHandlerValidatesEmptyPath:
    """Regression: empty string should not be treated as CWD."""

    def test_empty_path_gives_clear_error(self):
        """Path('').resolve() returns CWD, which is a directory — handler must reject.

        This is the symptom users saw: 'Not a file (maybe a directory?): '
        when the schema wasn't exposing file_path. The handler still hits
        this path if anyone calls with empty string. Make sure the error
        is informative.
        """
        result = send_file_tool("")
        parsed = json.loads(result)
        assert "error" in parsed
        # Error should mention the problem — either "not found" or "directory"
        # (depending on whether CWD exists, which it does in tests)
        err = parsed["error"].lower()
        assert "not found" in err or "directory" in err or "not a file" in err
