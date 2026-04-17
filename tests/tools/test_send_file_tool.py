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

from tools.send_file_tool import send_file_tool, MAX_FILE_SIZE
from gateway.platforms.base import BasePlatformAdapter


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
