"""Tests for video_analyze tool in tools/vision_tools.py."""
import pytest

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from tools.vision_tools import (
    _detect_video_mime_type,
    _video_to_base64_data_url,
    _handle_video_analyze,
    _MAX_VIDEO_BASE64_BYTES,
    video_analyze_tool,
    VIDEO_ANALYZE_SCHEMA,
)


# ---------------------------------------------------------------------------
# _detect_video_mime_type
# ---------------------------------------------------------------------------


class TestDetectVideoMimeType:
    """Extension-based MIME detection for video files."""

    def test_mp4(self, tmp_path):
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_webm(self, tmp_path):
        p = tmp_path / "clip.webm"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/webm"


    def test_case_insensitive(self, tmp_path):
        p = tmp_path / "clip.MP4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"


# ---------------------------------------------------------------------------
# _video_to_base64_data_url
# ---------------------------------------------------------------------------


class TestVideoToBase64DataUrl:
    """Base64 encoding of video files."""

    def test_produces_data_url(self, tmp_path):
        p = tmp_path / "test.mp4"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p)
        assert result.startswith("data:video/mp4;base64,")


    def test_default_mime_for_unknown_ext(self, tmp_path):
        p = tmp_path / "test.xyz"
        p.write_bytes(b"\x00\x01\x02\x03")
        result = _video_to_base64_data_url(p)
        # Falls back to video/mp4
        assert result.startswith("data:video/mp4;base64,")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestVideoAnalyzeSchema:
    """Schema structure is correct."""

    def test_schema_name(self):
        assert VIDEO_ANALYZE_SCHEMA["name"] == "video_analyze"


    def test_schema_description_mentions_video(self):
        assert "video" in VIDEO_ANALYZE_SCHEMA["description"].lower()


# ---------------------------------------------------------------------------
# _handle_video_analyze handler
# ---------------------------------------------------------------------------


class TestHandleVideoAnalyze:
    """Tests for the registry handler wrapper."""

    def test_returns_awaitable(self, tmp_path, monkeypatch):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00" * 100)
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "test"})
            result = _handle_video_analyze({"video_url": str(video_file), "question": "what is this?"})
            # Should return an awaitable (coroutine)
            assert asyncio.iscoroutine(result)
            # Clean up the unawaited coroutine
            result.close()


    def test_falls_back_to_vision_model_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "google/gemini-flash")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({"video_url": "/tmp/test.mp4", "question": "test"})
            )
            args = mock_tool.call_args[0]
            assert args[2] == "google/gemini-flash"


# ---------------------------------------------------------------------------
# video_analyze_tool — integration-style tests with mocked LLM
# ---------------------------------------------------------------------------


class TestVideoAnalyzeTool:
    """Core video analysis function tests."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_local_file_success(self, tmp_path, monkeypatch):
        """Analyze a local video file — happy path."""
        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A short video showing a demo."

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="A short video showing a demo."):
                result = self._run(video_analyze_tool(str(video), "What is this?"))

        data = json.loads(result)
        assert data["success"] is True
        assert "demo" in data["analysis"].lower()

    def test_local_file_read_guard_blocks_env_via_video_extension(self, tmp_path):
        """A .env file symlinked with a video extension must still be blocked.

        _detect_video_mime_type only checks the file extension, not file
        content, so without a read guard a model could point video_url at
        any credential-store file (renamed/symlinked to look like a video)
        and have its raw bytes base64-encoded and sent to the vision
        provider. Regression for the shared agent.file_safety chokepoint
        added to video_analyze_tool's local-file branch.
        """
        secret = tmp_path / ".env"
        secret.write_text("OPENAI_API_KEY=sk-super-secret\n", encoding="utf-8")
        disguised = tmp_path / "video.mp4"
        disguised.symlink_to(secret)

        with patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as mock_llm:
            result = self._run(video_analyze_tool(str(disguised), "What is this?"))

        data = json.loads(result)
        assert data["success"] is False
        assert "secret-bearing environment file" in data["error"]
        mock_llm.assert_not_awaited()


    def test_unsupported_format(self, tmp_path):
        """Unsupported extension raises error."""
        video = tmp_path / "clip.flv"
        video.write_bytes(b"\x00" * 100)

        result = self._run(video_analyze_tool(str(video), "What is this?"))
        data = json.loads(result)
        assert data["success"] is False
        assert "unsupported video format" in data["analysis"].lower()


    def test_api_message_format(self, tmp_path):
        """Verify the message sent to LLM uses video_url content type."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "OK"
            return mock_response

        with patch("tools.vision_tools.async_call_llm", side_effect=capture_llm):
            with patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
                self._run(video_analyze_tool(str(video), "Describe this"))

        messages = captured_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "video_url"
        assert "video_url" in content[1]
        assert content[1]["video_url"]["url"].startswith("data:video/mp4;base64,")
        # No hardcoded output cap — the aux client omits max_tokens so the
        # provider uses its full output budget (max-tokens-knob policy).
        assert "max_tokens" not in captured_kwargs

    def test_non_local_backend_reads_video_from_terminal_backend(self, tmp_path, monkeypatch):
        """Non-local terminal backends must not read local host video paths.

        The read routes through the shared media resolver
        (tools.image_source, ``permitted=("video",)``) which exec-reads the
        bytes inside the sandbox — so the analyzed video is the container's
        file, never the host's.
        """
        host_video = tmp_path / "clip.mp4"
        host_video.write_bytes(b"HOST-VIDEO")
        remote_bytes = b"REMOTE-SANDBOX-VIDEO"
        remote_b64 = base64.b64encode(remote_bytes).decode("ascii")
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        import tools.image_source as isrc
        import tools.terminal_tool as tt

        env_lookups = []

        def fake_get_active(task_id):
            env_lookups.append(task_id)
            return SimpleNamespace(
                execute=lambda cmd, **kw: {"returncode": 0, "output": remote_b64}
            )

        monkeypatch.setattr(tt, "ensure_task_env", lambda *a, **k: None)
        monkeypatch.setattr(isrc, "_get_active_env", fake_get_active)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "sandbox video"
            return mock_response

        with (
            patch("tools.vision_tools.async_call_llm", side_effect=capture_llm),
            patch("tools.vision_tools.extract_content_or_reasoning", return_value="sandbox video"),
        ):
            result = self._run(
                video_analyze_tool(str(host_video), "Describe this", task_id="task-123")
            )

        data = json.loads(result)
        assert data["success"] is True
        assert env_lookups == ["task-123"]
        video_url = captured_kwargs["messages"][0]["content"][1]["video_url"]["url"]
        uploaded_bytes = base64.b64decode(video_url.split(",", 1)[1])
        assert uploaded_bytes == remote_bytes
        assert uploaded_bytes != host_video.read_bytes()


# ---------------------------------------------------------------------------
# Toolset registration
# ---------------------------------------------------------------------------


class TestVideoToolsetRegistration:
    """Verify the tool is registered correctly."""

    def test_registered_in_video_toolset(self):
        from tools.registry import registry
        entry = registry.get_entry("video_analyze")
        assert entry is not None
        assert entry.toolset == "video"
        assert entry.is_async is True
        assert entry.emoji == "🎬"

    def test_in_core_tools(self):
        """video_analyze ships in _HERMES_CORE_TOOLS (video toolset enabled by
        default — see patch note enable-video-toolset-default.md)."""
        from toolsets import _HERMES_CORE_TOOLS
        assert "video_analyze" in _HERMES_CORE_TOOLS

    def test_in_video_toolset_definition(self):
        """Toolset 'video' should contain video_analyze."""
        from toolsets import TOOLSETS
        assert "video" in TOOLSETS
        assert "video_analyze" in TOOLSETS["video"]["tools"]


# ---------------------------------------------------------------------------
# Content validation + streaming-host extraction (fix/video-content-validation)
# ---------------------------------------------------------------------------


from tools.vision_tools import (  # noqa: E402
    _looks_like_video_bytes,
    _is_streaming_host,
    _download_video,
    _extract_stream_to_file,
)


class TestLooksLikeVideoBytes:
    """Reject non-video payloads (HTML pages) before they reach the model."""

    def test_html_doctype_rejected(self):
        assert _looks_like_video_bytes(b"<!DOCTYPE html><html><head>") is False

    def test_html_tag_rejected(self):
        assert _looks_like_video_bytes(b"<html lang='en'>") is False

    def test_leading_whitespace_html_rejected(self):
        assert _looks_like_video_bytes(b"   \n<!doctype html>") is False

    def test_json_rejected(self):
        assert _looks_like_video_bytes(b'{"error": "not found"}') is False

    def test_empty_rejected(self):
        assert _looks_like_video_bytes(b"") is False

    def test_mp4_ftyp_accepted(self):
        assert _looks_like_video_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00") is True

    def test_webm_ebml_accepted(self):
        assert _looks_like_video_bytes(b"\x1aE\xdf\xa3\x01\x00\x00\x00") is True

    def test_avi_riff_accepted(self):
        assert _looks_like_video_bytes(b"RIFF\x00\x00\x00\x00AVI ") is True

    def test_mpeg_ps_accepted(self):
        assert _looks_like_video_bytes(b"\x00\x00\x01\xba\x21\x00") is True

    def test_mov_moov_accepted(self):
        # QuickTime files can lead with 'moov'/'mdat' instead of 'ftyp'.
        assert _looks_like_video_bytes(b"\x00\x00\x00\x18moov\x00\x00\x00\x00") is True

    def test_mov_mdat_accepted(self):
        assert _looks_like_video_bytes(b"\x00\x00\x10\x00mdat\x00\x00\x00\x00") is True

    def test_riff_webp_rejected(self):
        # RIFF also wraps WebP images; only AVI at offset 8 is a video.
        assert _looks_like_video_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ") is False

    def test_riff_wav_rejected(self):
        assert _looks_like_video_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ") is False


class TestIsStreamingHost:
    """Detect URLs that serve HTML watch pages instead of direct media."""

    def test_youtube_watch(self):
        assert _is_streaming_host("https://www.youtube.com/watch?v=abc123") is True

    def test_youtu_be(self):
        assert _is_streaming_host("https://youtu.be/abc123") is True

    def test_x_status(self):
        assert _is_streaming_host("https://x.com/user/status/1") is True

    def test_twitter_status(self):
        assert _is_streaming_host("https://twitter.com/user/status/1") is True

    def test_vimeo(self):
        assert _is_streaming_host("https://vimeo.com/12345") is True

    def test_direct_mp4_not_streaming(self):
        assert _is_streaming_host("https://cdn.example.com/clip.mp4") is False

    def test_lookalike_domain_not_streaming(self):
        # Must not match a domain that merely contains 'youtube'.
        assert _is_streaming_host("https://fakeyoutube.evil.com/x") is False


class TestDownloadVideoRejectsNonVideo:
    """_download_video must raise on HTML/text payloads, not pass them through."""

    def _mk_response(self, *, content, content_type):
        resp = MagicMock()
        resp.headers = {"content-type": content_type, "content-length": str(len(content))}
        resp.content = content
        resp.url = "https://example.com/fake.mp4"
        resp.is_redirect = False
        resp.next_request = None
        resp.raise_for_status = MagicMock()
        return resp

    def _patch_client(self, resp):
        """Mock the streaming download API (`client.stream("GET", ...)`).

        Upstream moved _download_video off `client.get` onto a chunked
        `client.stream` context manager (bounded memory + running size cap);
        the non-video payload rejection this class pins is unchanged, so the
        mock follows the new shape.
        """
        body = resp.content

        async def _aiter_bytes():
            yield body

        resp.aiter_bytes = _aiter_bytes

        class _StreamCM:
            async def __aenter__(self_inner):
                return resp

            async def __aexit__(self_inner, *exc):
                return False

        client = AsyncMock()
        client.stream = MagicMock(return_value=_StreamCM())
        client.get = AsyncMock(return_value=resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        return client

    def test_html_content_type_rejected(self, tmp_path):
        resp = self._mk_response(
            content=b"<!DOCTYPE html><html>...</html>", content_type="text/html")
        dest = tmp_path / "out.mp4"
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=self._patch_client(resp)), \
             patch("tools.vision_tools.check_website_access", return_value=None):
            with pytest.raises(Exception) as exc:
                asyncio.run(_download_video("https://example.com/page", dest, max_retries=1))
            assert "non-video" in str(exc.value).lower() or "not a recognized" in str(exc.value).lower()
        assert not dest.exists()

    def test_html_bytes_without_content_type_rejected(self, tmp_path):
        # No giveaway Content-Type, but the body is clearly HTML.
        resp = self._mk_response(
            content=b"<html><head><title>YouTube</title></head>", content_type="")
        dest = tmp_path / "out.mp4"
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=self._patch_client(resp)), \
             patch("tools.vision_tools.check_website_access", return_value=None):
            with pytest.raises(Exception) as exc:
                asyncio.run(_download_video("https://example.com/page", dest, max_retries=1))
            assert "not a recognized" in str(exc.value).lower()
        assert not dest.exists()

    def test_real_video_bytes_accepted(self, tmp_path):
        body = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 256
        resp = self._mk_response(content=body, content_type="video/mp4")
        dest = tmp_path / "out.mp4"
        with patch("tools.url_safety.create_ssrf_safe_async_client", return_value=self._patch_client(resp)), \
             patch("tools.vision_tools.check_website_access", return_value=None):
            result = asyncio.run(_download_video("https://example.com/clip.mp4", dest, max_retries=1))
            assert result == dest
            assert dest.read_bytes() == body


class TestExtractStreamMissingYtdlp:
    """When yt-dlp is absent, fail honestly with a clear message."""

    def test_no_ytdlp_raises_clear_error(self, tmp_path):
        dest = tmp_path / "out.mp4"
        with patch("tools.vision_tools.shutil.which", return_value=None):
            with pytest.raises(RuntimeError) as exc:
                asyncio.run(_extract_stream_to_file("https://youtube.com/watch?v=x", dest))
            assert "yt-dlp" in str(exc.value).lower()


class TestExtractStreamPicksVideoFile:
    """yt-dlp output selection: ignore sidecars, keep the real container ext."""

    def _fake_proc(self, returncode=0):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = returncode
        proc.kill = MagicMock()
        return proc

    def _run(self, tmp_path, produced_names):
        dest = tmp_path / "temp_video_abc.mp4"

        async def fake_exec(*args, **kwargs):
            # Simulate yt-dlp writing files with the destination stem.
            for name in produced_names:
                (tmp_path / name).write_bytes(b"\x00\x00\x00\x18ftypmp42")
            return self._fake_proc()

        with patch("tools.vision_tools.shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            return asyncio.run(_extract_stream_to_file("https://youtube.com/watch?v=x", dest))

    def test_ignores_sidecar_files(self, tmp_path):
        # .info.json sorts before .mp4 alphabetically; must not be selected.
        result = self._run(tmp_path, ["temp_video_abc.info.json", "temp_video_abc.mp4"])
        assert result.suffix == ".mp4"
        assert result.name == "temp_video_abc.mp4"

    def test_preserves_webm_extension(self, tmp_path):
        # yt-dlp fell back to webm; we must keep .webm, not force .mp4.
        result = self._run(tmp_path, ["temp_video_abc.webm"])
        assert result.suffix == ".webm"

    def test_no_video_output_raises(self, tmp_path):
        with pytest.raises(RuntimeError) as exc:
            self._run(tmp_path, ["temp_video_abc.info.json", "temp_video_abc.jpg"])
        assert "no video output" in str(exc.value).lower()

    def test_no_config_flag_passed(self, tmp_path):
        dest = tmp_path / "temp_video_abc.mp4"
        captured = {}

        async def fake_exec(*args, **kwargs):
            captured["args"] = args
            (tmp_path / "temp_video_abc.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
            return self._fake_proc()

        with patch("tools.vision_tools.shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            asyncio.run(_extract_stream_to_file("https://youtube.com/watch?v=x", dest))
        assert "--no-config" in captured["args"]


from tools.vision_tools import _is_direct_media_url  # noqa: E402


class TestIsDirectMediaUrl:
    """Direct media URLs on streaming hosts should bypass yt-dlp."""

    def test_reddit_dash_mp4(self):
        assert _is_direct_media_url("https://v.redd.it/abc/DASH_720.mp4") is True

    def test_webm_path(self):
        assert _is_direct_media_url("https://cdn.example.com/clip.webm") is True

    def test_youtube_watch_not_direct(self):
        assert _is_direct_media_url("https://youtube.com/watch?v=abc") is False

    def test_query_string_ignored(self):
        # Extension lives in the path, not the query.
        assert _is_direct_media_url("https://x.com/i/status/1?foo=bar") is False


class TestExtractStreamPolicyRecheck:
    """yt-dlp-resolved media URLs must pass the website-access policy."""

    def _fake_proc(self, *, returncode=0, stdout=b""):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(stdout, b""))
        proc.returncode = returncode
        proc.kill = MagicMock()
        return proc

    def test_blocked_resolved_url_raises(self, tmp_path):
        dest = tmp_path / "temp_video_abc.mp4"
        resolve_proc = self._fake_proc(stdout=b"https://blocked-cdn.internal/stream.mp4\n")

        async def fake_exec(*args, **kwargs):
            return resolve_proc

        def fake_policy(url):
            if "blocked-cdn" in url:
                return {"message": "blocked host"}
            return None

        with patch("tools.vision_tools.shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("asyncio.create_subprocess_exec", side_effect=fake_exec), \
             patch("tools.vision_tools.check_website_access", side_effect=fake_policy):
            with pytest.raises(PermissionError) as exc:
                asyncio.run(_extract_stream_to_file("https://youtube.com/watch?v=x", dest))
            assert "blocked" in str(exc.value).lower()

    def test_allowed_resolved_url_proceeds(self, tmp_path):
        dest = tmp_path / "temp_video_abc.mp4"
        calls = {"n": 0}

        async def fake_exec(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return self._fake_proc(stdout=b"https://ok-cdn.example/stream.mp4\n")
            # Second call is the real download; write the output file.
            (tmp_path / "temp_video_abc.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
            return self._fake_proc()

        with patch("tools.vision_tools.shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("asyncio.create_subprocess_exec", side_effect=fake_exec), \
             patch("tools.vision_tools.check_website_access", return_value=None):
            result = asyncio.run(_extract_stream_to_file("https://youtube.com/watch?v=x", dest))
            assert result.suffix == ".mp4"
            assert calls["n"] == 2


class TestExtractStreamNonZeroExitPolicy:
    """A non-zero yt-dlp exit must fail fast when empty, and still enforce the
    policy on any URLs it did resolve (no silent bypass)."""

    def _fake_proc(self, *, returncode=0, stdout=b"", stderr=b""):
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
        proc.returncode = returncode
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        return proc

    def test_nonzero_exit_no_output_raises(self, tmp_path):
        dest = tmp_path / "temp_video_abc.mp4"
        resolve_proc = self._fake_proc(returncode=1, stdout=b"", stderr=b"boom")

        async def fake_exec(*args, **kwargs):
            return resolve_proc

        with patch("tools.vision_tools.shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("asyncio.create_subprocess_exec", side_effect=fake_exec), \
             patch("tools.vision_tools.check_website_access", return_value=None):
            with pytest.raises(RuntimeError) as exc:
                asyncio.run(_extract_stream_to_file("https://youtube.com/watch?v=x", dest))
            assert "failed to resolve" in str(exc.value).lower()

    def test_nonzero_exit_with_blocked_url_still_enforced(self, tmp_path):
        # yt-dlp warns (exit 1) but still printed a resolved URL; the policy
        # check must run on it rather than being skipped.
        dest = tmp_path / "temp_video_abc.mp4"
        resolve_proc = self._fake_proc(
            returncode=1, stdout=b"https://blocked-cdn.internal/stream.mp4\n"
        )

        async def fake_exec(*args, **kwargs):
            return resolve_proc

        def fake_policy(url):
            return {"message": "blocked host"} if "blocked-cdn" in url else None

        with patch("tools.vision_tools.shutil.which", return_value="/usr/bin/yt-dlp"), \
             patch("asyncio.create_subprocess_exec", side_effect=fake_exec), \
             patch("tools.vision_tools.check_website_access", side_effect=fake_policy):
            with pytest.raises(PermissionError) as exc:
                asyncio.run(_extract_stream_to_file("https://youtube.com/watch?v=x", dest))
            assert "blocked" in str(exc.value).lower()


class TestDirectMediaPreservesSuffix:
    """Direct streaming-host media keeps its real container extension so the
    suffix-based MIME detection downstream doesn't mislabel it as mp4."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_webm_url_keeps_webm_suffix(self, tmp_path, monkeypatch):
        captured = {}

        async def fake_validate(url):
            return True

        async def fake_download(url, destination, **kwargs):
            captured["suffix"] = destination.suffix
            destination.write_bytes(b"\x1aE\xdf\xa3" + b"\x00" * 64)
            return destination

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"

        monkeypatch.setattr("tools.vision_tools.get_hermes_dir", lambda *a, **k: tmp_path)
        with patch("tools.vision_tools._validate_image_url_async", side_effect=fake_validate), \
             patch("tools.vision_tools._download_video", side_effect=fake_download), \
             patch("tools.vision_tools.check_website_access", return_value=None), \
             patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock, return_value=mock_response), \
             patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            result = self._run(video_analyze_tool("https://v.redd.it/abc/clip.webm", "What?"))

        data = json.loads(result)
        assert data["success"] is True
        assert captured["suffix"] == ".webm"
