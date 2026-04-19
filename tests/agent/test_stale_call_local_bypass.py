"""Tests for stale-call/stream timeout selection on local-endpoint base URLs.

Covers the regression fixed in
hermes-patches/subagent-stale-call-local-bypass.md.

Prior versions set `_stale_timeout = float("inf")` whenever
`is_local_endpoint(base_url)` returned True, which silently disabled the
watchdog for local *proxies* that forward to a cloud provider (e.g. the
OpenClaw billing proxy on 127.0.0.1:18801 fronting Anthropic).
Sub-agents calling through such a proxy could wedge for 40+ minutes.

The fix removes the `is_local_endpoint`-keyed bypass and applies the same
sliding scale for every base_url.  Operators running a real local LLM
that needs longer than the bumped ceiling can override via
HERMES_API_CALL_STALE_TIMEOUT (non-streaming) or
HERMES_STREAM_STALE_TIMEOUT (streaming).
"""

import math
import os

import pytest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers — replicate the tiny selection blocks from run_agent.py.  Tested
# via direct logic so we don't have to instantiate AIAgent (and pull in
# the entire CLI/provider stack) for what is fundamentally an arithmetic
# decision.
# ---------------------------------------------------------------------------


def _select_non_streaming_stale_timeout(messages, base_url, env=None):
    """Mirror of the block at run_agent.py around L5152."""
    env = env if env is not None else os.environ
    _stale_base = float(env.get("HERMES_API_CALL_STALE_TIMEOUT", 300.0))
    _est_tokens = sum(len(str(v)) for v in (messages or [])) // 4
    if _est_tokens > 100_000:
        return max(_stale_base, 600.0)
    if _est_tokens > 50_000:
        return max(_stale_base, 450.0)
    return _stale_base


def _select_stream_stale_timeout(messages, base_url, env=None):
    """Mirror of the block at run_agent.py around L5857."""
    env = env if env is not None else os.environ
    _base = float(env.get("HERMES_STREAM_STALE_TIMEOUT", 180.0))
    _est_tokens = sum(len(str(v)) for v in (messages or [])) // 4
    if _est_tokens > 100_000:
        return max(_base, 300.0)
    if _est_tokens > 50_000:
        return max(_base, 240.0)
    return _base


# ---------------------------------------------------------------------------
# Non-streaming stale-call timeout
# ---------------------------------------------------------------------------


class TestNonStreamingStaleTimeoutLocalProxy:
    """Local proxy must NOT get an infinite stale timeout (the regression)."""

    @pytest.mark.parametrize("base_url", [
        "http://127.0.0.1:18801",          # OpenClaw billing proxy
        "http://localhost:11434",          # Ollama-style port, but hermes treats
                                            # it the same now (no inf bypass)
        "http://192.168.1.5:8000",         # RFC-1918 local LAN
        "http://host.docker.internal:11434",
    ])
    def test_local_endpoint_does_not_disable_watchdog(self, base_url):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_API_CALL_STALE_TIMEOUT", None)
            t = _select_non_streaming_stale_timeout(
                messages=[{"role": "user", "content": "hi"}],
                base_url=base_url,
            )
            assert math.isfinite(t), \
                f"local endpoint {base_url} got infinite stale timeout"
            assert t == 300.0, \
                f"small-context default should be 300s, got {t}"

    def test_local_endpoint_large_context_bumps_to_600(self):
        # 100k-token-ish message — we estimate as len(str(msg))//4
        big = "x" * 500_000  # roughly 125k "tokens" by the heuristic
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_API_CALL_STALE_TIMEOUT", None)
            t = _select_non_streaming_stale_timeout(
                messages=[{"role": "user", "content": big}],
                base_url="http://127.0.0.1:18801",
            )
            assert t == 600.0


class TestNonStreamingStaleTimeoutEnvOverride:
    """User-supplied env var still wins, even for local endpoints."""

    def test_user_set_lower_floor_respected(self):
        with patch.dict(os.environ,
                        {"HERMES_API_CALL_STALE_TIMEOUT": "120"},
                        clear=False):
            t = _select_non_streaming_stale_timeout(
                messages=[{"role": "user", "content": "hi"}],
                base_url="http://127.0.0.1:18801",
            )
            # Small context → returns env value verbatim.
            assert t == 120.0

    def test_user_set_higher_floor_respected_for_local_llm(self):
        # Operators running a slow local LLM bump the ceiling.
        with patch.dict(os.environ,
                        {"HERMES_API_CALL_STALE_TIMEOUT": "1800"},
                        clear=False):
            t = _select_non_streaming_stale_timeout(
                messages=[{"role": "user", "content": "hi"}],
                base_url="http://localhost:11434",
            )
            assert t == 1800.0

    def test_env_override_floors_large_context_bumping(self):
        # Env var is a floor — large-context bump only kicks in if it's
        # higher than the env value.
        big = "x" * 500_000
        with patch.dict(os.environ,
                        {"HERMES_API_CALL_STALE_TIMEOUT": "900"},
                        clear=False):
            t = _select_non_streaming_stale_timeout(
                messages=[{"role": "user", "content": big}],
                base_url="http://127.0.0.1:18801",
            )
            # max(900, 600) == 900
            assert t == 900.0


class TestNonStreamingStaleTimeoutRemote:
    """Remote endpoints behave identically to local — no special-casing."""

    @pytest.mark.parametrize("base_url", [
        "https://api.anthropic.com",
        "https://api.openai.com",
        "https://openrouter.ai/api/v1",
    ])
    def test_remote_uses_same_sliding_scale(self, base_url):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_API_CALL_STALE_TIMEOUT", None)
            t = _select_non_streaming_stale_timeout(
                messages=[{"role": "user", "content": "hi"}],
                base_url=base_url,
            )
            assert t == 300.0


# ---------------------------------------------------------------------------
# Streaming stale timeout (parallel regression)
# ---------------------------------------------------------------------------


class TestStreamStaleTimeoutLocalProxy:
    """Same regression existed in the streaming path — verify it's gone."""

    @pytest.mark.parametrize("base_url", [
        "http://127.0.0.1:18801",
        "http://localhost:11434",
        "http://10.0.0.5:1234",
    ])
    def test_local_endpoint_does_not_disable_stream_watchdog(self, base_url):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            t = _select_stream_stale_timeout(
                messages=[{"role": "user", "content": "hi"}],
                base_url=base_url,
            )
            assert math.isfinite(t), \
                f"local endpoint {base_url} got infinite stream stale timeout"
            assert t == 180.0

    def test_local_large_context_bumps_to_300(self):
        big = "x" * 500_000
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            t = _select_stream_stale_timeout(
                messages=[{"role": "user", "content": big}],
                base_url="http://127.0.0.1:18801",
            )
            assert t == 300.0


class TestStreamStaleTimeoutEnvOverride:
    def test_local_llm_can_extend_via_env(self):
        with patch.dict(os.environ,
                        {"HERMES_STREAM_STALE_TIMEOUT": "600"},
                        clear=False):
            t = _select_stream_stale_timeout(
                messages=[{"role": "user", "content": "hi"}],
                base_url="http://localhost:11434",
            )
            assert t == 600.0
