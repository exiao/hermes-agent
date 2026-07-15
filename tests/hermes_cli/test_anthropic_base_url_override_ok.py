"""Tests for hermes_cli.runtime_provider._anthropic_base_url_override_ok.

The guard decides whether a configured ``model.base_url`` (typically inherited
from ``providers.anthropic.base_url``) may back native ``provider: anthropic``
resolution, or whether it is a stale non-Anthropic leak that should be ignored
in favor of ``https://api.anthropic.com``.

The loopback branch is what lets a config-only proxy setup
(``providers.anthropic.base_url: http://127.0.0.1:18801``) route through the
local billing proxy for EVERY process (main, subagents, kanban workers, cron),
not just processes that happen to have ``ANTHROPIC_BASE_URL`` in their env.
It must never accept a non-Anthropic aggregator (OpenRouter, OpenAI, Codex).
"""

from __future__ import annotations

import pytest

from hermes_cli.runtime_provider import _anthropic_base_url_override_ok


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:18801",       # Hermes billing proxy
        "http://localhost:18801",
        "http://127.0.0.1:18801/",       # trailing slash
        "http://[::1]:18801",            # IPv6 loopback
        "https://api.anthropic.com",
        "https://foundry.example.azure.com",
        "https://gateway.example.com/anthropic",
    ],
)
def test_accepts_anthropic_and_loopback(url):
    assert _anthropic_base_url_override_ok(url, allow_loopback=True) is True


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://openrouter.ai/api/v1",              # stale aggregator leak
        "https://api.openai.com/v1",
        "https://chatgpt.com/backend-api/codex",     # Codex endpoint
        "https://api.kimi.com",                       # bare, no /coding
    ],
)
def test_rejects_non_anthropic_endpoints(url):
    assert _anthropic_base_url_override_ok(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434/v1",
        "http://localhost:1234/v1",
        "http://[::1]:1234/v1",
    ],
)
def test_rejects_loopback_model_base_url_without_anthropic_provider_config(url):
    """A leftover local OpenAI endpoint must not become an Anthropic proxy."""
    assert _anthropic_base_url_override_ok(url) is False
