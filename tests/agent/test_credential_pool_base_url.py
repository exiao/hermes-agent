"""Tests for credential pool honoring providers.<provider>.base_url overrides.

Regression coverage for
~/.hermes/plans/hermes-patches/credential-pool-honors-provider-base-url.md

Without these guards, env-seeded entries (and claude_code/hermes_pkce
entries) get the PROVIDER_REGISTRY default base_url baked in
(``https://api.anthropic.com``), which silently overrides the user's
configured local proxy on every credential swap.  Sub-agents on a
Claude-Code-via-proxy setup were the worst-affected — see session
20260419_005457_cd4516 for a 17-min hang.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def _setup_hermes_home(tmp_path, monkeypatch, config_yaml: str = "") -> None:
    """Point HERMES_HOME at a tmp dir and optionally write a config.yaml."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    if config_yaml:
        (hermes_home / "config.yaml").write_text(config_yaml)


def test_get_configured_provider_base_url_returns_user_value(tmp_path, monkeypatch):
    _setup_hermes_home(
        tmp_path, monkeypatch,
        config_yaml=(
            "providers:\n"
            "  anthropic:\n"
            "    base_url: http://127.0.0.1:18801\n"
            "    api_key: sk-ant-test\n"
        ),
    )
    from agent.credential_pool import _get_configured_provider_base_url, _load_config_safe
    _load_config_safe.cache_clear() if hasattr(_load_config_safe, "cache_clear") else None
    assert _get_configured_provider_base_url("anthropic") == "http://127.0.0.1:18801"


def test_get_configured_provider_base_url_returns_none_when_unset(tmp_path, monkeypatch):
    _setup_hermes_home(tmp_path, monkeypatch, config_yaml="providers:\n  openai:\n    api_key: sk-test\n")
    from agent.credential_pool import _get_configured_provider_base_url, _load_config_safe
    if hasattr(_load_config_safe, "cache_clear"):
        _load_config_safe.cache_clear()
    assert _get_configured_provider_base_url("anthropic") is None


def test_get_configured_provider_base_url_strips_trailing_slash(tmp_path, monkeypatch):
    _setup_hermes_home(
        tmp_path, monkeypatch,
        config_yaml=(
            "providers:\n"
            "  anthropic:\n"
            "    base_url: http://127.0.0.1:18801/\n"
        ),
    )
    from agent.credential_pool import _get_configured_provider_base_url, _load_config_safe
    if hasattr(_load_config_safe, "cache_clear"):
        _load_config_safe.cache_clear()
    assert _get_configured_provider_base_url("anthropic") == "http://127.0.0.1:18801"


def test_swap_credential_prefers_configured_proxy_over_entry_base_url(tmp_path, monkeypatch):
    """The actual bug: pool entry says api.anthropic.com, user says proxy URL.

    Swap must end up with the proxy URL, not the entry URL.
    """
    _setup_hermes_home(
        tmp_path, monkeypatch,
        config_yaml=(
            "providers:\n"
            "  anthropic:\n"
            "    base_url: http://127.0.0.1:18801\n"
        ),
    )
    from agent.credential_pool import _load_config_safe
    if hasattr(_load_config_safe, "cache_clear"):
        _load_config_safe.cache_clear()

    # Build a minimal AIAgent stand-in with just the fields _swap_credential touches.
    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent.base_url = "http://127.0.0.1:18801"
    agent.api_key = "old-key"
    agent._anthropic_api_key = "old-key"
    agent._anthropic_base_url = "http://127.0.0.1:18801"
    agent._anthropic_client = MagicMock()
    agent._is_anthropic_oauth = False

    # Entry seeded with the canonical Anthropic URL (the bug-trigger shape)
    entry = MagicMock()
    entry.runtime_api_key = "claude-code-oauth-token"
    entry.runtime_base_url = "https://api.anthropic.com"
    entry.base_url = "https://api.anthropic.com"
    entry.access_token = "claude-code-oauth-token"

    with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
        mock_build.return_value = MagicMock()
        agent._swap_credential(entry)

    assert agent.base_url == "http://127.0.0.1:18801", (
        f"_swap_credential leaked entry URL into self.base_url: {agent.base_url!r}"
    )
    assert agent._anthropic_base_url == "http://127.0.0.1:18801", (
        f"_swap_credential leaked entry URL into _anthropic_base_url: "
        f"{agent._anthropic_base_url!r}"
    )
    # Anthropic client must be rebuilt with the proxy URL, not the entry URL.
    mock_build.assert_called_once()
    _, build_args, _ = mock_build.mock_calls[0]
    assert build_args[1] == "http://127.0.0.1:18801"


def test_swap_credential_uses_entry_url_when_no_user_override(tmp_path, monkeypatch):
    """Backward-compat: when user has no providers.<provider>.base_url, the
    entry's base_url still wins (existing behavior).
    """
    _setup_hermes_home(tmp_path, monkeypatch, config_yaml="")
    from agent.credential_pool import _load_config_safe
    if hasattr(_load_config_safe, "cache_clear"):
        _load_config_safe.cache_clear()

    from run_agent import AIAgent
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent.base_url = "https://api.anthropic.com"
    agent.api_key = "old-key"
    agent._anthropic_api_key = "old-key"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._anthropic_client = MagicMock()
    agent._is_anthropic_oauth = False

    entry = MagicMock()
    entry.runtime_api_key = "new-key"
    entry.runtime_base_url = "https://api.anthropic.com"
    entry.base_url = "https://api.anthropic.com"
    entry.access_token = "new-key"

    with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
        mock_build.return_value = MagicMock()
        agent._swap_credential(entry)

    assert agent.base_url == "https://api.anthropic.com"
    assert agent._anthropic_base_url == "https://api.anthropic.com"
