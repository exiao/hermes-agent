"""A provider-pinned delegation child can opt in to its own fallback chain.

`delegation.provider` suppresses inheritance of the parent's fallback chain on
purpose (#80450: a quiet child must not silently reroute onto the parent's
models mid-run). That left a pinned child with NO recovery at all, so a single
upstream 429 ended the run outright — observed 2026-09-05 on a meta-ai-pinned
child. `delegation.fallback_providers` is the explicit opt-in.
"""
from __future__ import annotations

import pytest

from tools.delegate_tool import _delegation_fallback_chain


CHAIN = [
    {"provider": "anthropic", "model": "claude-sonnet-5"},
    {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"},
]


def _cfg(monkeypatch, delegation: dict):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *a, **k: {"delegation": delegation},
    )


def test_unset_returns_none_so_a_bare_pin_still_fails_loud(monkeypatch):
    """The #80450 contract: a pin with no declared chain does not reroute."""
    _cfg(monkeypatch, {"provider": "meta-ai"})
    assert _delegation_fallback_chain() is None


def test_empty_list_returns_none(monkeypatch):
    _cfg(monkeypatch, {"provider": "meta-ai", "fallback_providers": []})
    assert _delegation_fallback_chain() is None


def test_configured_chain_is_returned(monkeypatch):
    _cfg(monkeypatch, {"provider": "meta-ai", "fallback_providers": CHAIN})
    assert _delegation_fallback_chain() == CHAIN


def test_malformed_entries_are_dropped(monkeypatch):
    """A half-built entry must never reach AIAgent."""
    _cfg(monkeypatch, {"provider": "meta-ai", "fallback_providers": [
        {"provider": "anthropic"},              # no model
        {"model": "claude-sonnet-5"},           # no provider
        "openrouter/glm",                       # not a dict
        {"provider": "xai", "model": "grok-4.6"},
    ]})
    assert _delegation_fallback_chain() == [
        {"provider": "xai", "model": "grok-4.6"}
    ]


def test_non_string_entries_are_normalized(monkeypatch):
    """Fallback activation must not call strip on raw YAML scalars."""
    _cfg(monkeypatch, {"provider": "meta-ai", "fallback_providers": [
        {"provider": 123, "model": 456},
        {"provider": "  xai  ", "model": "  grok-4.6  "},
    ]})
    assert _delegation_fallback_chain() == [
        {"provider": "123", "model": "456"},
        {"provider": "xai", "model": "grok-4.6"},
    ]


def test_all_entries_malformed_returns_none(monkeypatch):
    _cfg(monkeypatch, {"provider": "meta-ai",
                       "fallback_providers": [{"provider": "anthropic"}]})
    assert _delegation_fallback_chain() is None


def test_non_list_value_returns_none(monkeypatch):
    """A scalar typo must not crash delegation."""
    _cfg(monkeypatch, {"provider": "meta-ai",
                       "fallback_providers": "anthropic"})
    assert _delegation_fallback_chain() is None


def test_config_failure_returns_none(monkeypatch):
    """Config trouble degrades to today's behaviour, never an exception."""
    def _boom(*a, **k):
        raise RuntimeError("config unreadable")
    monkeypatch.setattr("hermes_cli.config.load_config", _boom)
    assert _delegation_fallback_chain() is None
