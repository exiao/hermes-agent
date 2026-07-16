"""Regression: a direct alias with its own base_url must reset base_url
provenance during a /model switch.

Scenario (PR #118): an Anthropic runtime inherited
``providers.anthropic.base_url`` (a loopback proxy), so credential resolution
sets ``base_url_from_provider_config=True``. When the selected model resolves to
a *direct alias* that carries its OWN ``base_url``, the direct-alias override
block replaces ``base_url`` with the alias endpoint — but historically left the
provenance flag True. The CLI/TUI persistence paths read that flag as "do not
write model.base_url", so ``/model <alias> --global`` saved the model while
DROPPING the alias endpoint; after restart the model ran against the proxy
instead of the alias URL.

Invariant: once the alias endpoint overrides base_url, base_url IS the source of
truth (``base_url_from_provider_config`` must be False) so the endpoint persists.
"""
from unittest.mock import patch

import hermes_cli.model_switch as ms
from hermes_cli.model_switch import DirectAlias, switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}

_ALIAS_URL = "https://alias.example.com/anthropic"
_PROXY_URL = "https://127.0.0.1:8788/anthropic"  # inherited providers.anthropic.base_url


def _run(monkeypatch):
    # A direct alias whose provider matches the current provider (anthropic) and
    # which carries its own base_url — this is what triggers the override block.
    monkeypatch.setattr(
        ms, "DIRECT_ALIASES", {"myalias": DirectAlias("claude-sonnet-4-6", "anthropic", _ALIAS_URL)}
    )
    with (
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "sk-ant-fake",
                "base_url": _PROXY_URL,
                # The inherited providers.anthropic.base_url provenance: this is
                # exactly the flag the override must reset once it replaces base_url.
                "base_url_from_provider_config": True,
                "api_mode": "anthropic_messages",
            },
        ),
        patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input="myalias",
            current_provider="anthropic",
            current_model="claude-opus-4-6",
            current_base_url=_PROXY_URL,
            current_api_key="sk-ant-fake",
            is_global=True,
        )


def test_direct_alias_base_url_override_resets_provenance(monkeypatch):
    result = _run(monkeypatch)

    assert result.success, f"switch_model failed: {result.error_message}"
    # The alias endpoint wins over the inherited proxy URL...
    assert result.base_url == _ALIAS_URL, (
        f"expected alias endpoint to override base_url; got {result.base_url}"
    )
    # ...and provenance flips to False so persistence writes model.base_url
    # (the alias URL) instead of dropping it and falling back to the proxy.
    assert result.base_url_from_provider_config is False, (
        "alias-overridden base_url must NOT be marked provider-config-owned, "
        "or /model --global drops the endpoint and reverts to the proxy on restart"
    )
