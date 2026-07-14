"""Credential resolution respects an active profile secret scope."""

from agent import secret_scope as ss
from hermes_cli.auth import resolve_api_key_provider_credentials


def test_api_key_provider_prefers_active_secret_scope(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value_prefer_dotenv",
        lambda _name: "stale-dotenv-key",
    )
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"OPENAI_API_KEY": "profile-scoped-key"})
    try:
        credentials = resolve_api_key_provider_credentials("openai-api")
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert credentials["api_key"] == "profile-scoped-key"


def test_copilot_catalog_token_uses_active_scope_not_gh_fallback(monkeypatch):
    """The Copilot catalog fetch must resolve its GitHub token from the active
    profile scope, not the process env / ``gh auth token`` fallback.

    Regression for the #100 P2: model discovery marked Copilot available from a
    profile's scoped COPILOT_GITHUB_TOKEN, then resolve_api_key_provider_credentials
    fell through to resolve_copilot_token() which reads process env / gh, so the
    catalog could be fetched (and cached) with the DEFAULT profile's GitHub
    identity. With a scope active, the scoped token must win.
    """
    from hermes_cli.auth import _resolve_copilot_raw_token, PROVIDER_REGISTRY

    # Would be returned by the process/gh fallback — must NOT be chosen.
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_PROCESS_DEFAULT_PROFILE", "gh_cli"),
    )
    pconfig = PROVIDER_REGISTRY["copilot"]

    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"COPILOT_GITHUB_TOKEN": "gho_PROFILE_SCOPED"})
    try:
        raw, source = _resolve_copilot_raw_token(pconfig)
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert raw == "gho_PROFILE_SCOPED"
    assert source == "COPILOT_GITHUB_TOKEN"


def test_copilot_raw_token_falls_back_to_gh_without_scope(monkeypatch):
    """With no active scope (single-profile CLI/TUI), the Copilot token still
    resolves via the gh/process fallback exactly as before."""
    from hermes_cli.auth import _resolve_copilot_raw_token, PROVIDER_REGISTRY

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_GH_FALLBACK", "gh_cli"),
    )
    pconfig = PROVIDER_REGISTRY["copilot"]

    ss.set_multiplex_active(False)
    raw, source = _resolve_copilot_raw_token(pconfig)

    assert raw == "gho_GH_FALLBACK"
    assert source == "gh_cli"
