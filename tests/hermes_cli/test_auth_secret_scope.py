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
