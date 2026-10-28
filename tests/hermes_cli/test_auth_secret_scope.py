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


def test_credential_pool_seed_prefers_active_scope_over_raw_dotenv(monkeypatch):
    """Pool env seeding must use the resolved active scope, not raw .env.

    Regression for the #100 P2s on raw ``op://`` refs and stale plaintext .env
    values: ``_profile_runtime_scope`` has already resolved the correct profile
    value into ``get_secret``. If the pool seed prefers the file value first, it
    can persist ``op://...`` or stale plaintext instead of the scoped secret.
    """
    from agent import credential_pool

    monkeypatch.setattr(
        credential_pool,
        "load_env",
        lambda: {"OPENROUTER_API_KEY": "op://Private/Stale/key"},
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-default-leak")
    monkeypatch.setattr(
        "hermes_cli.auth.is_source_suppressed", lambda _p, _s: False
    )

    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"OPENROUTER_API_KEY": "sk-profile-resolved"})
    try:
        entries = []
        changed, sources = credential_pool._seed_from_env("openrouter", entries)
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert changed
    assert sources == {"env:OPENROUTER_API_KEY"}
    assert entries[0].access_token == "sk-profile-resolved"


def test_api_key_provider_fails_closed_when_multiplex_active_but_unscoped(
    monkeypatch,
):
    """Codex P1 on #100: when gateway.multiplex_profiles is on but this resolver
    is reached OUTSIDE a scope (current_secret_scope() is None), the dotenv-
    preferred read would return the process/default profile's .env key to an
    unscoped multiplex caller — defeating the fail-closed guard. For an API-key
    provider (e.g. kimi-coding-cn) it must instead route through get_secret and
    fail closed, returning no credential."""
    # get_env_value_prefer_dotenv would return the default profile's .env key —
    # it must NOT be reached under an unscoped multiplex call.
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value_prefer_dotenv",
        lambda _name: "default-profile-dotenv-key",
    )
    ss.set_multiplex_active(True)
    # No scope installed.
    try:
        credentials = resolve_api_key_provider_credentials("kimi-coding-cn")
    finally:
        ss.set_multiplex_active(False)

    assert not credentials.get("api_key"), (
        "unscoped multiplex call must fail closed, not return the default "
        "profile's dotenv key"
    )


def test_api_key_provider_unscoped_multiplex_skips_credential_pool(monkeypatch):
    """Codex P1 follow-on on #100: failing closed on the env-var loop is not
    enough — execution must return BEFORE the credential-pool fallback, which
    load_pool()._seed_from_env can still seed from the default profile's .env
    and return as credential_pool:<provider>. An unscoped multiplex call must
    resolve nothing at all."""
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value_prefer_dotenv",
        lambda _name: "",
    )

    class _Pool:
        def has_credentials(self):
            return True

        def peek(self):
            return type("E", (), {"access_token": "default-profile-pool-key"})()

    # If the pool fallback is reached, it would return the default profile's key.
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda _slug: _Pool())

    ss.set_multiplex_active(True)
    try:
        credentials = resolve_api_key_provider_credentials("kimi-coding-cn")
    finally:
        ss.set_multiplex_active(False)

    assert not credentials.get("api_key"), (
        "unscoped multiplex call must not borrow the default profile's "
        "credential-pool key"
    )


def test_api_key_provider_single_profile_still_reads_dotenv(monkeypatch):
    """Single-profile (multiplex OFF, no scope): the dotenv-preferred lookup is
    unchanged — an ordinary CLI/TUI run still resolves its key from .env."""
    monkeypatch.setattr(
        "hermes_cli.config.get_env_value_prefer_dotenv",
        lambda _name: "single-profile-dotenv-key",
    )
    ss.set_multiplex_active(False)
    credentials = resolve_api_key_provider_credentials("kimi-coding-cn")
    assert credentials.get("api_key") == "single-profile-dotenv-key"


def test_api_key_provider_base_url_fails_closed_when_unscoped_multiplex(monkeypatch):
    """Codex P2 on #100: resolve_api_key_provider_credentials must not crash by
    reaching get_secret() for the base URL on the unscoped multiplex path. Without
    a scope installed, get_secret raises UnscopedSecretError; the credential has
    already failed closed above, so the base URL must resolve empty (registry
    default) rather than raising."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    # openai-api has a base_url_env_var (OPENAI_BASE_URL).
    assert PROVIDER_REGISTRY["openai-api"].base_url_env_var

    ss.set_multiplex_active(True)
    # No scope installed → get_secret would raise UnscopedSecretError.
    try:
        credentials = resolve_api_key_provider_credentials("openai-api")
    finally:
        ss.set_multiplex_active(False)

    # Fails closed on the credential and does NOT crash on the base URL read.
    assert not credentials.get("api_key")
    assert credentials.get("base_url") == PROVIDER_REGISTRY["openai-api"].inference_base_url
