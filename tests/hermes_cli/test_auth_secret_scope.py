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
    import agent.credential_pool as credential_pool

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


def test_copilot_default_profile_scope_miss_keeps_gh_fallback(monkeypatch, tmp_path):
    """A scoped miss for the DEFAULT profile must still fall back to gh — the
    process gh credential is the default profile's own identity."""
    from hermes_cli.auth import _resolve_copilot_raw_token, PROVIDER_REGISTRY

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_DEFAULT_PROFILE_GH", "gh_cli"),
    )
    # Default-profile home: NOT under a "profiles/" segment.
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: default_home)
    pconfig = PROVIDER_REGISTRY["copilot"]

    ss.set_multiplex_active(True)
    # Scope active but NO Copilot key present (a miss).
    token = ss.set_secret_scope({"OPENAI_API_KEY": "sk-other"})
    try:
        raw, source = _resolve_copilot_raw_token(pconfig)
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert raw == "gho_DEFAULT_PROFILE_GH"
    assert source == "gh_cli"


def test_copilot_named_profile_scope_miss_is_authoritative(monkeypatch, tmp_path):
    """A scoped miss for a NAMED profile must be authoritative — it must never
    borrow the process/gh token (which is another profile's identity)."""
    from hermes_cli.auth import _resolve_copilot_raw_token, PROVIDER_REGISTRY

    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_OTHER_PROFILE_GH", "gh_cli"),
    )
    # Named-profile home: <root>/profiles/<name>.
    named_home = tmp_path / ".hermes" / "profiles" / "coder"
    named_home.mkdir(parents=True)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: named_home)
    pconfig = PROVIDER_REGISTRY["copilot"]

    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"OPENAI_API_KEY": "sk-other"})
    try:
        raw, source = _resolve_copilot_raw_token(pconfig)
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)

    assert raw == ""
    assert source == ""


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


def test_copilot_unscoped_multiplex_fails_closed(monkeypatch):
    """Codex P1 on #100: when gateway.multiplex_profiles is on but the Copilot
    resolver is reached OUTSIDE a scope (current_secret_scope() is None, so the
    scoped helper returns None), it must NOT fall through to resolve_copilot_token()
    — that reads the process env / gh auth token, i.e. the DEFAULT profile's GitHub
    identity, and would hand it to an unscoped multiplex caller. It must fail closed,
    matching the generic API-key path."""
    from hermes_cli.auth import _resolve_copilot_raw_token, PROVIDER_REGISTRY

    # The gh/process fallback would return the default profile's token — it must
    # NOT be reached under an unscoped multiplex call.
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho_DEFAULT_PROFILE_GH", "gh_cli"),
    )
    pconfig = PROVIDER_REGISTRY["copilot"]

    ss.set_multiplex_active(True)
    # No scope installed (current_secret_scope() is None).
    try:
        raw, source = _resolve_copilot_raw_token(pconfig)
    finally:
        ss.set_multiplex_active(False)

    assert raw == "", (
        "unscoped multiplex Copilot call must fail closed, not borrow the "
        "default profile's gh token"
    )
    assert source == ""


def test_copilot_unscoped_single_profile_still_falls_back_to_gh(monkeypatch):
    """Single-profile (multiplex OFF, no scope): the Copilot gh/process fallback
    is unchanged — an ordinary CLI/TUI run still resolves its token via gh."""
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
