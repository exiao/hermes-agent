"""Profile-scoped credential resolution for multi-profile gateway multiplexing.

The multiplexing gateway serves many profiles from one process; each profile's
``.env`` keys **cannot** be unioned into ``os.environ`` (profile A's keys would
leak into profile B's turns and subprocesses). This module is a fail-closed,
context-local secret scope: ``set_secret_scope(mapping)`` installs the active
profile's secrets for the current task (a contextvar, so it propagates into the
agent's worker thread via ``copy_context()``); ``get_secret(name)`` reads from
it and, when multiplexing is active with no scope set, RAISES rather than
falling back to ``os.environ``. Design: ``docs/design/multiplexing-gateway.md``.
"""
from __future__ import annotations

import os
import re
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Dict, Mapping, Optional


# Process-global (describes the deployment mode, not a per-task value): set once
# at gateway startup when gateway.multiplex_profiles is true.
_MULTIPLEX_ACTIVE: bool = False


def set_multiplex_active(active: bool) -> None:
    """Mark whether the process is a profile multiplexer (get_secret fails closed)."""
    global _MULTIPLEX_ACTIVE
    _MULTIPLEX_ACTIVE = bool(active)


def is_multiplex_active() -> bool:
    return _MULTIPLEX_ACTIVE


_SECRET_SCOPE: ContextVar[Optional[Mapping[str, str]]] = ContextVar("_SECRET_SCOPE", default=None)


class UnscopedSecretError(RuntimeError):
    """A secret was read in multiplex mode with no scope installed.

    The fix is to wrap the call path in ``set_secret_scope(...)`` (the per-turn
    / per-adapter profile scope), not to widen the global allowlist.
    """


def set_secret_scope(secrets: Optional[Mapping[str, str]]) -> Token:
    """Install the active profile's secret mapping; ``None`` clears. Returns a reset token."""
    return _SECRET_SCOPE.set(secrets)


def reset_secret_scope(token: Token) -> None:
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[Mapping[str, str]]:
    """The active secret mapping, or None when no scope is installed."""
    return _SECRET_SCOPE.get()


# Genuinely-global env vars: process/deployment settings, NOT profile secrets.
# They keep reading os.environ even in multiplex mode (routing them through the
# fail-closed path would wrongly crash). Keep this tight — when in doubt a
# value is a profile secret. Membership is exact name OR prefix.
_GLOBAL_ENV_EXACT = frozenset({
    # Hermes runtime / deployment
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_GATEWAY_LOCK_DIR",
    "HERMES_MAX_ITERATIONS", "HERMES_MAX_TOKENS", "HERMES_API_TIMEOUT",
    "HERMES_REDACT_SECRETS", "HERMES_NOUS_TIMEOUT_SECONDS",
    "_HERMES_GATEWAY",
    # OS / interpreter
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ", "PWD", "SHELL", "TMPDIR",
    "VIRTUAL_ENV", "PYTHONPATH", "SSL_CERT_FILE",
    # Kanban paths (per-board, not per-profile-secret)
    "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_BOARD",
    # API-server LISTENER settings — deployment config (compose/systemd env),
    # which the scoped runner reload must keep seeing or containers silently
    # lose the api_server platform. API_SERVER_KEY is a credential: NOT here.
    # See #64674, #69379.
    "API_SERVER_ENABLED", "API_SERVER_HOST", "API_SERVER_PORT",
    "API_SERVER_CORS_ORIGINS",
    # Relay-connector ROUTING stamps injected by managed deploys. Every reader
    # (gateway.config, relay_url()/registration/self-provision) must resolve
    # the SAME value or the adapter registers while the platform is absent
    # from config. GATEWAY_RELAY_SECRET/_ID/_DELIVERY_KEY and IDP_* are auth
    # material and deliberately stay profile-scoped.
    "GATEWAY_RELAY_URL", "GATEWAY_RELAY_ENDPOINT",
    "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS",
    "GATEWAY_RELAY_PLATFORMS", "GATEWAY_RELAY_BOT_IDS",
    "GATEWAY_RELAY_ROUTE_KEYS", "GATEWAY_RELAY_INSTANCE_ID",
    "GATEWAY_RELAY_WAKE_URL", "GATEWAY_RELAY_DISPLAY_NAME",
})
_GLOBAL_ENV_PREFIXES = (
    "HERMES_KANBAN_",
    "HERMES_TELEGRAM_",   # tuning knobs (batch delays, fallback toggles) — NOT the token
    "TERMINAL_",          # terminal/sandbox backend settings
)


def _is_global_env(name: str) -> bool:
    """True for genuinely process-global (non-profile-secret) env vars."""
    return name in _GLOBAL_ENV_EXACT or name.startswith(_GLOBAL_ENV_PREFIXES)


def _environ_or(name: str, default: Optional[str]) -> Optional[str]:
    val = os.environ.get(name)
    return val if val is not None else default


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a credential by env-var name, honoring the active profile scope.

    Global vars always read ``os.environ``. With a scope installed, a miss returns
    ``default`` under multiplexing (never another profile's ``os.environ`` value)
    but falls through to ``os.environ`` otherwise — single-profile deployments
    inject credentials via the process env (systemd, ``op run``), so the scope
    must stay a ``.env`` overlay, not a blindfold (otherwise cron 401s). With no
    scope: multiplex INACTIVE reads ``os.environ``; ACTIVE raises (fail closed).
    """
    if _is_global_env(name):
        return _environ_or(name, default)
    scope = _SECRET_SCOPE.get()
    if scope is not None:
        val = scope.get(name)
        if val is not None:
            return val
        return default if _MULTIPLEX_ACTIVE else _environ_or(name, default)
    if _MULTIPLEX_ACTIVE:
        raise UnscopedSecretError(
            f"get_secret({name!r}) called with no profile secret scope active "
            f"while multiplexing is on. This credential read must run inside a "
            f"set_secret_scope(...) block (the per-turn / per-adapter profile "
            f"scope). Reading os.environ here would risk leaking another "
            f"profile's value. See docs/design/multiplexing-gateway.md "
            f"(Workstream A)."
        )
    return _environ_or(name, default)


def _strip_inline_comment(value: str) -> str:
    """Strip a dotenv-style inline comment (python-dotenv semantics): quoted values
    scan to the matching close quote (backslash-aware for double quotes) and drop a
    trailing ``# ...``, else stay untouched; unquoted values truncate only at a
    ``#`` PRECEDED BY WHITESPACE (``foo#bar`` survives, ``value # c`` → ``value``)."""
    value = value.strip()
    if not value:
        return value
    quote = value[0]
    if quote in ("'", '"'):
        i = 1
        while i < len(value):
            ch = value[i]
            if quote == '"' and ch == "\\":
                i += 2  # skip the escaped character
                continue
            if ch == quote:
                return value[: i + 1] if value[i + 1:].lstrip().startswith("#") else value
            i += 1
        return value  # unterminated quote: leave as-is
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a ``.env`` file into a dict WITHOUT touching ``os.environ``: ``export``
    prefix, ``#`` comments, and the writer's quote escapes reversed via the canonical
    ``_parse_env_value``. ``utf-8-sig`` so a BOM doesn't prefix the first key."""
    secrets: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return secrets

    from hermes_cli.config import _parse_env_value

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if sep and key:
            secrets[key] = _parse_env_value(_strip_inline_comment(value))
    return secrets


def build_profile_secret_scope(hermes_home: Path) -> Dict[str, str]:
    """Build an isolated, resolved secret mapping for one profile home.

    Named profiles inherit the shared root ``.env`` as a base layer but never
    the gateway process environment. The default profile owns that process
    environment. External sources resolve against this private mapping, so
    their bootstrap credentials cannot cross profile boundaries.
    """
    home = Path(hermes_home)
    is_named_profile = home.parent.name == "profiles"
    secrets: Dict[str, str] = {}

    if not is_named_profile:
        secrets.update(
            (name, value)
            for name, value in os.environ.items()
            if value and not _is_global_env(name) and not name.startswith("OP_")
        )

    if is_named_profile:
        try:
            from hermes_constants import get_default_hermes_root

            root_env = Path(get_default_hermes_root(home)) / ".env"
            if root_env.resolve() != (home / ".env").resolve():
                secrets.update(load_env_file(root_env))
        except Exception:
            pass
    secrets.update(load_env_file(home / ".env"))

    # ``.op.env`` carries 1Password bootstrap auth. It is local to this home
    # and lower precedence than a value explicitly written to ``.env``.
    for name, value in load_env_file(home / ".op.env").items():
        secrets.setdefault(name, value)

    registry_resolved: set[str] = set()
    try:
        from hermes_cli.env_loader import _load_secrets_config
        from agent.secret_sources.registry import apply_all

        sources_cfg = _load_secrets_config(home)
        report = apply_all(
            sources_cfg, home, environ=secrets, scoped=is_named_profile
        )
        registry_resolved.update(report.provenance)
    except Exception:
        sources_cfg = {}

    # Resolve direct ``op://`` values and the configured 1Password mapping in
    # one pass. Configured mappings win when override_existing is enabled.
    raw_refs = {
        name: value
        for name, value in secrets.items()
        if isinstance(value, str) and value.strip().startswith("op://")
    }
    op_cfg = sources_cfg.get("onepassword")
    op_cfg = op_cfg if isinstance(op_cfg, dict) else {}
    configured_refs = {}
    if op_cfg.get("enabled") and isinstance(op_cfg.get("env"), dict):
        configured_refs = {
            str(name): str(value)
            for name, value in op_cfg["env"].items()
            if isinstance(value, str) and value.strip().startswith("op://")
        }
    override_existing = bool(op_cfg.get("override_existing", True))
    op_refs = (
        {**raw_refs, **configured_refs}
        if override_existing
        else {**configured_refs, **raw_refs}
    )
    if op_refs:
        resolved: Dict[str, str] = {}
        try:
            from agent.secret_sources.base import (
                reset_source_environment,
                set_source_environment,
            )
            from agent.secret_sources.onepassword import fetch_onepassword_secrets

            token = set_source_environment(secrets)
            try:
                token_env = str(
                    op_cfg.get("service_account_token_env")
                    or "OP_SERVICE_ACCOUNT_TOKEN"
                )
                local_token = str(secrets.get(token_env) or "").strip()
                include_process_auth = not is_named_profile
                resolved, _warnings = fetch_onepassword_secrets(
                    references=op_refs,
                    account=str(op_cfg.get("account") or ""),
                    token_env=token_env,
                    token_value=(
                        local_token
                        if local_token
                        else (None if include_process_auth else "")
                    ),
                    include_process_auth=include_process_auth,
                    auth_env={
                        name: value
                        for name, value in secrets.items()
                        if name in {"OP_ACCOUNT", "OP_CONNECT_HOST", "OP_CONNECT_TOKEN"}
                        or name.startswith("OP_SESSION_")
                    },
                    binary_path=str(op_cfg.get("binary_path") or ""),
                    cache_ttl_seconds=float(op_cfg.get("cache_ttl_seconds", 300)),
                    home_path=home,
                )
            finally:
                reset_source_environment(token)
        except Exception:
            resolved = {}
        secrets.update((name, value) for name, value in resolved.items() if value)
        fail_closed = set(raw_refs)
        if override_existing:
            fail_closed.update(configured_refs)
        for name in fail_closed:
            if name not in resolved and name not in registry_resolved:
                secrets.pop(name, None)

    # Preserve the upstream per-home source snapshot contract. A gateway may
    # hydrate sources off-thread before building this scope.
    try:
        from hermes_cli.env_loader import get_secret_source_values

        secrets.update(get_secret_source_values(home))
    except Exception:
        pass

    for key in list(secrets):
        if _is_global_env(key):
            secrets.pop(key, None)
    return secrets
