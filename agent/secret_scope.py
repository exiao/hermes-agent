"""Profile-scoped credential resolution for multi-profile gateway multiplexing.

The multiplexing gateway serves many profiles from one process. Each profile
has its own ``.env`` with its own provider keys and platform tokens, so we
**cannot** union them into the process-global ``os.environ`` (that would leak
profile A's keys to profile B's turns, and to every subprocess spawned with
``env=dict(os.environ)``).

This module provides a fail-closed, context-local secret scope:

- ``set_secret_scope(mapping)`` installs the active profile's secrets for the
  current task (a contextvar, so it propagates into the agent's worker thread
  via ``copy_context()`` exactly like the HERMES_HOME override).
- ``get_secret(name)`` reads from that scope. When multiplexing is **active**
  and no scope is set, it RAISES rather than silently falling back to
  ``os.environ`` — an un-migrated or newly-added call site fails loud at that
  exact line instead of leaking another profile's value. When multiplexing is
  **off** (the default), it transparently reads ``os.environ`` so the
  single-profile gateway and every non-gateway caller behave exactly as before.

Design rationale lives in ``docs/design/multiplexing-gateway.md`` (Workstream A).
"""
from __future__ import annotations

import os
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Dict, Mapping, Optional


# ── multiplex-active flag ────────────────────────────────────────────────
# Process-global: set once at gateway startup when gateway.multiplex_profiles
# is true. Governs whether get_secret() fails closed on an unscoped read.
# A plain module global (not a contextvar): it describes the deployment mode,
# not a per-task value.
_MULTIPLEX_ACTIVE: bool = False


def set_multiplex_active(active: bool) -> None:
    """Mark whether the process is running as a profile multiplexer.

    Called once at gateway startup. When True, ``get_secret`` fails closed on
    an unscoped read instead of falling back to ``os.environ``.
    """
    global _MULTIPLEX_ACTIVE
    _MULTIPLEX_ACTIVE = bool(active)


def is_multiplex_active() -> bool:
    """Return whether the process is running as a profile multiplexer."""
    return _MULTIPLEX_ACTIVE


# ── the secret scope contextvar ──────────────────────────────────────────
_SECRET_SCOPE: ContextVar[Optional[Mapping[str, str]]] = ContextVar(
    "_SECRET_SCOPE", default=None
)


class UnscopedSecretError(RuntimeError):
    """Raised when a secret is read in multiplex mode with no scope installed.

    This is the fail-closed signal: it means a credential read reached
    ``get_secret`` without a profile scope active, which in a multiplexer would
    otherwise leak whichever profile's value happened to be in ``os.environ``.
    The fix is to wrap the call path in ``set_secret_scope(...)`` (the per-turn
    / per-adapter profile scope), not to widen the allowlist.
    """


def set_secret_scope(secrets: Optional[Mapping[str, str]]) -> Token:
    """Install the active profile's secret mapping for the current context.

    Returns a token for ``reset_secret_scope``. Pass ``None`` to clear.
    """
    return _SECRET_SCOPE.set(secrets)


def reset_secret_scope(token: Token) -> None:
    """Restore the previous secret scope."""
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[Mapping[str, str]]:
    """Return the active secret mapping, or None when no scope is installed."""
    return _SECRET_SCOPE.get()


# ── genuinely-global env vars (NOT per-profile secrets) ──────────────────
# These are process/deployment-level settings, not profile credentials. They
# legitimately live in os.environ and must keep reading from it even in
# multiplex mode — routing them through the fail-closed path would wrongly
# crash. Anything matching is read from os.environ regardless of scope.
#
# Membership test is by exact name OR prefix (see _is_global_env). Keep this
# list tight: when in doubt a value is a profile secret, not a global.
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
})
_GLOBAL_ENV_PREFIXES = (
    "HERMES_KANBAN_",
    "HERMES_TELEGRAM_",   # tuning knobs (batch delays, fallback toggles) — NOT the token
    "TERMINAL_",          # terminal/sandbox backend settings
)


def _is_global_env(name: str) -> bool:
    """Return True for genuinely process-global (non-profile-secret) env vars."""
    if name in _GLOBAL_ENV_EXACT:
        return True
    return any(name.startswith(p) for p in _GLOBAL_ENV_PREFIXES)


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a credential by env-var name, honoring the active profile scope.

    Resolution order:

    1. Genuinely-global vars (``_is_global_env``) always read ``os.environ`` —
       they are deployment settings, not profile secrets.
    2. When a secret scope is installed (multiplexed turn), read from it; an
       absent key returns ``default``. The scope is authoritative — we do NOT
       fall through to ``os.environ``, because in a multiplexer ``os.environ``
       may hold another profile's value.
    3. No scope installed:
       - multiplex INACTIVE (default deployment): read ``os.environ`` —
         identical to the legacy ``os.getenv`` behavior every caller had before.
       - multiplex ACTIVE: FAIL CLOSED. Raise ``UnscopedSecretError`` so the
         missing scope is caught loudly instead of leaking a cross-profile value.
    """
    if _is_global_env(name):
        val = os.environ.get(name)
        return val if val is not None else default

    scope = _SECRET_SCOPE.get()
    if scope is not None:
        val = scope.get(name)
        return val if val is not None else default

    if _MULTIPLEX_ACTIVE:
        raise UnscopedSecretError(
            f"get_secret({name!r}) called with no profile secret scope active "
            f"while multiplexing is on. This credential read must run inside a "
            f"set_secret_scope(...) block (the per-turn / per-adapter profile "
            f"scope). Reading os.environ here would risk leaking another "
            f"profile's value. See docs/design/multiplexing-gateway.md "
            f"(Workstream A)."
        )

    val = os.environ.get(name)
    return val if val is not None else default


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a ``.env`` file into a plain dict WITHOUT touching ``os.environ``.

    Used to load a profile's secrets into an isolated mapping for
    ``set_secret_scope``. Mirrors python-dotenv's basic parsing (KEY=VALUE,
    ``export`` prefix, ``#`` comments, optional matching quotes) but never
    mutates the process environment — that isolation is the whole point.
    """
    secrets: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return secrets

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        secrets[key] = value

    return secrets


def build_profile_secret_scope(hermes_home: Path) -> Dict[str, str]:
    """Build a resolved profile-secret mapping from ``<home>/.env``.

    Direct ``op://`` references are resolved for this profile before the mapping
    is installed. Passing the raw reference through ``get_secret`` would send the
    reference string itself as an API key; falling back to process ``os.environ``
    would be worse because that value may belong to another profile. Failed
    references are omitted so callers fail closed rather than authenticating with
    either unsafe value.

    Returns a fresh dict (safe to install via ``set_secret_scope``). Genuinely
    global vars are intentionally NOT copied in — ``get_secret`` reads those
    from ``os.environ`` directly, so the scope holds only profile secrets.
    """
    home = Path(hermes_home)
    secrets = load_env_file(home / ".env")

    # Secret sources normally populate os.environ. Resolve them into this
    # isolated mapping instead so a multiplexed scope can remain authoritative.
    try:
        from hermes_cli.env_loader import _load_secrets_config
        from agent.secret_sources.registry import apply_all

        sources_cfg = _load_secrets_config(home)
        source_values = dict(secrets)
        apply_all(sources_cfg, home, environ=source_values)
        secrets = source_values
    except Exception:
        pass

    raw_op_refs = {
        name: value
        for name, value in secrets.items()
        if isinstance(value, str) and value.strip().startswith("op://")
    }

    op_cfg: Dict[str, object] = {}
    try:
        from hermes_cli.env_loader import _load_secrets_config

        sources_cfg = _load_secrets_config(home)
        candidate = sources_cfg.get("onepassword")
        if isinstance(candidate, dict):
            op_cfg = candidate
    except Exception:
        op_cfg = {}

    configured_refs: Dict[str, str] = {}
    if op_cfg.get("enabled"):
        configured = op_cfg.get("env")
        if isinstance(configured, dict):
            configured_refs = {
                str(name): str(value)
                for name, value in configured.items()
                if isinstance(name, str)
                and isinstance(value, str)
                and value.strip().startswith("op://")
            }
    override_existing = bool(op_cfg.get("override_existing", True))
    op_refs = (
        {**raw_op_refs, **configured_refs}
        if override_existing
        else {**configured_refs, **raw_op_refs}
    )

    if op_refs:
        resolved: Dict[str, str] = {}
        try:
            from agent.secret_sources.onepassword import fetch_onepassword_secrets

            token_env = str(
                op_cfg.get("service_account_token_env")
                or "OP_SERVICE_ACCOUNT_TOKEN"
            )
            bootstrap = load_env_file(home / ".op.env")
            profile_auth_values = {**bootstrap, **secrets}
            auth_env = {
                name: value
                for name, value in profile_auth_values.items()
                if name in {"OP_ACCOUNT", "OP_CONNECT_HOST", "OP_CONNECT_TOKEN"}
                or name.startswith("OP_SESSION_")
            }
            local_token = str(
                secrets.get(token_env) or bootstrap.get(token_env) or ""
            ).strip()
            include_process_auth = home.parent.name != "profiles"
            # A default/profile-owner scope may keep the legacy shell/desktop
            # auth fallback. Named profiles pass an explicit empty token and
            # disable process auth so they can never borrow the gateway's
            # 1Password identity.
            token_value: Optional[str] = (
                local_token if local_token else (None if include_process_auth else "")
            )
            try:
                cache_ttl = float(str(op_cfg.get("cache_ttl_seconds", 300)))
            except (TypeError, ValueError):
                cache_ttl = 300.0
            resolved, _warnings = fetch_onepassword_secrets(
                references=op_refs,
                account=str(op_cfg.get("account") or ""),
                token_env=token_env,
                token_value=token_value,
                # Named profiles must not inherit the gateway process's
                # OP_SESSION_*/OP_CONNECT identity. The default profile still
                # keeps desktop-session behavior.
                include_process_auth=include_process_auth,
                auth_env=auth_env,
                binary_path=str(op_cfg.get("binary_path") or ""),
                cache_ttl_seconds=cache_ttl,
                home_path=home,
            )
        except Exception:
            # A missing/unauthenticated source must not turn a reference string
            # into a credential or leak the process/default profile's value.
            resolved = {}
        for name, value in resolved.items():
            rendered = str(value or "")
            if rendered.strip() and (
                override_existing or name not in secrets or name in raw_op_refs
            ):
                secrets[name] = rendered
        fail_closed_names = set(raw_op_refs)
        if override_existing:
            fail_closed_names.update(configured_refs)
        for name in fail_closed_names:
            if name not in resolved:
                secrets.pop(name, None)
    return secrets

