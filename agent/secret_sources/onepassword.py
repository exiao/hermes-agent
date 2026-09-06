"""1Password (`op` CLI) secret source.

Users map env-var names to ``op://vault/item/field`` references in
``secrets.onepassword.env``; each is resolved with one ``op read -- <ref>``
call using whatever auth the user's ``op`` already has (``OP_SERVICE_ACCOUNT_TOKEN``
headless, ``OP_SESSION_*`` interactive) — Hermes never authenticates on the
user's behalf, and failures never block startup. Complete pulls are cached
in-process and under ``<hermes_home>/cache/op_cache.json`` (values only; auth
material is fingerprinted, never stored).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # noqa: F401 — tests monkeypatch ``op.subprocess.run``
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.secret_sources._cache import CachedFetch, SecretCache, fingerprint as _fingerprint
from agent.secret_sources.base import (
    ErrorKind, FetchResult, SecretSource, classify_cli_error, coerce_float,
    get_source_environment, is_valid_env_name,
)

logger = logging.getLogger(__name__)

_OP_RUN_TIMEOUT = 30

# `op` itself reads OP_SERVICE_ACCOUNT_TOKEN; `service_account_token_env` lets
# the user source it from another name, and _op_child_env normalizes it back.
_DEFAULT_TOKEN_ENV = "OP_SERVICE_ACCOUNT_TOKEN"

# Minimal allowlisted child env (never the full post-dotenv os.environ, which
# holds every provider credential). OP_SESSION_* and the token are added
# dynamically in _op_child_env().
_OP_ENV_ALLOWLIST = (
    "PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
    "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR",
    "OP_ACCOUNT", "OP_CONNECT_HOST", "OP_CONNECT_TOKEN",
    # Lets a user skip op's desktop-app integration probe (which can hang with
    # no timeout on a wedged desktop container) and go straight to token auth.
    "OP_LOAD_DESKTOP_APP_SETTINGS",
)

# L1 key folds in str(home_path) so a HERMES_HOME switch inside one long-lived
# process (the gateway) can't return another profile's secrets. The disk key
# omits home because the file already lives under <home>/cache/.
_CacheKey = Tuple[str, str, str, str]  # (auth_fp, account, home, refs_fp)
_DISK_CACHE_BASENAME = "op_cache.json"


def _disk_key_str(cache_key: _CacheKey) -> str:
    auth_fp, account, _home, refs_fp = cache_key
    return f"{auth_fp}|{account}|{refs_fp}"


_STORE: SecretCache[_CacheKey] = SecretCache(_DISK_CACHE_BASENAME, key_serializer=_disk_key_str)
_CACHE = _STORE.memory  # tests flush L1 directly

_MISSING_BINARY_HINT = (
    "Install the 1Password CLI (https://developer.1password.com/docs/cli/get-started/) "
    "or set secrets.onepassword.binary_path."
)

# First matching rule wins.
_OP_ERROR_RULES = (
    (ErrorKind.TIMEOUT, ("timed out",)),
    (ErrorKind.BINARY_MISSING, ("not found on path", "not an executable", "failed to invoke")),
    (ErrorKind.AUTH_FAILED, ("unauthorized", "not signed in", "session expired",
                             "authentication", "401", "403")),
    (ErrorKind.EMPTY_VALUE, ("empty value",)),
    (ErrorKind.NETWORK, ("network", "connection", "resolve host", "dns")),
)


def _classify_op_error(message: str) -> ErrorKind:
    return classify_cli_error(message, _OP_ERROR_RULES)


def _validate_references(references: Optional[Dict[str, str]]) -> Tuple[Dict[str, str], List[str]]:
    """``(valid_refs, warnings)``: keep valid env names bound to stripped ``op://`` strings."""
    valid: Dict[str, str] = {}
    warnings: List[str] = []
    for name, ref in (references or {}).items():
        if not is_valid_env_name(name):
            warnings.append(f"Skipping {name!r}: not a valid env-var name")
        elif not isinstance(ref, str):
            warnings.append(f"Skipping {name!r}: reference is not a string")
        elif not ref.strip().startswith("op://"):
            warnings.append(f"Skipping {name!r}: {ref!r} is not an op:// secret reference")
        else:
            valid[name] = ref.strip()
    return valid, warnings


def _auth_fingerprint(
    token_env: str,
    *,
    token_value: Optional[str] = None,
    include_process_auth: bool = True,
    auth_env: Optional[Dict[str, str]] = None,
) -> str:
    """SHA-256 prefix over the auth material `op` would use.

    Folds in the service-account token, ``OP_ACCOUNT``, the 1Password Connect
    ``OP_CONNECT_HOST``/``OP_CONNECT_TOKEN``, and *all* ``OP_SESSION_*`` vars
    (the names `op` actually exports for interactive sessions —
    ``OP_SESSION_<account_shorthand>``).  Signing out and into a different
    identity therefore changes the cache key, so a value cached under a
    previous identity is never served under a new one.  Never logged or
    displayed; the raw token never leaves this hash.
    """
    source_env = get_source_environment()
    resolved_token = (
        source_env.get(token_env, "") if token_value is None else token_value
    )
    parts: List[str] = [
        f"token={resolved_token}",
        f"auth_mode={'process' if include_process_auth else 'isolated'}",
    ]
    if include_process_auth:
        parts.append(f"account={source_env.get('OP_ACCOUNT', '')}")
        parts.append(f"connect_host={source_env.get('OP_CONNECT_HOST', '')}")
        parts.append(f"connect_token={source_env.get('OP_CONNECT_TOKEN', '')}")
        # The auth PATH vars select WHICH on-disk 1Password config/session `op`
        # reads. Two process owners differing only by HOME must not collide on
        # one cache key, or a value cached under one identity is served under
        # the other.
        for key in (
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "XDG_CONFIG_HOME",
            "XDG_RUNTIME_DIR",
        ):
            parts.append(f"{key}={source_env.get(key, '')}")
        for key in sorted(source_env):
            if key.startswith("OP_SESSION_"):
                parts.append(f"{key}={source_env[key]}")
    if auth_env:
        for key in sorted(auth_env):
            parts.append(f"profile:{key}={auth_env[key]}")
    material = "\n".join(parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _refs_fingerprint(references: Dict[str, str]) -> str:
    return _fingerprint("\n".join(f"{name}={references[name]}" for name in sorted(references)))


def find_op(binary_path: str = "") -> Optional[Path]:
    """Resolve a usable ``op`` binary, or None. A pinned ``binary_path`` is used
    verbatim — pinned-but-missing returns None rather than falling back to PATH."""
    found = binary_path or shutil.which("op")
    if not found or (binary_path and not os.access(binary_path, os.X_OK)):
        return None
    return Path(found)


def _scrub(text: str) -> str:
    """Full ECMA-48 ANSI strip (so a control sequence can't hide text after a redaction marker) + trim."""
    from tools.ansi_strip import strip_ansi

    return strip_ansi(text).replace("\x1b", "").strip()


def _op_child_env(
    token_value: str,
    *,
    include_process_auth: bool = True,
    auth_env: Optional[Dict[str, str]] = None,
    isolated_home: Optional[Path] = None,
) -> Dict[str, str]:
    """Build a minimal allowlisted environment for the ``op`` child process."""
    source_env = get_source_environment()
    env: Dict[str, str] = {}
    auth_path_vars = {
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
    }
    for key in _OP_ENV_ALLOWLIST:
        # A named multiplexed profile (include_process_auth=False) must not
        # inherit the process owner's 1Password identity: neither OP_* vars nor
        # the auth PATH vars that point `op` at a desktop/session config dir.
        if not include_process_auth and (
            key.startswith("OP_") or key in auth_path_vars
        ):
            continue
        val = source_env.get(key)
        if val is not None:
            env[key] = val
    # Desktop / interactive session credentials belong to the process profile.
    # A named multiplexed profile must not inherit them.
    if include_process_auth:
        for key, val in source_env.items():
            if key.startswith("OP_SESSION_"):
                env[key] = val
    if isolated_home is not None:
        isolated = Path(isolated_home)
        env["HOME"] = str(isolated)
        env["USERPROFILE"] = str(isolated)
        env["XDG_CONFIG_HOME"] = str(isolated / ".config")
        env["APPDATA"] = str(isolated / "AppData" / "Roaming")
        env["LOCALAPPDATA"] = str(isolated / "AppData" / "Local")
    if auth_env:
        for key, val in auth_env.items():
            if key in _OP_ENV_ALLOWLIST or key.startswith("OP_SESSION_"):
                env[key] = val
    # `op` reads OP_SERVICE_ACCOUNT_TOKEN regardless of which env var the user
    # configured Hermes to source it from, so normalize to that name here.
    if token_value:
        env["OP_SERVICE_ACCOUNT_TOKEN"] = token_value
    env["NO_COLOR"] = "1"
    return env


def _run_op_read(
    op: Path,
    reference: str,
    *,
    account: str = "",
    token_value: str = "",
    include_process_auth: bool = True,
    auth_env: Optional[Dict[str, str]] = None,
    isolated_home: Optional[Path] = None,
) -> str:
    """Resolve a single ``op://`` reference to its value.

    Raises :class:`RuntimeError` on any failure — including a ``returncode 0``
    with empty output, which would otherwise silently clobber a good
    ``.env``/shell credential with ``""``.
    """
    cmd: List[str] = [str(op), "read"]
    if account:
        cmd += ["--account", account]
    # `--` terminates option parsing so a reference can never be mis-parsed as
    # an `op` flag even if validation is ever loosened.
    cmd += ["--", reference]

    try:
        proc = subprocess.run(  # noqa: S603 — op path is user-trusted, argv list
            cmd,
            env=_op_child_env(
                token_value,
                include_process_auth=include_process_auth,
                auth_env=auth_env,
                isolated_home=isolated_home,
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_OP_RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"op read timed out after {_OP_RUN_TIMEOUT}s for {reference!r}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke op: {exc}") from exc

    if proc.returncode != 0:
        err = _scrub(proc.stderr or "")[:200]
        if err:
            raise RuntimeError(f"op read failed for {reference!r}: {err}")
        raise RuntimeError(
            f"op read exited {proc.returncode} for {reference!r}"
        )

    # `op` appends a trailing newline; strip only that so a value with
    # intentional internal/edge spaces survives.  But a value that is empty or
    # whitespace-only is treated as empty: applying it would silently clobber a
    # good .env/shell credential with effectively nothing.
    value = (proc.stdout or "").rstrip("\r\n")
    if not value.strip():
        raise RuntimeError(f"op read returned an empty value for {reference!r}")
    return value


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_onepassword_secrets(
    *, references: Dict[str, str], account: str = "", token_env: str = _DEFAULT_TOKEN_ENV,
    token_value: Optional[str] = None, include_process_auth: bool = True,
    auth_env: Optional[Dict[str, str]] = None,
    binary: Optional[Path] = None, binary_path: str = "", use_cache: bool = True,
    cache_ttl_seconds: float = 300, home_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Resolve ``references`` (name → ``op://…``) to ``(secrets, warnings)``.

    Raises ``RuntimeError`` only when no ``op`` binary is available; per-ref
    failures become warnings. Only a complete, error-free pull is cached, so a
    transient auth failure isn't frozen in for the whole TTL window.
    """
    valid, warnings = _validate_references(references)
    if not valid:
        return {}, warnings

    resolved_token = (
        get_source_environment().get(token_env, "")
        if token_value is None else token_value
    ).strip()
    cache_key: _CacheKey = (_auth_fingerprint(
        token_env,
        token_value=resolved_token,
        include_process_auth=include_process_auth,
        auth_env=auth_env,
    ), account or "",
                            str(home_path) if home_path is not None else "", _refs_fingerprint(valid))

    if use_cache:
        cached = _STORE.lookup(cache_key, cache_ttl_seconds, home_path)
        if cached is not None:
            return dict(cached.secrets), warnings

    op = binary or find_op(binary_path)
    if op is None:
        raise RuntimeError("op CLI not found.  Install the 1Password CLI "
                           "(https://developer.1password.com/docs/cli/get-started/) or set "
                           "secrets.onepassword.binary_path to its absolute location.")

    secrets: Dict[str, str] = {}
    read_errors = 0
    for name in sorted(valid):
        try:
            secrets[name] = _run_op_read(
                op,
                valid[name],
                account=account,
                token_value=resolved_token,
                include_process_auth=include_process_auth,
                auth_env=auth_env,
                isolated_home=home_path if not include_process_auth else None,
            )
        except RuntimeError as exc:
            warnings.append(str(exc))
            read_errors += 1

    if use_cache and not read_errors and secrets:
        _STORE.store(cache_key, CachedFetch(secrets=dict(secrets), fetched_at=time.time()),
                     cache_ttl_seconds, home_path)

    return secrets, warnings


def _missing_binary_error(binary_path: str) -> str:
    if binary_path:
        return f"secrets.onepassword.binary_path ({binary_path!r}) is not an executable op binary."
    return ("secrets.onepassword.enabled is true but the op CLI was not found on PATH.  Install it "
            "(https://developer.1password.com/docs/cli/get-started/) or set secrets.onepassword.binary_path.")


def apply_onepassword_secrets(
    *, enabled: bool, env: Optional[Dict[str, str]] = None, account: str = "",
    service_account_token_env: str = _DEFAULT_TOKEN_ENV, binary_path: str = "",
    override_existing: bool = True, cache_ttl_seconds: float = 300, home_path: Optional[Path] = None,
) -> FetchResult:
    """Resolve configured ``op://`` references and set them on ``os.environ``
    (``hermes secrets onepassword sync --apply``). Never raises. Refs already
    satisfied by the env (when ``override_existing`` is false) and the token var
    are skipped *before* fetching, so ``op`` never runs for a discarded value."""
    result = FetchResult()
    if not enabled:
        return result

    valid, warnings = _validate_references(env)
    result.warnings.extend(warnings)

    def _guarded(name: str) -> bool:
        """True when ``name`` must not be applied (token var or env already set)."""
        return name == service_account_token_env or (not override_existing and bool(os.environ.get(name)))

    result.skipped.extend(n for n in valid if _guarded(n))
    refs_to_fetch = {n: ref for n, ref in valid.items() if not _guarded(n)}
    if not refs_to_fetch:
        return result

    binary = find_op(binary_path)
    result.binary_path = binary
    if binary is None:
        result.error = _missing_binary_error(binary_path)
        return result

    try:
        secrets, fetch_warnings = fetch_onepassword_secrets(
            references=refs_to_fetch, account=account, token_env=service_account_token_env,
            binary=binary, cache_ttl_seconds=cache_ttl_seconds, home_path=home_path)
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(fetch_warnings)
    for name, value in secrets.items():
        if _guarded(name):  # defensive re-check: keys should already be ⊆ refs_to_fetch
            if name not in result.skipped:
                result.skipped.append(name)
            continue
        os.environ[name] = value
        result.applied.append(name)
    return result


class OnePasswordSource(SecretSource):
    """1Password as a registered **mapped** source (explicit per-var bindings, so
    its claims outrank bulk sources on contested vars)."""

    name = "onepassword"
    label = "1Password"
    shape = "mapped"
    scheme = "op"
    token_env_key = "service_account_token_env"
    default_token_env = _DEFAULT_TOKEN_ENV
    # override_existing defaults True: an explicit VAR→op:// binding is the
    # strongest user intent; a stale .env line must not silently defeat it.
    override_existing_default = True
    _AUTH_HINT = ("Run `hermes secrets onepassword token` to paste a fresh service-account token "
                  "({token_env}), or `op signin` for an interactive session.")
    remediation_hints = {ErrorKind.AUTH_FAILED: _AUTH_HINT, ErrorKind.AUTH_EXPIRED: _AUTH_HINT,
                         ErrorKind.BINARY_MISSING: _MISSING_BINARY_HINT}

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "env": {"description": "Map of ENV_VAR -> op://vault/item/field reference", "default": {}},
            "account": {"description": "op --account shorthand (empty = default account)", "default": ""},
            "service_account_token_env": {"description": "Env var holding the service-account token "
                                                         "(unset = desktop/interactive session)",
                                          "default": _DEFAULT_TOKEN_ENV},
            "binary_path": {"description": "Pin the op binary (empty = resolve via PATH)", "default": ""},
            "cache_ttl_seconds": {"description": "Disk+memory cache TTL; 0 disables", "default": 300},
            "override_existing": {"description": "Resolved values overwrite .env/shell values", "default": True},
        }

    def fetch(
        self,
        cfg: dict,
        home_path: Path,
        environ: Optional[Dict[str, str]] = None,
    ) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        env_map = cfg.get("env")
        valid, warnings = _validate_references(env_map if isinstance(env_map, dict) else None)
        result.warnings.extend(warnings)
        if not valid:
            if not warnings:
                result.fail("secrets.onepassword.enabled is true but the env: map is "
                            "empty.  Add ENV_VAR: op://vault/item/field entries.", ErrorKind.NOT_CONFIGURED)
            return result

        binary_path = str(cfg.get("binary_path") or "")
        binary = find_op(binary_path)
        result.binary_path = binary
        if binary is None:
            return result.fail(_missing_binary_error(binary_path), ErrorKind.BINARY_MISSING)

        try:
            scoped_auth = None
            if environ is not None:
                scoped_auth = {
                    key: value for key, value in environ.items()
                    if key in {"OP_ACCOUNT", "OP_CONNECT_HOST", "OP_CONNECT_TOKEN"}
                    or key.startswith("OP_SESSION_")
                }
            secrets, fetch_warnings = fetch_onepassword_secrets(
                references=valid, account=str(cfg.get("account") or ""), token_env=self.token_env(cfg),
                token_value=(environ or {}).get(self.token_env(cfg)) if environ is not None else None,
                include_process_auth=environ is None,
                auth_env=scoped_auth,
                binary=binary, cache_ttl_seconds=coerce_float(cfg.get("cache_ttl_seconds", 300), 300.0),
                home_path=home_path)
        except RuntimeError as exc:
            return result.fail(str(exc), _classify_op_error(str(exc)))

        result.secrets = secrets
        result.warnings.extend(fetch_warnings)
        return result


def clear_caches(home_path: Optional[Path] = None) -> None:
    """Drop in-process AND disk caches (after a token rotation, so the next
    startup resolves fresh instead of serving values cached under the old token)."""
    _STORE.clear(home_path)


_reset_cache_for_tests = clear_caches


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import hashlib  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'DiskCache': ('agent.secret_sources._cache', 'DiskCache'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
