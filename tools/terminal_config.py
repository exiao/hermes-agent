"""Context-local terminal configuration for concurrent profile turns.

Terminal settings historically flow through ``TERMINAL_*`` environment variables.
That remains the process and child-process contract for CLI and single-profile
surfaces. Multiplexed gateway turns, however, overlap across awaits and must not
replace those process-global values. This module provides the narrow contextvar
overlay used by terminal consumers during a profile runtime scope.
"""
from __future__ import annotations

import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class TerminalConfigScope:
    values: Mapping[str, str]
    environment_key: str | None


_TERMINAL_CONFIG_SCOPE: ContextVar[TerminalConfigScope | None] = ContextVar(
    "_TERMINAL_CONFIG_SCOPE", default=None
)


def set_terminal_config_scope(
    config: Mapping[str, str], *, environment_key: str | None = None
) -> Token:
    """Install terminal environment values for the active task context."""
    return _TERMINAL_CONFIG_SCOPE.set(
        TerminalConfigScope(values=dict(config), environment_key=environment_key)
    )


def reset_terminal_config_scope(token: Token) -> None:
    """Restore the terminal configuration that was active before this scope."""
    _TERMINAL_CONFIG_SCOPE.reset(token)


def has_terminal_config_scope() -> bool:
    """Return whether the current task has a profile-local terminal overlay."""
    return _TERMINAL_CONFIG_SCOPE.get() is not None


def get_terminal_environment_key() -> str | None:
    """Return the profile-specific environment-cache partition, when scoped."""
    scope = _TERMINAL_CONFIG_SCOPE.get()
    return scope.environment_key if scope is not None else None


def get_terminal_env(name: str, default: str | None = None) -> str | None:
    """Read a terminal setting from the current scope, then process environment."""
    scope = _TERMINAL_CONFIG_SCOPE.get()
    if scope is not None:
        return scope.values.get(name, default)
    return os.getenv(name, default)
