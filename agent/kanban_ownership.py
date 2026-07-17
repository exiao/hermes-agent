"""Thread-local Kanban ownership masking for delegated child agents.

This module deliberately has no Hermes imports.  A delegated child shares its
worker parent's process environment, but it must not inherit authority to
complete, block, time out, or otherwise mutate the parent's Kanban task.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_delegate_child_masks_kanban_ownership: ContextVar[bool] = ContextVar(
    "_delegate_child_masks_kanban_ownership", default=False
)


@contextmanager
def delegated_child_kanban_env() -> Iterator[None]:
    """Mask a parent worker's Kanban ownership for a delegated child."""
    token = _delegate_child_masks_kanban_ownership.set(True)
    try:
        yield
    finally:
        _delegate_child_masks_kanban_ownership.reset(token)


def delegated_child_masks_kanban_ownership() -> bool:
    """Return whether the current context is a delegated child agent."""
    return _delegate_child_masks_kanban_ownership.get()
