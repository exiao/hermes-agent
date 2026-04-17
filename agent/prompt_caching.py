"""Anthropic prompt caching (system_and_3 strategy).

Reduces input token costs by ~75% on multi-turn conversations by caching
the conversation prefix. Uses 4 cache_control breakpoints (Anthropic max):
  1. System prompt (stable across all turns)
  2-4. Last 3 non-system messages (rolling window)

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List


# Default cache TTL for Anthropic prompt caching.
#
# Anthropic offers two TTLs, both generally available (no beta header needed
# since 2025-04-11 — see SDK CHANGELOG commit 35201ba):
#   - "5m" : 1.25x base input price on cache writes, 0.1x on reads
#   - "1h" : 2.00x base input price on cache writes, 0.1x on reads
#
# 1h is the right default for Hermes because:
#   1. Parent/child (delegate_task) flows reuse the same prefix across calls
#      that often exceed 5 minutes once subagents do real work.
#   2. Long-running interactive sessions rehit the system prompt + tool defs
#      for hours.
#   3. Break-even vs 5m is >1 cache hit/hour. We consistently exceed that.
#
# Tune here if pricing or usage patterns change. This is the single source of
# truth; every caching call-site reads this constant.
CACHE_TTL: str = "1h"


def make_cache_marker(cache_ttl: str = CACHE_TTL) -> Dict[str, Any]:
    """Return an Anthropic ``cache_control`` marker honoring the TTL.

    Use this at every inline cache_control injection site so a single
    ``CACHE_TTL`` change propagates everywhere. The marker structure is:

        {"type": "ephemeral"}                          # implicit 5m
        {"type": "ephemeral", "ttl": "1h"}             # explicit 1h
    """
    marker: Dict[str, Any] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = CACHE_TTL,
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Apply system_and_3 caching strategy to messages for Anthropic models.

    Places up to 4 cache_control breakpoints: system prompt + last 3 non-system messages.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = make_cache_marker(cache_ttl)

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
