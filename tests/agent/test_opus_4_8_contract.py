"""Tests for Claude Opus 4.8 adapter contract parity with 4.7.

Opus 4.8 inherits the 4.7 server-managed contract: it forbids client-supplied
sampling params (temperature/top_p/top_k -> HTTP 400), supports adaptive
thinking, and accepts the ``xhigh`` effort level. The substring lists in
anthropic_adapter must cover 4.8 the same way they cover 4.7, otherwise every
4.8 request 400s with ``temperature is deprecated for this model``.
"""

from __future__ import annotations

import pytest

from agent.anthropic_adapter import (
    _forbids_sampling_params,
    _supports_adaptive_thinking,
    _supports_xhigh_effort,
    _get_anthropic_max_output,
)


# Models that must follow the 4.7+ no-sampling-params contract.
_NO_SAMPLING = [
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-4-8-20250715",
    "anthropic/claude-opus-4-8",
    "anthropic.claude-opus-4.8",
]

# Models that still accept sampling params.
_ALLOWS_SAMPLING = [
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-3-opus",
]


@pytest.mark.parametrize("model", _NO_SAMPLING)
def test_forbids_sampling_params_for_4_7_plus(model: str) -> None:
    assert _forbids_sampling_params(model) is True


@pytest.mark.parametrize("model", _ALLOWS_SAMPLING)
def test_allows_sampling_params_for_pre_4_7(model: str) -> None:
    assert _forbids_sampling_params(model) is False


@pytest.mark.parametrize("model", ["claude-opus-4-7", "claude-opus-4-8"])
def test_xhigh_effort_supported_4_7_plus(model: str) -> None:
    assert _supports_xhigh_effort(model) is True


def test_xhigh_effort_rejected_4_6() -> None:
    assert _supports_xhigh_effort("claude-opus-4-6") is False


@pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8"])
def test_adaptive_thinking_supported_4_6_plus(model: str) -> None:
    assert _supports_adaptive_thinking(model) is True


def test_output_limit_resolves_for_4_8() -> None:
    # 4-8 has an explicit 128k entry; date-stamped variants resolve via substring.
    assert _get_anthropic_max_output("claude-opus-4-8") == 128_000
    assert _get_anthropic_max_output("claude-opus-4-8-20250715") == 128_000
