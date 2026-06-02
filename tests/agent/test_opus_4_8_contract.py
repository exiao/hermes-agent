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


class TestParseClaudeVersion:
    """Numeric version parsing future-proofs the capability checks."""

    @pytest.mark.parametrize("model,expected", [
        ("claude-opus-4-8", (4, 8)),
        ("claude-opus-4.8", (4, 8)),
        ("claude-opus-4-8-20250715", (4, 8)),
        ("anthropic/claude-opus-4-8", (4, 8)),
        ("anthropic.claude-opus-4.8", (4, 8)),
        ("claude-opus-4-7", (4, 7)),
        ("claude-opus-4-6", (4, 6)),
        ("claude-sonnet-4-6", (4, 6)),
        ("claude-opus-4-5", (4, 5)),
        ("claude-opus-4", (4, 0)),
        ("claude-sonnet-4", (4, 0)),
        ("claude-3-7-sonnet", (3, 7)),
        ("claude-3-5-sonnet", (3, 5)),
        ("claude-3-opus", (3, 0)),
        ("claude-opus-4-10", (4, 10)),
        # Future families parse correctly without any code change.
        ("claude-opus-4-9", (4, 9)),
        ("claude-opus-5-0", (5, 0)),
        ("claude-sonnet-5-2", (5, 2)),
    ])
    def test_parses_version(self, model: str, expected) -> None:
        from agent.anthropic_adapter import _parse_claude_version
        assert _parse_claude_version(model) == expected

    @pytest.mark.parametrize("model", [
        # A minor-less name with a date stamp must NOT read the date as minor.
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250101",
    ])
    def test_date_stamp_is_not_minor_version(self, model: str) -> None:
        from agent.anthropic_adapter import _parse_claude_version
        # Parses as major=4, minor=0 — so it stays on the pre-4.6 contract.
        assert _parse_claude_version(model) == (4, 0)

    @pytest.mark.parametrize("model", ["minimax", "qwen3", "gpt-5.5", "gemini-3-pro"])
    def test_non_claude_returns_none(self, model: str) -> None:
        from agent.anthropic_adapter import _parse_claude_version
        assert _parse_claude_version(model) is None


class TestFutureModelContractInferred:
    """A future Claude family (4.9, 5.0) inherits the 4.7+ contract from its
    version number alone — no substring-list edit required. This is the whole
    point of the parser: it prevents the 4.8-style breakage from recurring."""

    @pytest.mark.parametrize("model", [
        "claude-opus-4-9", "claude-opus-5-0", "claude-sonnet-5-2",
    ])
    def test_future_models_forbid_sampling_and_support_xhigh(self, model: str) -> None:
        assert _forbids_sampling_params(model) is True
        assert _supports_xhigh_effort(model) is True
        assert _supports_adaptive_thinking(model) is True

    def test_date_stamped_pre_4_6_model_stays_on_legacy_contract(self) -> None:
        # Regression guard: claude-sonnet-4-<date> must NOT be treated as 4.6+.
        model = "claude-sonnet-4-20250514"
        assert _supports_adaptive_thinking(model) is False
        assert _supports_xhigh_effort(model) is False
        assert _forbids_sampling_params(model) is False
