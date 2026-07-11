"""Regression tests: auxiliary `_try_anthropic()` must honor
``providers.anthropic.base_url`` (e.g. a local billing proxy) when
``model.base_url`` is unset, mirroring the main session's
providers.<name>.* inheritance in hermes_cli/runtime_provider.py.

Without this, every Anthropic-routed side channel (goal judge, title
generation, memory extract, vision fallback) bypasses the operator's
proxy and 401s against api.anthropic.com.
"""
from unittest.mock import MagicMock, patch

import yaml


def _base_url_passed_to_build(mock_build):
    args, _kwargs = mock_build.call_args
    assert len(args) >= 2, f"expected (token, base_url), got args={args}"
    return args[1]


def _run_try_anthropic(tmp_path, monkeypatch, config: dict):
    from agent.auxiliary_client import _try_anthropic

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))

    with (
        patch(
            "agent.auxiliary_client._select_pool_entry", return_value=(False, None)
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="***",
        ),
        patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
    ):
        mock_build.return_value = MagicMock()
        client, model = _try_anthropic()

    assert client is not None, "auxiliary client must still be created"
    return _base_url_passed_to_build(mock_build)


class TestProvidersAnthropicBaseUrlInheritance:
    def test_providers_base_url_applies_when_model_base_url_unset(
        self, tmp_path, monkeypatch
    ):
        """The billing-proxy layout: model.base_url unset,
        providers.anthropic.base_url points at a local proxy."""
        actual = _run_try_anthropic(
            tmp_path,
            monkeypatch,
            {
                "model": {"provider": "anthropic", "default": "claude-haiku-4-5"},
                "providers": {
                    "anthropic": {"base_url": "http://127.0.0.1:18801"}
                },
            },
        )
        assert actual == "http://127.0.0.1:18801", (
            f"providers.anthropic.base_url must reach the auxiliary client; got {actual!r}"
        )

    def test_model_base_url_anthropic_host_wins_over_providers(
        self, tmp_path, monkeypatch
    ):
        """An Anthropic-compatible model.base_url keeps priority."""
        actual = _run_try_anthropic(
            tmp_path,
            monkeypatch,
            {
                "model": {
                    "provider": "anthropic",
                    "base_url": "https://api.anthropic.com",
                },
                "providers": {
                    "anthropic": {"base_url": "http://127.0.0.1:18801"}
                },
            },
        )
        assert actual == "https://api.anthropic.com"

    def test_foreign_model_base_url_still_gated_falls_back_to_providers(
        self, tmp_path, monkeypatch
    ):
        """#52608 gate unchanged: a foreign model.base_url is rejected, and the
        providers.anthropic.base_url then applies instead of the default."""
        actual = _run_try_anthropic(
            tmp_path,
            monkeypatch,
            {
                "model": {
                    "provider": "anthropic",
                    "base_url": "https://openrouter.ai/api/v1",
                },
                "providers": {
                    "anthropic": {"base_url": "http://127.0.0.1:18801"}
                },
            },
        )
        assert actual == "http://127.0.0.1:18801"

    def test_no_providers_block_keeps_default(self, tmp_path, monkeypatch):
        actual = _run_try_anthropic(
            tmp_path,
            monkeypatch,
            {"model": {"provider": "anthropic"}},
        )
        assert actual == "https://api.anthropic.com"
