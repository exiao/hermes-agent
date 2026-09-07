"""Credential rotation must not carry route-scoped TLS policy."""

from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from agent.credential_pool import PooledCredential
from run_agent import AIAgent


def _agent(**attrs) -> Any:
    agent = object.__new__(AIAgent)
    for name, value in attrs.items():
        setattr(agent, name, value)
    return agent


def test_credential_rotation_replaces_route_scoped_tls_settings():
    agent = _agent(
        api_mode="chat_completions",
        provider="custom",
        model="shared-model",
        api_key="old",
        base_url="https://a.example/v1",
        _client_kwargs={
            "api_key": "old",
            "base_url": "https://a.example/v1",
            "ssl_verify": False,
            "ssl_ca_cert": "/a.pem",
        },
        _apply_client_headers_for_base_url=MagicMock(),
        _replace_primary_openai_client=MagicMock(),
    )
    agent._reapply_route_client_config = MethodType(
        AIAgent._reapply_route_client_config,
        agent,
    )
    entry = SimpleNamespace(
        runtime_api_key="new",
        access_token="",
        runtime_base_url="https://b.example/v1",
        base_url="https://b.example/v1",
    )
    config = {
        "custom_providers": [
            {
                "name": "b",
                "base_url": "https://b.example/v1",
                "ssl_verify": True,
            }
        ]
    }

    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        AIAgent._swap_credential(agent, entry)

    assert agent._client_kwargs["ssl_verify"] is True
    assert "ssl_ca_cert" not in agent._client_kwargs
    agent._replace_primary_openai_client.assert_called_once_with(
        reason="credential_rotation"
    )


def test_credential_rotation_does_not_carry_global_headers_across_routes():
    agent = _agent(
        api_mode="chat_completions",
        provider="custom",
        model="shared-model",
        api_key="old",
        base_url="https://a.example/v1",
        _client_kwargs={
            "api_key": "old",
            "base_url": "https://a.example/v1",
            "default_headers": {"Authorization": "old-secret"},
        },
        _replace_primary_openai_client=MagicMock(),
    )
    agent._apply_client_headers_for_base_url = MethodType(
        AIAgent._apply_client_headers_for_base_url,
        agent,
    )
    agent._apply_user_default_headers = MethodType(
        AIAgent._apply_user_default_headers,
        agent,
    )
    agent._reapply_route_client_config = MethodType(
        AIAgent._reapply_route_client_config,
        agent,
    )
    entry = SimpleNamespace(
        runtime_api_key="new",
        access_token="",
        runtime_base_url="https://b.example/v1",
        base_url="https://b.example/v1",
    )
    config = {
        "model": {
            "default_headers": {"Authorization": "global-secret"},
        },
        "custom_providers": [
            {
                "name": "b",
                "base_url": "https://b.example/v1",
                "extra_headers": {"X-Route": "b"},
            }
        ],
    }

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=config["custom_providers"],
        ),
    ):
        AIAgent._swap_credential(agent, entry)

    headers = agent._client_kwargs["default_headers"]
    assert "Authorization" not in headers
    assert headers["X-Route"] == "b"


def test_credential_rotation_updates_read_only_runtime_base_url():
    """A configured proxy survives rotation with the real pooled entry type."""
    agent = _agent(
        api_mode="chat_completions",
        provider="openai-codex",
        model="gpt-5.6-luna-900k",
        api_key="old",
        base_url="https://old.example/v1",
        _client_kwargs={
            "api_key": "old",
            "base_url": "https://old.example/v1",
        },
        _apply_client_headers_for_base_url=MagicMock(),
        _replace_primary_openai_client=MagicMock(),
    )
    agent._reapply_route_client_config = MethodType(
        AIAgent._reapply_route_client_config,
        agent,
    )
    entry = PooledCredential(
        provider="openai-codex",
        id="next",
        label="next",
        auth_type="oauth",
        priority=1,
        source="test",
        access_token="new",
        base_url="https://stale.example/v1",
    )

    with patch(
        "agent.credential_pool._get_configured_provider_base_url",
        return_value="https://proxy.example/v1",
    ):
        AIAgent._swap_credential(agent, entry)  # type: ignore[arg-type]

    assert agent.base_url == "https://proxy.example/v1"
    assert agent._client_kwargs["base_url"] == "https://proxy.example/v1"
    agent._replace_primary_openai_client.assert_called_once_with(
        reason="credential_rotation"
    )
