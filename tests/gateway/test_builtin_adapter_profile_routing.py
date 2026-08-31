"""Built-in adapters must resolve gateway.profile_routes at ingress.

Regression: ``build_source`` reads routes through ``self.gateway_runner``. The
runner attached that back-reference to plugin adapters and to two built-ins
(api_server, webhook) — every other routed built-in (Signal) stamped
``source.profile = None``. The adapter then keyed ``_active_sessions`` and the
clarify text-intercept bypass under ``agent:main:<chat>`` while the runner ran
the turn under ``agent:<profile>:<chat>``, so a reply to a clarify prompt missed
its pending entry and fell through to the busy handler's interrupt ack.
"""

from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import Platform
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner
from gateway.session import build_session_key


CHAT_ID = "group:routed-chat"


@pytest.fixture
def runner(monkeypatch):
    """A minimal runner that routes CHAT_ID to the served ``manager`` profile."""
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=True,
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
        profile_routes=parse_profile_routes(
            [
                {
                    "name": "routed-manager",
                    "platform": "signal",
                    "chat_id": CHAT_ID,
                    "profile": "manager",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes",
        lambda config: [("manager", "/tmp/manager")],
    )
    return runner


@pytest.fixture
def signal_adapter(runner, monkeypatch):
    """A real Signal adapter built the way the gateway builds it."""
    monkeypatch.setenv("SIGNAL_HTTP_URL", "http://127.0.0.1:8080")
    monkeypatch.setenv("SIGNAL_ACCOUNT", "+15550000000")
    config = PlatformConfig(
        enabled=True,
        extra={"http_url": "http://127.0.0.1:8080", "account": "+15550000000"},
    )
    adapter = runner._create_adapter(Platform.SIGNAL, config)
    assert adapter is not None, "Signal adapter should build from a valid config"
    return adapter


def test_builtin_adapter_stamps_the_routed_profile(signal_adapter):
    """The adapter's own ingress must resolve the route, not just the runner's."""
    source = signal_adapter.build_source(
        chat_id=CHAT_ID, chat_type="group", user_id="+15551111111", user_name="E X"
    )
    assert source.profile == "manager"


def test_adapter_and_runner_agree_on_the_session_key(runner, signal_adapter):
    """Both sides must derive one key, or the clarify bypass looks in the wrong lane."""
    source = signal_adapter.build_source(
        chat_id=CHAT_ID, chat_type="group", user_id="+15551111111", user_name="E X"
    )
    adapter_key = build_session_key(
        source,
        group_sessions_per_user=signal_adapter.config.extra.get(
            "group_sessions_per_user", True
        ),
        thread_sessions_per_user=signal_adapter.config.extra.get(
            "thread_sessions_per_user", False
        ),
        profile=signal_adapter._session_key_profile(source),
    )
    runner_key = build_session_key(
        source,
        group_sessions_per_user=runner.config.group_sessions_per_user,
        thread_sessions_per_user=runner.config.thread_sessions_per_user,
        profile=runner._profile_name_for_source(source),
    )
    assert adapter_key == runner_key == f"agent:manager:signal:group:{CHAT_ID}"


def test_unrouted_chat_stays_on_the_default_profile(signal_adapter):
    source = signal_adapter.build_source(
        chat_id="group:other-chat", chat_type="group", user_id="+1", user_name="E X"
    )
    assert source.profile is None
