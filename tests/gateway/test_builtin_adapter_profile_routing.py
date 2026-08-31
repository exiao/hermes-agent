"""Built-in adapters must resolve gateway.profile_routes at ingress.

Regression: ``build_source`` reads routes through ``self.gateway_runner``.
The runner only attached that back-reference to plugin adapters and two
built-ins, so a routed built-in platform (Signal) stamped
``source.profile = None``. The adapter then keyed ``_active_sessions`` and the
clarify text-intercept bypass under ``agent:main:…`` while the runner ran the
turn under ``agent:<profile>:…`` — a clarify reply missed its pending entry and
fell through to the busy handler's "Interrupting current task" ack.
"""

from types import SimpleNamespace

from gateway.platforms.base import Platform
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner


CHAT_ID = "group:routed-chat"


def _runner_with_route():
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
    return runner


def test_created_builtin_adapter_gets_runner_backref(monkeypatch):
    runner = _runner_with_route()
    built = SimpleNamespace(gateway_runner=None)
    monkeypatch.setattr(
        GatewayRunner, "_build_adapter", lambda self, platform, config: built
    )

    adapter = runner._create_adapter(Platform.SIGNAL, SimpleNamespace(extra={}))

    assert adapter is built
    assert adapter.gateway_runner is runner


def test_routed_source_carries_the_profile(monkeypatch):
    """With the back-reference in place, build_source stamps the profile."""
    runner = _runner_with_route()
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes",
        lambda config: [("manager", "/tmp/manager")],
    )

    source = SimpleNamespace(
        platform=Platform.SIGNAL,
        chat_id=CHAT_ID,
        thread_id=None,
        guild_id=None,
        parent_chat_id=None,
    )
    assert runner._profile_name_for_source(source) == "manager"


def test_unrouted_chat_stays_on_the_default_profile(monkeypatch):
    runner = _runner_with_route()
    monkeypatch.setattr(
        "gateway.run._multiplex_profile_homes",
        lambda config: [("manager", "/tmp/manager")],
    )

    source = SimpleNamespace(
        platform=Platform.SIGNAL,
        chat_id="group:some-other-chat",
        thread_id=None,
        guild_id=None,
        parent_chat_id=None,
    )
    assert runner._profile_name_for_source(source) is None
