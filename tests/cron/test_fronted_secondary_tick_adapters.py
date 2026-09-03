"""Multiplexed secondary profiles that are FRONTED still deliver their cron.

Bug: the multiplex ticker hands each secondary only ``profile_adapters[name]``
so a secondary never ships cron through the default profile's bot. That rule
assumed the secondary's own bot is merely *not connected yet*, and had a
permanent hole:

The documented way to multiplex a secondary is a
``platforms.signal: {enabled: false}`` block — it stops the profile starting a
DUPLICATE adapter and declares the shared default adapter fronts it. Such a
profile owns no adapter and never will, so it fell through to ``{}`` forever,
``allow_disabled_native`` computed False, no transport resolved, and its own
``enabled: false`` block failed the delivery gate with
``platform 'signal' not configured/enabled`` on every single tick.

``_fronted_tick_adapters`` closes that hole for an EXPLICITLY disabled platform
only, so a profile with (or still connecting) its own bot is never mis-routed.
"""

from unittest.mock import MagicMock, patch

from cron.scheduler_provider import _fronted_tick_adapters
from gateway.config import Platform, PlatformConfig


def _config(platforms):
    config = MagicMock()
    config.platforms = platforms
    return config


def _run(own, shared, platforms):
    with patch("gateway.config.load_gateway_config", return_value=_config(platforms)):
        return _fronted_tick_adapters("manager", own, shared)


class TestFrontedSecondaryTickAdapters:
    def test_disabled_platform_borrows_the_shared_adapter(self):
        """enabled: false == 'the default adapter fronts me'. This is the bug."""
        shared_adapter = MagicMock()
        result = _run(
            {},
            {Platform.SIGNAL: shared_adapter},
            {Platform.SIGNAL: PlatformConfig(enabled=False)},
        )
        assert result[Platform.SIGNAL] is shared_adapter

    def test_enabled_platform_never_borrows(self):
        """A profile with its own bot must keep strict isolation even before
        that bot connects — borrowing here would ship cron via the wrong bot."""
        result = _run(
            {},
            {Platform.SIGNAL: MagicMock()},
            {Platform.SIGNAL: PlatformConfig(enabled=True)},
        )
        assert result == {}

    def test_absent_config_block_never_borrows(self):
        """No explicit block is not an explicit 'front me'; stay strict."""
        result = _run({}, {Platform.SIGNAL: MagicMock()}, {})
        assert result == {}

    def test_own_adapter_always_wins(self):
        """A connected own bot is never displaced by the shared one."""
        own_adapter = MagicMock()
        result = _run(
            {Platform.SIGNAL: own_adapter},
            {Platform.SIGNAL: MagicMock()},
            {Platform.SIGNAL: PlatformConfig(enabled=False)},
        )
        assert result[Platform.SIGNAL] is own_adapter

    def test_borrow_is_per_platform(self):
        """Only the disabled platform is borrowed, not the whole shared map."""
        signal_adapter = MagicMock()
        result = _run(
            {},
            {Platform.SIGNAL: signal_adapter, Platform.DISCORD: MagicMock()},
            {
                Platform.SIGNAL: PlatformConfig(enabled=False),
                Platform.DISCORD: PlatformConfig(enabled=True),
            },
        )
        assert result == {Platform.SIGNAL: signal_adapter}

    def test_no_shared_adapters_is_a_noop(self):
        assert _run({}, {}, {Platform.SIGNAL: PlatformConfig(enabled=False)}) == {}

    def test_config_read_failure_falls_back_to_strict(self):
        """A broken config must not break the tick, and must not mis-route."""
        with patch(
            "gateway.config.load_gateway_config", side_effect=RuntimeError("boom")
        ):
            result = _fronted_tick_adapters("manager", {}, {Platform.SIGNAL: MagicMock()})
        assert result == {}
