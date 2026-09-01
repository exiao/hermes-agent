"""Cron delivery when a live native adapter has no per-profile config block.

Bug: a multiplexed secondary profile (``gateway.multiplex_profiles`` with a
``profile_routes`` entry) is fronted by the DEFAULT profile's Signal adapter —
it owns no ``platforms.signal`` block of its own, and setting one to
``enabled: false`` is the documented way to stop it from starting a duplicate
adapter. ``resolve_delivery_transport`` correctly resolves the live native
adapter (config block absent OR enabled), but the cron delivery loop then
re-applied a native ``pconfig.enabled`` gate that only exempted RELAY — so
every cron job in that profile failed with
``platform 'signal' not configured/enabled`` and the manager's project
check-ins never reached the chat.

The same shape hits the documented "explicitly supplied live adapter with no
config block" case that ``resolve_delivery_transport`` promises to support.
"""

import asyncio
import logging
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from gateway.delivery import resolve_delivery_transport


def _clear_home_env(monkeypatch):
    for var in ("SIGNAL_HOME_CHANNEL", "SIGNAL_HOME_CHANNEL_THREAD_ID"):
        monkeypatch.delenv(var, raising=False)


class TestNativeAdapterWithoutProfileConfig:
    def _job(self):
        return {
            "id": "mgr-checkin",
            "name": "cpe-mono check-in",
            "deliver": "origin",
            "origin": {"platform": "signal", "chat_id": "group:abc="},
        }

    def _run(self, adapters, gateway_config):
        loop = MagicMock()
        loop.is_running.return_value = True

        def fake_run_coro(coro, _loop):
            future = Future()
            try:
                future.set_result(asyncio.run(coro))
            except BaseException as e:  # noqa: BLE001
                future.set_exception(e)
            return future

        router = MagicMock()

        async def _deliver_to_platform(target, content, metadata):
            return {"success": True, "raw_response": None}

        router._deliver_to_platform = _deliver_to_platform

        with patch("gateway.config.load_gateway_config",
                   return_value=gateway_config), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch("gateway.delivery.DeliveryRouter", return_value=router), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
            return _deliver_result(self._job(), "3 cards blocked.",
                                   adapters=adapters, loop=loop)

    def _config(self, platforms):
        config = MagicMock()
        config.platforms = platforms
        config.get_home_channel = lambda p: None
        return config

    def test_live_shared_native_adapter_with_disabled_profile_config_delivers(self, monkeypatch):
        """A routed secondary may disable its duplicate while using the shared adapter."""
        _clear_home_env(monkeypatch)
        result = self._run(
            {Platform.SIGNAL: AsyncMock()},
            self._config({Platform.SIGNAL: PlatformConfig(enabled=False)}),
        )
        assert result is None  # None == delivered without errors

    def test_explicitly_disabled_native_adapter_still_rejected(self, monkeypatch):
        """An explicit ``enabled: false`` block with NO live adapter must still
        be refused — the gate is only bypassed for a RESOLVED transport."""
        _clear_home_env(monkeypatch)
        config = self._config({Platform.SIGNAL: PlatformConfig(enabled=False)})
        result = self._run({}, config)
        assert result is not None
        assert "not configured/enabled" in result

    def test_no_adapter_and_no_config_still_rejected(self, monkeypatch):
        """Nothing live and nothing configured stays an error."""
        _clear_home_env(monkeypatch)
        result = self._run({}, self._config({}))
        assert result is not None
        assert "not configured/enabled" in result

    def test_no_adapter_with_enabled_config_logs_info_with_transport_cause(
        self, monkeypatch, caplog
    ):
        """An enabled target without a live transport is not a config failure."""
        _clear_home_env(monkeypatch)
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            result = self._run(
                {},
                self._config({Platform.SIGNAL: PlatformConfig(enabled=True)}),
            )

        assert result is not None
        assert "not configured/enabled" not in result
        assert "no live adapter or relay available" in result
        matching = [
            record
            for record in caplog.records
            if "no live adapter or relay available" in record.message
        ]
        assert matching
        assert all(record.levelno == logging.INFO for record in matching)

    def test_disabled_native_requires_shared_transport_opt_in(self):
        """The resolver keeps explicit disabled-native behavior strict by default."""
        config = self._config({Platform.SIGNAL: PlatformConfig(enabled=False)})
        adapter = AsyncMock()

        assert resolve_delivery_transport(Platform.SIGNAL, config, {Platform.SIGNAL: adapter}) is None
        transport = resolve_delivery_transport(
            Platform.SIGNAL,
            config,
            {Platform.SIGNAL: adapter},
            allow_disabled_native=True,
        )
        assert transport is not None
        assert transport.adapter is adapter
