import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from gateway.config import Platform, SessionResetPolicy
from gateway.run_turn import GatewayTurnMixin
from gateway.session import SessionSource

@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['idle', 'suspended', 'resume_pending_expired'])
@pytest.mark.parametrize('platform', [Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
async def test_reset_notice_keeps_context_but_hides_cloud_admin_details(reason, platform):
    runner = MagicMock()
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner._adapter_for_source.return_value = adapter
    runner._reset_notice_session_info.return_value = 'model details'
    runner.session_store.config.get_reset_policy.return_value = SessionResetPolicy(notify=True)
    entry = MagicMock(auto_reset_reason=reason, reset_had_activity=True)
    source = SessionSource(platform=platform, chat_id='123', user_id='123')
    notes = []
    with patch('gateway.run_turn.build_channel_continuity_note', return_value=None):
        await GatewayTurnMixin._hmwa_deliver_auto_reset_notice(runner, entry, source, notes)
    assert notes
    assert entry.auto_reset_reason is None
    assert adapter.send.await_count == (0 if platform == Platform.WHATSAPP_CLOUD else 1)

@pytest.mark.asyncio
async def test_cloud_first_contact_does_not_offer_home_channel():
    runner = MagicMock()
    runner.async_session_store.has_any_sessions = AsyncMock(return_value=True)
    runner._deliver_platform_notice = AsyncMock()
    source = SessionSource(platform=Platform.WHATSAPP_CLOUD, chat_id='123', user_id='123')
    await GatewayTurnMixin._hmwa_first_contact_notes(runner, source, [], [])
    runner._deliver_platform_notice.assert_not_awaited()
    runner.config.get_home_channel.assert_not_called()
