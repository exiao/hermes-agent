"""
Tests for BasePlatformAdapter._send_with_retry and _is_retryable_error.

Verifies that:
- Transient network errors trigger retry with backoff
- Permanent errors fall back to plain-text immediately (no retry)
- User receives a delivery-failure notice when all retries are exhausted
- Successful sends on retry return success
- SendResult.retryable flag is respected
"""
import pytest
from unittest.mock import AsyncMock, patch

from gateway.platforms.base import BasePlatformAdapter, SendResult, _RETRYABLE_ERROR_PATTERNS
from gateway.platforms.base import Platform, PlatformConfig


# Real Signal send failure observed for group-send HTTP 5xx: the JSON-RPC error
# message wraps the Java exception whose package name contains "network".
_SIGNAL_500_ERROR = (
    "Failed to send message: "
    "org.signal.network.exceptions.NonSuccessfulResponseCodeException: "
    "[500] Bad response: 500 : {\"code\":500,\"message\":\"There was an error "
    "processing your request. It has been logged (ID 05db3b542d515cd7).\"} "
    "(IOException) (UnexpectedErrorException)"
)


# ---------------------------------------------------------------------------
# Minimal concrete adapter for testing (no real network)
# ---------------------------------------------------------------------------

class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        cfg = PlatformConfig()
        super().__init__(cfg, Platform.TELEGRAM)
        self._send_results = []   # queue of SendResult to return per call
        self._send_calls = []     # record of (chat_id, content) sent

    def _next_result(self) -> SendResult:
        if self._send_results:
            return self._send_results.pop(0)
        return SendResult(success=True, message_id="ok")

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs) -> SendResult:
        self._send_calls.append((chat_id, content))
        return self._next_result()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send_typing(self, chat_id, metadata=None) -> None:
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "test", "type": "direct", "chat_id": chat_id}


# ---------------------------------------------------------------------------
# _is_retryable_error
# ---------------------------------------------------------------------------

class TestIsRetryableError:
    def test_none_is_not_retryable(self):
        assert not _StubAdapter._is_retryable_error(None)

    def test_empty_string_is_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("")

    @pytest.mark.parametrize("pattern", _RETRYABLE_ERROR_PATTERNS)
    def test_known_pattern_is_retryable(self, pattern):
        assert _StubAdapter._is_retryable_error(f"httpx.{pattern.title()}: connection dropped")

    def test_permission_error_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("Forbidden: bot was blocked by the user")

    def test_bad_request_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("Bad Request: can't parse entities")

    def test_case_insensitive(self):
        assert _StubAdapter._is_retryable_error("CONNECTERROR: host unreachable")

    def test_timeout_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("ReadTimeout: request timed out")

    def test_timed_out_not_retryable(self):
        assert not _StubAdapter._is_retryable_error("Timed out waiting for response")

    def test_connect_timeout_is_retryable(self):
        assert _StubAdapter._is_retryable_error("ConnectTimeout: connection timed out")

    def test_signal_500_not_retryable_despite_network_in_package(self):
        """A Signal HTTP 500 surfaces as
        ``org.signal.network.exceptions.NonSuccessfulResponseCodeException`` —
        the package name contains "network" but the send is ambiguous (may have
        been delivered), so it must NOT be classified retryable. Regression for
        duplicate group replies during Signal-side 5xx windows."""
        assert not _StubAdapter._is_retryable_error(_SIGNAL_500_ERROR)

    def test_genuine_signal_push_network_error_still_retryable(self):
        """A real connection-level failure in the same package (no HTTP status)
        is still retryable — the request never reached the server."""
        assert _StubAdapter._is_retryable_error(
            "org.signal.network.exceptions.PushNetworkException: broken pipe"
        )


# ---------------------------------------------------------------------------
# _is_ambiguous_delivery_error
# ---------------------------------------------------------------------------

class TestIsAmbiguousDeliveryError:
    def test_none_is_not_ambiguous(self):
        assert not _StubAdapter._is_ambiguous_delivery_error(None)

    @pytest.mark.parametrize("code", ["500", "502", "503", "504"])
    def test_5xx_is_ambiguous(self, code):
        assert _StubAdapter._is_ambiguous_delivery_error(f"Bad response: {code}")

    @pytest.mark.parametrize(
        "error",
        ["HTTP 500: Graph API unavailable", "graph error 1 (HTTP 503): unavailable"],
    )
    def test_http_5xx_is_ambiguous(self, error):
        assert _StubAdapter._is_ambiguous_delivery_error(error)

    def test_signal_500_is_ambiguous(self):
        assert _StubAdapter._is_ambiguous_delivery_error(_SIGNAL_500_ERROR)

    def test_4xx_is_not_ambiguous(self):
        """4xx is an outright rejection (nothing delivered) — still eligible for
        the plain-text fallback."""
        assert not _StubAdapter._is_ambiguous_delivery_error("[400] Bad response: 400")

    def test_no_status_is_not_ambiguous(self):
        assert not _StubAdapter._is_ambiguous_delivery_error("ConnectError: refused")


# ---------------------------------------------------------------------------
# _is_timeout_error
# ---------------------------------------------------------------------------

class TestIsTimeoutError:
    def test_none_is_not_timeout(self):
        assert not _StubAdapter._is_timeout_error(None)

    def test_empty_is_not_timeout(self):
        assert not _StubAdapter._is_timeout_error("")

    def test_timed_out(self):
        assert _StubAdapter._is_timeout_error("Timed out waiting for response")

    def test_read_timeout(self):
        assert _StubAdapter._is_timeout_error("ReadTimeout: request timed out")

    def test_write_timeout(self):
        assert _StubAdapter._is_timeout_error("WriteTimeout: send stalled")

    def test_connect_timeout_not_flagged(self):
        """ConnectTimeout is a connection error, not a delivery-ambiguous timeout."""
        assert not _StubAdapter._is_timeout_error("ConnectTimeout: host unreachable")

    def test_connection_error_not_timeout(self):
        assert not _StubAdapter._is_timeout_error("ConnectionError: host unreachable")


# ---------------------------------------------------------------------------
# _send_with_retry — success on first attempt
# ---------------------------------------------------------------------------

class TestSendWithRetrySuccess:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        adapter = _StubAdapter()
        adapter._send_results = [SendResult(success=True, message_id="123")]
        result = await adapter._send_with_retry("chat1", "hello")
        assert result.success
        assert len(adapter._send_calls) == 1

    @pytest.mark.asyncio
    async def test_returns_message_id(self):
        adapter = _StubAdapter()
        adapter._send_results = [SendResult(success=True, message_id="abc")]
        result = await adapter._send_with_retry("chat1", "hi")
        assert result.message_id == "abc"


# ---------------------------------------------------------------------------
# _send_with_retry — network error with successful retry
# ---------------------------------------------------------------------------

class TestSendWithRetryNetworkRetry:
    @pytest.mark.asyncio
    async def test_retries_on_connect_error_and_succeeds(self):
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="httpx.ConnectError: connection refused"),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert result.success
        assert len(adapter._send_calls) == 2  # initial + 1 retry

    @pytest.mark.asyncio
    async def test_timeout_not_retried_to_prevent_duplicates(self):
        """ReadTimeout is NOT retried because the request may have reached
        the server — retrying a non-idempotent send risks duplicate delivery.
        It also skips plain-text fallback (timeout is not a formatting issue)."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="ReadTimeout: request timed out"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=3, base_delay=0)
        # No retry, no fallback — timeout returns failure immediately
        mock_sleep.assert_not_called()
        assert not result.success
        assert len(adapter._send_calls) == 1

    @pytest.mark.asyncio
    async def test_signal_500_not_retried_and_no_fallback(self):
        """A Signal group-send HTTP 500 is ambiguous (may have been delivered).
        It must not be retried and must not trigger the plain-text fallback,
        otherwise the recipient gets duplicate copies. Only one send happens."""
        adapter = _StubAdapter()
        adapter._send_results = [SendResult(success=False, error=_SIGNAL_500_ERROR)]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=3, base_delay=0)
        mock_sleep.assert_not_called()
        assert not result.success
        assert len(adapter._send_calls) == 1  # no retry, no fallback → no duplicate

    @pytest.mark.asyncio
    async def test_signal_500_retried_when_platform_opts_in(self):
        """A platform that sets retryable=True on its 5xx still gets a retry —
        the ambiguous-delivery guard only overrides the error-string heuristic,
        not an explicit opt-in."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error=_SIGNAL_500_ERROR, retryable=True),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert result.success
        assert len(adapter._send_calls) == 2

    @pytest.mark.asyncio
    async def test_signal_500_after_transient_retry_has_no_plaintext_fallback(self):
        """An ambiguous 5xx on a retry may still mean the message was delivered.

        The retry loop must return that failure rather than falling through to
        the plain-text fallback, which would duplicate the reply.
        """
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="httpx.ConnectError: connection refused"),
            SendResult(success=False, error=_SIGNAL_500_ERROR),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert not result.success
        assert len(adapter._send_calls) == 2

    @pytest.mark.asyncio
    async def test_connect_timeout_still_retried(self):
        """ConnectTimeout is safe to retry — the connection was never established."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="ConnectTimeout: connection timed out"),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert result.success
        assert len(adapter._send_calls) == 2

    @pytest.mark.asyncio
    async def test_retryable_flag_respected(self):
        """SendResult.retryable=True should trigger retry even if error string doesn't match."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="internal platform error", retryable=True),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert result.success
        assert len(adapter._send_calls) == 2

    @pytest.mark.asyncio
    async def test_network_to_nonnetwork_transition_falls_back_to_plaintext(self):
        """If error switches from network to formatting mid-retry, fall through to plain-text fallback."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="httpx.ConnectError: host unreachable"),
            SendResult(success=False, error="Bad Request: can't parse entities"),
            SendResult(success=True, message_id="fallback_ok"),  # plain-text fallback
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "**bold**", max_retries=2, base_delay=0)
        assert result.success
        # 3 calls: initial (network) + 1 retry (non-network, breaks loop) + plain-text fallback
        assert len(adapter._send_calls) == 3
        assert "plain text" in adapter._send_calls[-1][1].lower()


# ---------------------------------------------------------------------------
# _send_with_retry — all retries exhausted → user notification
# ---------------------------------------------------------------------------

class TestSendWithRetryExhausted:
    @pytest.mark.asyncio
    async def test_sends_user_notice_after_exhaustion(self):
        adapter = _StubAdapter()
        network_err = SendResult(success=False, error="httpx.ConnectError: host unreachable")
        # initial + 2 retries + notice attempt
        adapter._send_results = [network_err, network_err, network_err, SendResult(success=True)]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        # Result is the last failed one (before notice)
        assert not result.success
        # 4 total calls: 1 initial + 2 retries + 1 notice
        assert len(adapter._send_calls) == 4
        # The notice content should mention delivery failure
        notice_content = adapter._send_calls[-1][1]
        assert "delivery failed" in notice_content.lower() or "Message delivery failed" in notice_content

    @pytest.mark.asyncio
    async def test_notice_send_exception_doesnt_propagate(self):
        """If the notice itself throws, _send_with_retry should not raise."""
        adapter = _StubAdapter()
        network_err = SendResult(success=False, error="ConnectError")
        adapter._send_results = [network_err, network_err, network_err]

        original_send = adapter.send
        call_count = [0]

        async def send_with_notice_failure(chat_id, content, **kwargs):
            call_count[0] += 1
            if call_count[0] > 3:
                raise RuntimeError("notice send also failed")
            return network_err

        adapter.send = send_with_notice_failure
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=0)
        assert not result.success  # still failed, but no exception raised


# ---------------------------------------------------------------------------
# _send_with_retry — non-network failure → plain-text fallback (no retry)
# ---------------------------------------------------------------------------

class TestSendWithRetryFallback:
    @pytest.mark.asyncio
    async def test_non_network_error_falls_back_immediately(self):
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="Bad Request: can't parse entities"),
            SendResult(success=True, message_id="fallback_ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "**bold**", max_retries=2, base_delay=0)
        # No sleep — no retry loop for non-network errors
        mock_sleep.assert_not_called()
        assert result.success
        assert len(adapter._send_calls) == 2
        # Fallback content should be plain-text notice
        assert "plain text" in adapter._send_calls[1][1].lower()

    @pytest.mark.asyncio
    async def test_fallback_failure_logged_but_not_raised(self):
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="Forbidden: bot blocked"),
            SendResult(success=False, error="Forbidden: bot blocked"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2)
        assert not result.success
        assert len(adapter._send_calls) == 2  # original + fallback only


# ---------------------------------------------------------------------------
# _send_with_retry — retry_after honor
# ---------------------------------------------------------------------------

class TestSendWithRetryAfter:
    @pytest.mark.asyncio
    async def test_retry_after_honored_on_first_retry(self):
        """When the initial result has retry_after, the first retry waits that long."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="Flood control exceeded. Retry in 37 seconds",
                       retryable=True, retry_after=37.0),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=2.0)
        assert result.success
        # First sleep should use retry_after (~37s + jitter), not base_delay (~2s)
        first_sleep = mock_sleep.call_args_list[0][0][0]
        assert first_sleep >= 36.0  # 37 - 1 (max jitter)

    @pytest.mark.asyncio
    async def test_retry_after_from_subsequent_result(self):
        """If a retry itself returns retry_after, the next retry honors it."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="ConnectError", retryable=True),
            SendResult(success=False, error="Flood control exceeded. Retry in 30 seconds",
                       retryable=True, retry_after=30.0),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=3, base_delay=2.0)
        assert result.success
        # Second sleep should use the retry_after from the second result
        second_sleep = mock_sleep.call_args_list[1][0][0]
        assert second_sleep >= 29.0  # 30 - 1 (max jitter)

    @pytest.mark.asyncio
    async def test_no_retry_after_uses_default_backoff(self):
        """Without retry_after, default exponential backoff is used."""
        adapter = _StubAdapter()
        adapter._send_results = [
            SendResult(success=False, error="ConnectError", retryable=True),
            SendResult(success=True, message_id="ok"),
        ]
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await adapter._send_with_retry("chat1", "hello", max_retries=2, base_delay=2.0)
        assert result.success
        # Sleep should be ~2s (base_delay * 2^0 + jitter), NOT 37s
        first_sleep = mock_sleep.call_args_list[0][0][0]
        assert first_sleep < 5.0
