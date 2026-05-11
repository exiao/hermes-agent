from types import SimpleNamespace

from run_agent import _extract_retry_after_seconds, _is_short_google_capacity_wait


class Headers(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class RetryAfterError(Exception):
    def __init__(self, retry_after=None, headers=None):
        super().__init__("rate limited")
        self.retry_after = retry_after
        self.response = SimpleNamespace(headers=Headers(headers or {}))


def test_extract_retry_after_prefers_exception_attribute():
    err = RetryAfterError(retry_after=13, headers={"Retry-After": "2"})

    assert _extract_retry_after_seconds(err) == 13


def test_extract_retry_after_reads_headers_when_attribute_missing():
    err = RetryAfterError(headers={"retry-after": "9"})

    assert _extract_retry_after_seconds(err) == 9


def test_extract_retry_after_caps_to_max_seconds():
    err = RetryAfterError(retry_after=300)

    assert _extract_retry_after_seconds(err, max_seconds=120) == 120


def test_extract_retry_after_rejects_non_finite_values():
    assert _extract_retry_after_seconds(RetryAfterError(retry_after="nan")) is None
    assert _extract_retry_after_seconds(RetryAfterError(retry_after="inf")) is None


def test_extract_retry_after_parses_gemini_reset_message():
    err = Exception(
        "Gemini quota exhausted (You have exhausted your capacity on this model. "
        "Your quota will reset after 13s.). Check /gquota for remaining daily requests."
    )

    assert _extract_retry_after_seconds(err) == 13


def test_short_google_capacity_wait_honors_cloudcode_retry_window():
    assert _is_short_google_capacity_wait(
        provider="google-gemini-cli",
        base_url="cloudcode-pa://google",
        retry_after=13,
    )
    assert _is_short_google_capacity_wait(
        provider=None,
        base_url="CloudCode-PA://google",
        retry_after=13,
    )


def test_short_google_capacity_wait_rejects_long_or_non_google_windows():
    assert not _is_short_google_capacity_wait(
        provider="google-gemini-cli",
        base_url="cloudcode-pa://google",
        retry_after=61,
    )
    assert not _is_short_google_capacity_wait(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        retry_after=13,
    )
