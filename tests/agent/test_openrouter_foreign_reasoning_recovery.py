"""OpenRouter 400 "encrypted_content was not issued to this caller" must strip replayed
``reasoning_details`` and retry, not burn three identical retries (memory-gc cron, 2026-10-07)."""
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import _recover_format_errors
from agent.turn_retry_state import TurnRetryState

_RAW = ('{"error":{"code":null,"message":"reasoning `encrypted_content` was not issued to this caller",'
        '"param":null,"type":"invalid_request_error"}}')


class _Err(Exception):
    status_code = 400
    body = {"error": {"message": "Provider returned error", "code": 400,
                      "metadata": {"raw": _RAW, "provider_name": "Meta"}}}


class _Agent:
    log_prefix = ""
    api_mode = "chat_completions"

    def __getattr__(self, name):
        return lambda *a, **k: None


def _history():
    return [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "a", "reasoning_details": [{"type": "reasoning.encrypted", "data": "X"}]}]


def test_strips_reasoning_details_once_then_gives_up():
    err, retry = _Err("HTTP 400"), TurnRetryState()
    classified = classify_api_error(err, provider="openrouter", model="meta/muse-spark-1.3-contributor")
    assert classified.reason == FailoverReason.invalid_encrypted_content
    messages, api_messages = _history(), _history()

    assert _recover_format_errors(_Agent(), err, classified, retry, messages, api_messages) is True
    assert not any("reasoning_details" in m for m in messages + api_messages)
    assert _recover_format_errors(_Agent(), err, classified, retry, _history(), _history()) is False


def test_nothing_to_strip_does_not_retry():
    err, retry = _Err("HTTP 400"), TurnRetryState()
    classified = classify_api_error(err, provider="openrouter", model="m")
    plain = [{"role": "assistant", "content": "a"}]
    assert _recover_format_errors(_Agent(), err, classified, retry, plain, list(plain)) is False
