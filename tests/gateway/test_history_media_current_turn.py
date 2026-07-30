"""Regression: current-turn MEDIA must not be treated as prior history."""

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


class _Store:
    def __init__(self, transcript):
        self.transcript = transcript

    def peek_session_id(self, _session_key):
        return "session-1"

    def load_transcript(self, _session_id):
        return list(self.transcript)


class _Adapter(BasePlatformAdapter):
    def __init__(self, transcript):
        super().__init__(PlatformConfig(), Platform.SIGNAL)
        self.set_session_store(_Store(transcript))

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def test_current_turn_media_is_not_treated_as_history():
    path = "/tmp/current.png"
    adapter = _Adapter([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "make image"},
        {"role": "assistant", "tool_calls": [{"id": "t1", "function": {"name": "image_generate"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": f"MEDIA:{path}"},
        {"role": "assistant", "content": f"MEDIA:{path}"},
        {"role": "user", "content": "[alice|u2]\nunrelated group message", "observed": True},
    ])
    paths = adapter._history_media_paths_for_session("session-key")
    assert not paths or path not in paths


def test_redirect_correction_keeps_pre_redirect_media_out_of_history():
    prior_path = "/tmp/prior-redirect.png"
    current_path = "/tmp/current-redirect.png"
    adapter = _Adapter([
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": f"MEDIA:{prior_path}"},
        {"role": "user", "content": "make image"},
        {"role": "assistant", "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "tool_call_id": "t1", "content": f"MEDIA:{current_path}"},
        {
            "role": "assistant",
            "content": "",
            "api_content": "[This response was interrupted by a user correction.]",
            "display_kind": "hidden",
        },
        {"role": "user", "content": "change course"},
        {"role": "assistant", "content": f"MEDIA:{current_path}"},
    ])
    assert adapter._history_media_paths_for_session("session-key") == {prior_path}


def test_prior_turn_media_remains_in_history_dedup_set():
    path = "/tmp/prior.png"
    adapter = _Adapter([
        {"role": "user", "content": "make image"},
        {"role": "assistant", "content": f"Done. MEDIA:{path}"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "you are welcome"},
    ])
    assert adapter._history_media_paths_for_session("session-key") == {path}
