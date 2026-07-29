"""Single-attachment Signal sends must use the attachment-scaled RPC timeout.

Regression test for the phantom "RPC send file failed" bug: send_document /
send_image_file / send_voice / send_video all route through _send_attachment,
which called self._rpc("send", params) with no timeout and therefore inherited
the 30s httpx client default. signal-cli uploads the attachment serially inside
the call, so a document upload that takes longer than 30s was aborted client
side and logged as a failure even though signal-cli completed the send.

The batched-image path (send_multiple_images) already used
_signal_send_timeout(n); this asserts the single-attachment paths do too.
"""

import inspect

from gateway.platforms.signal_rate_limit import _signal_send_timeout


def test_signal_send_timeout_floor_beats_httpx_default():
    """A single attachment must get more than the 30s httpx default."""
    assert _signal_send_timeout(1) >= 60.0
    assert _signal_send_timeout(1) > 30.0


def test_send_attachment_passes_scaled_timeout():
    """_send_attachment must not fall back to the client default timeout."""
    from gateway.platforms.signal import SignalAdapter

    src = inspect.getsource(SignalAdapter._send_attachment)
    assert "_signal_send_timeout(" in src, (
        "_send_attachment must pass an explicit attachment-scaled timeout to "
        "_rpc; without it the 30s httpx default truncates slow uploads and "
        "produces a phantom 'RPC send file failed'."
    )


def test_send_image_passes_scaled_timeout():
    """send_image (url/file:// path) shares the same failure mode."""
    from gateway.platforms.signal import SignalAdapter

    src = inspect.getsource(SignalAdapter.send_image)
    assert "_signal_send_timeout(" in src
