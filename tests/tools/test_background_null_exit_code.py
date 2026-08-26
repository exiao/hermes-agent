"""A recovered background process must not report a fabricated exit code.

Hermes checkpoints running background processes so they survive a gateway
restart. On restart they are re-registered as ``detached=True``: the pipe is
gone, so only the host PID is known.

When such a session's PID is no longer ours, ``_refresh_detached_session``
marks it exited and sets ``exit_code = None`` because the real code is
unrecoverable. That ``None`` then flows straight into the completion
notification, which renders it as:

    [IMPORTANT: Background process <id> exited (exit code None).

"exited (exit code None)" reads as a finished process whose status is simply
unknown-but-fine. In practice the user cannot tell that apart from a real
completion, and a still-running job looks finished. The notification must say
the exit code is unavailable, not print ``None``.
"""
from __future__ import annotations

from tools.process_registry import format_process_notification


def _completion(**over):
    evt = {
        "type": "completion",
        "session_id": "proc_test",
        "command": "bash -c 'long job' > /tmp/out.txt 2>&1",
        "exit_code": 0,
        "completion_reason": "exited",
        "output": "",
    }
    evt.update(over)
    return evt


def test_real_exit_code_still_reads_normally():
    """A genuine clean exit is unchanged."""
    text = format_process_notification(_completion(exit_code=0))
    assert "completed normally" in text
    assert "exit code 0" in text


def test_real_failure_still_reads_normally():
    text = format_process_notification(_completion(exit_code=1))
    assert "exit code 1" in text


def test_unknown_exit_code_is_not_rendered_as_none():
    """A recovered session with no waitable handle must not print `None`.

    ``_refresh_detached_session`` sets ``exit_code = None`` deliberately: the
    process object is gone and the code cannot be recovered. The notification
    has to communicate that, rather than presenting None as a result.
    """
    text = format_process_notification(_completion(exit_code=None))

    assert "exit code None" not in text, (
        "notification prints a literal None as if it were an exit status"
    )
    assert "unavailable" in text.lower() or "unknown" in text.lower(), (
        f"notification should say the code is unavailable; got: {text[:200]}"
    )


def test_unknown_exit_code_is_not_called_completed_normally():
    """None is not success. It must not be dressed up as a clean finish."""
    text = format_process_notification(_completion(exit_code=None))
    assert "completed normally" not in text
