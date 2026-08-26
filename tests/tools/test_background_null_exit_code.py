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
