"""A failure ping must say WHY, not just that something failed.

gave_up / crashed / timed_out used to send only a headline. A real event from
the board rendered as:

    ⏱ Kanban t_afeb884a timed out (max_runtime=0s); will retry

while its payload held the actual cause:

    error: "Iteration budget exhausted (200/200) — task could not complete
            within the allowed iterations"

The reason was in the DB the whole time. The message dropped it, so the reader
had to open the board to learn anything. The 0s was invented too: that event
shape carries no limit_seconds, so the fallback printed a limit never set.
"""

from __future__ import annotations

import pytest

from gateway.kanban_watchers import _NOTIFY_DETAIL_MAX, _failure_detail

BUDGET_ERROR = (
    "Iteration budget exhausted (200/200) — task could not complete "
    "within the allowed iterations"
)


def test_the_reason_a_task_failed_is_reported() -> None:
    """The exact payload from t_afeb884a, which pinged with no cause."""
    detail = _failure_detail(
        {"error": BUDGET_ERROR, "failures": 1, "retry_status": "ready"}
    )

    assert "Iteration budget exhausted (200/200)" in detail


def test_no_fabricated_runtime_when_the_event_has_no_limit() -> None:
    """The old code printed max_runtime=0s for this shape."""
    detail = _failure_detail({"error": BUDGET_ERROR, "retry_status": "ready"})

    assert "0s" not in detail


def test_runtime_is_reported_when_both_halves_are_real() -> None:
    detail = _failure_detail(
        {"elapsed_seconds": 5403, "limit_seconds": 5400, "retry_status": "ready"}
    )

    assert "ran 5403s of 5400s" in detail


def test_attempts_are_reported_against_their_ceiling() -> None:
    detail = _failure_detail(
        {
            "error": "pid 18342 exited with code 1",
            "failures": 2,
            "effective_limit": 2,
            "limit_source": "dispatcher",
        }
    )

    assert "attempt 2 of 2 (dispatcher limit)" in detail


@pytest.mark.parametrize(
    ("retry_status", "expected"),
    [("ready", "will retry"), ("gave_up", "not retrying (gave_up)")],
    ids=["retrying", "finished"],
)
def test_whether_it_retries_is_read_not_assumed(
    retry_status: str, expected: str
) -> None:
    """"will retry" was previously asserted unconditionally — a guess that was
    wrong whenever the dispatcher had stopped."""
    assert expected in _failure_detail({"retry_status": retry_status})


def test_gave_up_is_terminal_even_when_source_phase_says_ready() -> None:
    detail = _failure_detail(
        {"retry_status": "ready"},
        terminal=True,
    )

    assert "not retrying (blocked)" in detail
    assert "will retry" not in detail


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"pid": 18342, "exit_kind": "nonzero_exit", "exit_code": 1}, "exited with code 1"),
        ({"pid": 18342, "exit_kind": "signaled", "exit_code": 9}, "killed by signal 9"),
    ],
)
def test_crash_exit_details_are_reported(payload: dict, expected: str) -> None:
    assert expected in _failure_detail(payload)


@pytest.mark.parametrize("exit_kind", ["nonzero_exit", "signaled"])
def test_crash_exit_details_omit_missing_pid(exit_kind: str) -> None:
    detail = _failure_detail({"exit_kind": exit_kind, "exit_code": 1})

    assert "pid None" not in detail


def test_invalid_failure_count_does_not_crash() -> None:
    assert _failure_detail({"failures": "not-a-number", "effective_limit": 3}) == ""


def test_an_empty_payload_adds_nothing() -> None:
    """No payload must not mean an empty line or a stray separator."""
    assert _failure_detail(None) == ""
    assert _failure_detail({}) == ""


def test_a_field_that_is_absent_is_not_mentioned() -> None:
    detail = _failure_detail({"retry_status": "ready"})

    assert "attempt" not in detail
    assert "ran " not in detail
    assert "budget" not in detail


def test_a_pathological_error_is_clipped_visibly() -> None:
    """One runaway traceback must not flood the channel, and the clip must
    advertise itself rather than cut mid-sentence."""
    detail = _failure_detail({"error": "x" * (_NOTIFY_DETAIL_MAX + 500)})

    assert "more chars; see board" in detail


def test_the_detail_starts_on_its_own_line() -> None:
    """It is appended to a headline, so it must not run on."""
    assert _failure_detail({"retry_status": "ready"}).startswith("\n")
