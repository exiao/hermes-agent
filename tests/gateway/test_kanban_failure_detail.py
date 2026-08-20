"""A failure ping must say WHY, not just that something failed.

gave_up / crashed / timed_out used to send only a headline. A real event
rendered as "⏱ Kanban t_afeb884a timed out (max_runtime=0s); will retry" while
its payload held the cause ("Iteration budget exhausted (200/200) ..."). The
0s was invented too: that shape carries no limit_seconds.
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
    assert "0s" not in detail  # the old code fabricated max_runtime=0s


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"elapsed_seconds": 5403, "limit_seconds": 5400}, "ran 5403s of 5400s"),
        (
            {"failures": 2, "effective_limit": 2, "limit_source": "dispatcher"},
            "attempt 2 of 2 (dispatcher limit)",
        ),
        ({"exit_kind": "nonzero_exit", "exit_code": 1, "pid": 18342}, "exited with code 1"),
        ({"exit_kind": "signaled", "exit_code": 9, "pid": 18342}, "killed by signal 9"),
        ({"budget_used": 200, "budget_max": 200}, "budget 200/200"),
        ({"retry_status": "ready"}, "will retry"),
        ({"retry_status": "gave_up"}, "not retrying (gave_up)"),
    ],
    ids=["runtime", "attempts", "crash-exit", "crash-signal", "budget", "retrying", "finished"],
)
def test_each_fact_the_payload_carries_is_reported(payload: dict, expected: str) -> None:
    assert expected in _failure_detail(payload)


def test_gave_up_is_terminal_even_when_the_payload_says_ready() -> None:
    detail = _failure_detail({"retry_status": "ready"}, terminal=True)

    assert "not retrying (blocked)" in detail
    assert "will retry" not in detail


def test_a_field_that_is_absent_is_not_mentioned() -> None:
    detail = _failure_detail({"retry_status": "ready"})

    assert _failure_detail(None) == _failure_detail({}) == ""
    assert detail.startswith("\n")  # appended to a headline, must not run on
    assert "attempt" not in detail
    assert "ran " not in detail
    assert "budget" not in detail


def test_a_pathological_error_is_clipped_visibly() -> None:
    """One runaway traceback must not flood the channel."""
    detail = _failure_detail({"error": "x" * (_NOTIFY_DETAIL_MAX + 500)})

    assert "more chars; see board" in detail
