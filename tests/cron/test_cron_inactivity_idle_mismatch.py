"""Regression: the cron inactivity watchdog must report the idle time it
actually tripped on, and must not trip on a transient/error idle reading.

Live failures on 2026-09-01 and 2026-09-03 raised:

    TimeoutError: Cron job 'kanban-blocked-triage-escalator' idle for 2s
    (limit 600s) - last activity: waiting for non-streaming API response

"idle for 2s (limit 600s)" is self-contradictory: the watchdog only trips
when ``idle >= limit``, so a kill that reports 2s cannot have been caused
by the idle reading it printed.

Two distinct defects are covered here:

1. RE-READ RACE (the wrong number). ``_inactivity_watchdog_loop`` decides
   on the idle value it sampled, then throws it away and returns a bare
   bool. The error message is built from a SECOND, later
   ``get_activity_summary()`` call in ``run_job``. If the agent resumes
   between the two reads, the message reports the fresh low idle value
   instead of the one that caused the kill. The tripping value must be
   carried out of the loop.

2. TRANSIENT-READ TRIP (the wrong kill). ``get_idle_seconds`` swallows
   exceptions and returns 0.0, so a single bogus spike trips the kill on
   one sample. The watchdog must observe the breach on consecutive polls
   before declaring the job dead.
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import _inactivity_watchdog_loop  # noqa: E402


def _run(readings, limit_s=600.0):
    """Drive the loop over a fixed list of idle readings."""
    stop = threading.Event()
    seq = list(readings)
    calls = {"n": 0}

    def get_idle():
        calls["n"] += 1
        if seq:
            return seq.pop(0)
        stop.set()
        return 0.0

    return _inactivity_watchdog_loop(
        get_idle_seconds=get_idle,
        limit_s=limit_s,
        poll_s=0.001,
        stop=stop,
        future_done=lambda: False,
    ), calls


def test_watchdog_reports_the_idle_value_it_tripped_on():
    """Defect 1: the tripping idle value must survive out of the loop.

    A bare ``True`` forces the caller to re-read the clock, which is what
    produced "idle for 2s (limit 600s)" in production.
    """
    result, _ = _run([10.0, 900.0, 900.0], limit_s=600.0)
    assert result, "watchdog should trip when idle exceeds the limit"
    tripped_idle = getattr(result, "idle_seconds", None)
    assert tripped_idle is not None, (
        "watchdog returned a bare bool, so the caller must re-read the "
        "activity clock and can report an idle value that never tripped it"
    )
    assert tripped_idle >= 600.0, (
        "reported idle (%r) must be the value that breached the limit"
        % (tripped_idle,)
    )


def test_watchdog_does_not_trip_on_a_single_transient_reading():
    """Defect 2: one bogus spike must not kill a healthy job."""
    result, _ = _run([10.0, 900.0, 10.0, 10.0], limit_s=600.0)
    assert not result, (
        "a single over-limit sample surrounded by healthy readings must "
        "not kill the job; the breach has to persist across polls"
    )


def test_watchdog_still_trips_on_a_sustained_stall():
    """The real stall this watchdog exists for must still be caught."""
    result, _ = _run([700.0, 800.0, 900.0], limit_s=600.0)
    assert result, "a sustained over-limit stall must still trip the watchdog"
    assert getattr(result, "idle_seconds", 0.0) >= 600.0


def test_watchdog_returns_false_when_future_completes():
    """Unchanged behaviour: a finished job is not a stalled job."""
    stop = threading.Event()
    result = _inactivity_watchdog_loop(
        get_idle_seconds=lambda: 9999.0,
        limit_s=600.0,
        poll_s=0.001,
        stop=stop,
        future_done=lambda: True,
    )
    assert not result
