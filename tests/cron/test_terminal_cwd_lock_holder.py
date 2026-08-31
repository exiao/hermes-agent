"""The TERMINAL_CWD lock error must name the real holder (#79768 follow-up).

The original message asserted "another cron job (a workdir writer, or
long-running readers) has held it" and told the reader to stagger schedules or
drop a workdir. When the holder has DIED without releasing — a gateway restart
or a killed run mid-job — that text names a cause that cannot be true (there may
be no workdir job configured at all) and sends the reader to a fix that changes
nothing. These tests pin the distinction.
"""

import threading


def _lock():
    import cron.scheduler as sched

    return sched._ReadWriteLock()


def test_holder_description_names_an_uncontended_lock_as_unheld():
    """No holder must read as NO job, not as a workdir writer."""
    assert _lock().holder_description() == "NO job holds it"


def test_holder_description_names_the_writer_job():
    """A real writer is named, so the stagger/remove-workdir advice applies."""
    lock = _lock()
    assert lock.acquire_write(timeout=5, job="nightly-report")
    try:
        assert "nightly-report" in lock.holder_description()
    finally:
        lock.release_write()


def test_holder_description_reports_readers_when_readers_hold_it():
    lock = _lock()
    assert lock.acquire_read(timeout=5)
    try:
        assert "workdir-less job(s) still hold it" in lock.holder_description()
    finally:
        lock.release_read()


def test_release_write_clears_the_named_holder():
    """A finished writer must not keep being blamed by later waiters."""
    lock = _lock()
    assert lock.acquire_write(timeout=5, job="nightly-report")
    lock.release_write()
    assert lock.holder_description() == "NO job holds it"
    assert "nightly-report" not in lock.holder_description()


def test_writer_job_is_optional():
    """acquire_write stays usable without a job name (existing callers)."""
    lock = _lock()
    assert lock.acquire_write(timeout=5)
    try:
        assert "unnamed workdir job" in lock.holder_description()
    finally:
        lock.release_write()


def test_a_waiter_that_times_out_against_a_dead_holder_sees_no_holder():
    """The exact production case: the holder vanished without releasing.

    A writer that never releases (its thread was killed) leaves the lock held,
    so a reader times out. The description must NOT claim a live workdir job is
    responsible, because that is the misdiagnosis this change exists to stop.
    """
    lock = _lock()
    started = threading.Event()

    def dead_writer():
        lock.acquire_write(timeout=5, job="killed-job")
        started.set()
        # Never releases: models a run terminated mid-job.

    t = threading.Thread(target=dead_writer, daemon=True)
    t.start()
    assert started.wait(timeout=5)

    assert lock.acquire_read(timeout=0.2) is False
    described = lock.holder_description()
    assert "killed-job" in described
    # The advice must stay truthful: a named holder means the workdir advice
    # is at least addressed at something real.
    assert "NO job holds it" not in described


def test_timeout_holder_is_captured_before_the_blocker_releases():
    """The description must come from the moment of the timeout, not after.

    Codex's P2 on #261: `holder_description()` retakes the condition lock
    separately, so a blocker that releases in that window reads back as
    "NO job holds it" — the message then blames a dead run when a live
    workdir job was genuinely holding the lock. Capture at timeout instead.
    """
    lock = _lock()
    released = threading.Event()

    def blocker():
        assert lock.acquire_write(timeout=5, job="live-job")
        released.wait(timeout=5)
        lock.release_write()

    t = threading.Thread(target=blocker, daemon=True)
    t.start()

    while True:
        with lock._cond:
            if lock._writer_active:
                break

    assert lock.acquire_read(timeout=0.2) is False
    # The blocker releases DURING the window the old code described in.
    released.set()
    t.join(timeout=5)

    assert "live-job" in lock.last_timeout_holder()
    assert lock.holder_description() == "NO job holds it"


def test_write_timeout_describes_readers_not_the_timing_out_writer():
    """A write waiter must not diagnose itself as queued ahead."""
    lock = _lock()
    assert lock.acquire_read(timeout=5)
    try:
        assert lock.acquire_write(timeout=0.01, job="timed-out-writer") is False
        description = lock.last_timeout_holder()
    finally:
        lock.release_read()

    assert "workdir-less job(s) still hold it" in description
    assert "queued ahead" not in description


def test_timeout_holder_is_scoped_to_the_waiting_thread():
    """Concurrent timeout descriptions must not overwrite one another."""
    lock = _lock()
    assert lock.acquire_read(timeout=5)
    started = threading.Barrier(3)
    finished = threading.Barrier(3)
    descriptions = {}
    describe = lock._describe_locked

    def describe_with_waiter(**kwargs):
        return f"{threading.current_thread().name}: {describe(**kwargs)}"

    lock._describe_locked = describe_with_waiter

    def waiter(name):
        started.wait(timeout=5)
        assert lock.acquire_write(timeout=0.05, job=name) is False
        finished.wait(timeout=5)
        descriptions[name] = lock.last_timeout_holder()

    threads = [
        threading.Thread(target=waiter, args=(name,), name=name)
        for name in ("one", "two")
    ]
    for thread in threads:
        thread.start()
    started.wait(timeout=5)
    finished.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    lock.release_read()

    assert set(descriptions) == {"one", "two"}
    assert descriptions["one"].startswith("one: ")
    assert descriptions["two"].startswith("two: ")
