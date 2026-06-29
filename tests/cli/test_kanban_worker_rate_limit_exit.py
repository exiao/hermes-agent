"""Worker-side rate-limit exit-code mapping for kanban single-query runs.

Regression coverage for the bug where a proxy/provider rate-limit storm
silently blocked healthy kanban tasks: the dispatcher already had a carve-out
that releases a task back to ``ready`` WITHOUT counting a failure when the
worker exits with ``KANBAN_RATE_LIMIT_EXIT_CODE`` (EX_TEMPFAIL / 75) — but the
kanban worker is spawned with ``hermes chat -q`` (NON-quiet), and the exit-code
carve-out only lived in the ``--quiet`` branch, so a worker that died on a
transient 429 exited with a generic crash code instead. The dispatcher then
counted it as a ``consecutive_failure`` and tripped the circuit breaker.

These tests pin the decision logic (``_kanban_worker_rate_limit_exit_code``)
that both the quiet and non-quiet single-query paths now share:

* a worker run that failed on a transient throttle (``rate_limit`` /
  ``billing``) maps to the EX_TEMPFAIL sentinel — so it never burns the
  task's retry budget;
* a real defect crash (any other ``failure_reason``) does NOT map to the
  sentinel — so a genuine failure still counts and can trip the breaker;
* a non-worker run (no ``HERMES_KANBAN_TASK``) never maps to the sentinel.
"""

import pytest

from cli import _kanban_worker_rate_limit_exit_code
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_rate_limit_worker_maps_to_tempfail_sentinel(reason):
    """A kanban worker that died on a quota wall returns the EX_TEMPFAIL
    sentinel, NOT a generic crash code — so the dispatcher requeues it
    without burning the failure budget."""
    code = _kanban_worker_rate_limit_exit_code(reason, is_kanban_worker=True)
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE
    assert code == 75  # BSD EX_TEMPFAIL — the value the reap classifier maps.


@pytest.mark.parametrize(
    "reason",
    [
        None,                 # clean / non-failure
        "context_overflow",   # real defect class
        "invalid_tool_call",
        "auth",
        "unknown",
        "",
    ],
)
def test_real_defect_worker_does_not_map_to_sentinel(reason):
    """A genuine crash reason (anything that isn't a transient throttle) must
    NOT borrow the rate-limit carve-out — it stays None so the caller keeps
    its normal crash exit code and the breaker can still trip."""
    assert (
        _kanban_worker_rate_limit_exit_code(reason, is_kanban_worker=True)
        is None
    )


@pytest.mark.parametrize("reason", ["rate_limit", "billing", "auth", None])
def test_non_worker_run_never_maps_to_sentinel(reason):
    """The carve-out is scoped to kanban workers. A plain ``hermes chat -q``
    (no HERMES_KANBAN_TASK) keeps the normal 0/1 exit contract automation
    wrappers expect, even on a rate-limit failure."""
    assert (
        _kanban_worker_rate_limit_exit_code(reason, is_kanban_worker=False)
        is None
    )
