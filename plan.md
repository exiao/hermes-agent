# PR #133 babysit plan

## Objective
Drive `exiao/hermes-agent#133` to merge-ready by resolving live blocking review findings and restoring green CI on the current feature-branch head.

## Steps
1. Verify live PR state, all review sources, CI failure logs, and unresolved review threads against the head SHA.
2. Use an isolated worktree under `~/projects/_worktrees/` at the verified PR head; trace the send queue and timeout implementation plus existing tests.
3. Reproduce the timeout-before-admission failure with a narrow regression test, implement the smallest queue-admission timeout fix, and assess the reconnect cache finding for scope/correctness.
4. Run focused verification, inspect the diff, commit explicit paths, and push to the PR branch.
5. Re-query all review sources and CI on the new head; reply to and resolve addressed threads. Complete only when checks are green and no actionable threads remain.

## Acceptance criteria
- Timeout begins only after queued work is admitted, with regression coverage.
- Any in-scope cache lifecycle defect is fixed on this branch with coverage.
- Latest PR head has passing required checks and zero unresolved actionable review threads.
