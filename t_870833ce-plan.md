# PR #137 babysit plan

## Objective
Make `exiao/hermes-agent#137` merge-ready without merging it: fix live review/CI blockers on the PR branch, verify behavior, resolve addressed threads, and hand off as done.

## Steps
1. Confirm the live PR head, branch, CI failures, review bodies/comments/threads, and use a dedicated isolated worktree at that head.
2. Reproduce each real blocker with a focused regression test, then implement the minimal in-scope fix.
3. Run focused tests and the relevant runtime-path verification; inspect the final diff.
4. Commit explicit paths and push the PR branch; re-check the new live head, CI, review threads, and any fresh bot feedback.
5. Reply to and resolve only addressed threads; complete the Kanban card when all required checks are green and no actionable threads remain.

## Acceptance criteria
- Latest PR head contains the persistence-failure and `/btw` session-override fixes, with regression coverage.
- Relevant tests and runtime-facing gateway path pass.
- Required CI is green on the latest head; every actionable review thread is resolved.
- No merge is performed; completion includes PR URL, changed files, verification, and plan-vs-actual status.
