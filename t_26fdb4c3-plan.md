# PR #120 babysit plan

## Objective
Drive exiao/hermes-agent PR #120 (`fix/execute-code-guard-eager-redact`) to merge-ready without merging it.

## Steps
1. Confirm the PR is open and that no other active babysitter card owns this PR.
2. Inspect all live CI failures, formal reviews, issue comments, and unresolved review threads at the current head.
3. Create/use a dedicated detached PR worktree under `~/projects/_worktrees`, identify and reproduce any code-caused failure, and make the narrow regression-tested fix.
4. Commit only explicit paths, push only to the PR branch, then recheck CI and review threads at the new live head.
5. Resolve addressed threads, verify all required checks are green and the PR is mergeable, then complete the Kanban task.

## Acceptance criteria
- PR remains open; no duplicate babysitter owns it.
- Current-head CI has no failed or pending required checks.
- All actionable review threads are resolved and formal-review bodies have been read.
- Any fix is committed/pushed to the PR branch with targeted verification.
- No merge/deploy is performed.
