# PR #130 babysit plan

## Objective
Drive exiao/hermes-agent#130 to merge-ready without merging it.

## Steps
1. Confirm the PR is live/open and no duplicate pr-babysitter card owns it.
2. Create a dedicated PR-head worktree under ~/projects/_worktrees/, then inspect the PR diff, all review sources, and CI.
3. Reproduce and minimally fix any live blocker with a focused regression test; commit explicit paths and push only the feature branch.
4. Re-check the remote head, CI, all review threads/comments/reviews, resolve addressed threads, and verify the final diff is scoped.
5. Complete the Kanban card only when checks are green and no actionable unresolved review thread remains; otherwise hand off only a genuine human decision.

## Acceptance criteria
- PR remains open and unmerged by this worker.
- All reported checks pass on the current head.
- Every actionable review thread is resolved with evidence.
- Diff is scoped to the Signal ambiguous-send retry issue.
