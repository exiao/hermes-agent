# PR #138 babysit plan

## Objective
Drive `exiao/hermes-agent#138` to merge-ready without merging it.

## Steps
1. Verify live PR state, CI, review bodies/comments/threads, and no duplicate babysitter card.
2. Create a dedicated detached worktree under `~/projects/_worktrees/` at the current PR head.
3. Diagnose each live CI/review blocker, reproduce it, and make the smallest on-branch fix with focused tests.
4. Commit explicit paths, push the PR branch, reply to and resolve addressed review threads.
5. Re-check current head, all review sources/threads, CI, diff scope, and mergeability.

## Acceptance criteria
- PR remains open and is not merged by this worker.
- All required/reported CI checks are green (or documented legitimate skips); no unresolved actionable review threads remain on latest head.
- Diff is scoped and locally verified; kanban completion records plan-vs-actual and evidence.
