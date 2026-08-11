# PR #123 babysit plan

## Objective
Make `exiao/hermes-agent#123` merge-ready: investigate each failing CI slice and every live review/comment source, fix any verified in-scope blockers on the PR branch, and leave all threads resolved with green required CI.

## Steps
1. Inspect live PR metadata, CI failure logs, formal reviews, issue comments, and GraphQL review threads; re-check the PR head before edits.
2. Create a dedicated worktree under `~/projects/_worktrees/` at the exact PR head and read the relevant implementation/tests plus project guidance.
3. Reproduce each real CI failure with the repository test wrapper; trace and make minimal test-backed in-scope fixes where required.
4. Run focused verification, inspect the scoped diff, commit explicit paths, push the PR branch, then re-check CI and newly created review threads.
5. Reply to and resolve addressed threads. Confirm green checks, clean merge state/diff, and compare completion with this plan.

## Acceptance criteria
- Live PR remains open and is based on the expected head (or all checks are repeated after any head advance).
- No unresolved, current blocking review threads or comments remain.
- Required CI checks pass on the latest head; focused local tests demonstrate any code change.
- Branch diff is scoped to the verified blockers, pushed, and ready for Eric to merge.
