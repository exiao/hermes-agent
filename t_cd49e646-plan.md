# PR #135 babysit plan

## Objective
Drive `exiao/hermes-agent#135` to merge-ready without merging it.

## Steps
1. Verify live open PR metadata, CI, all review/comment sources, and the dedicated worktree against the remote head.
2. Diagnose failed CI and every live review finding against the current implementation.
3. Add a focused failing regression test, implement the smallest on-branch fix, and run targeted verification.
4. Commit and push only explicit files; re-check live head, CI, reviews/threads, reply and resolve addressed threads.
5. Re-read this plan and complete only when CI is green, no live unresolved threads remain, and the PR diff is scoped.

## Acceptance criteria
- Required CI passes on the latest PR head.
- Every actionable review thread is resolved with evidence.
- Any code change is committed, pushed to the existing feature branch, and covered by focused verification.
- PR is handed off as merge-ready; no merge is performed.
