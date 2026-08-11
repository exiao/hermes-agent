# PR #140 babysit plan

## Objective
Drive `exiao/hermes-agent#140` to merge-ready without merging it.

## Steps
1. Verify live PR state, CI, duplicate board ownership, all review sources, and current diff.
2. Use an isolated detached worktree at the live PR head; reproduce every live finding with a focused test or runtime path.
3. For in-scope correctness findings, add a regression test first, implement the smallest fix, run focused and relevant verification, commit, and push to the PR branch.
4. Re-query the new-head review threads and checks; reply to and resolve each addressed thread.
5. Re-read this plan, confirm all steps are complete or explicitly cut, then hand off with merge-ready evidence.

## Acceptance criteria
PR is open, mergeable, CI has no failed or pending required checks, all actionable review threads are resolved on the latest head, and the diff remains scoped to Signal message chunking.