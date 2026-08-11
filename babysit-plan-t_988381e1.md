# PR #126 babysit plan

## Objective
Drive `exiao/hermes-agent#126` (`fix/tirith-lookalike-strip`) to merge-ready without merging it.

## Steps
1. Verify the live PR head, terminal state, duplicate ownership, and all review/comment sources.
2. Create a dedicated PR worktree at the live head; inspect the changed code and reproduce any live finding.
3. Make the minimal on-branch fix with focused regression coverage if required; run the relevant test path and inspect the diff.
4. Commit only explicit paths, push the feature branch, then re-check all threads/reviews and latest-head CI.
5. Resolve/reply to addressed threads and complete only when CI is green and no blocking unresolved finding remains.

## Acceptance criteria
- PR is open, clean, and mergeable on its latest head.
- Every live blocking review finding is fixed or demonstrably stale and resolved with a concise reply.
- Required CI passes and focused verification is recorded.
- No merge, deploy, or protected-branch push is performed.
