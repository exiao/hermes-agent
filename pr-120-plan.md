# PR #120 babysit plan

## Objective
Drive exiao/hermes-agent#120 to merge-ready by diagnosing its three failed CI slices, making only necessary on-branch fixes, and resolving all addressed review threads.

## Steps
1. Verify live PR metadata, sibling-card uniqueness, CI failures, reviews, comments, and GraphQL review threads.
2. Create a dedicated worktree at the live PR head and inspect the failed-job logs and implicated tests.
3. Reproduce any real failure with the project test wrapper; make the narrowest test-backed fix if the PR caused it.
4. Run focused verification, inspect the diff, commit explicit paths, and push the PR branch.
5. Recheck CI and fresh review threads; reply/resolve addressed threads. Complete only with green checks and no unresolved actionable threads.

## Acceptance criteria
All required CI checks pass on the current PR head; no unresolved actionable review thread remains; diff stays scoped to the PR/CI blocker; completion records PR URL, changed files, and verification evidence.
