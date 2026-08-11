# PR #133 babysit plan

Objective: make `exiao/hermes-agent#133` merge-ready without merging it.

1. Verify the live PR is open, identify the exact head, review threads, formal/issue comments, and failed CI evidence.
2. Use a dedicated worktree under `~/projects/_worktrees/` aligned to the live feature head; inspect the WhatsApp bridge and its existing tests.
3. Reproduce each live in-scope review finding with focused tests, then implement the narrowest fix and regression coverage.
4. Run focused runtime/test verification, review the diff, commit explicit paths, and push only the PR branch.
5. Re-check latest-head CI and every comment source; reply and resolve addressed threads. Complete only when green and clean, otherwise make a bounded, evidence-based handoff.

Acceptance criteria: no unresolved actionable review threads; focused regression tests demonstrate both race protections; CI has no failures on the final head; plan steps reconciled in the completion handoff.
