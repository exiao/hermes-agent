# PR #132 babysit plan

## Objective
Drive `exiao/hermes-agent#132` to merge-ready without merging it: validate the live head, every review/comment source, CI, and scope; fix only verified blockers on the PR branch.

## Steps
1. Record live PR state and rule out a duplicate board card. (Done: PR is open at `38c2b9fca481`, CLEAN/MERGEABLE; no active sibling card.)
2. Inspect the PR diff, formal reviews, issue comments, inline review comments, and GraphQL review threads against the live head.
3. If an actionable finding exists, reproduce it with a focused test, make the minimal on-branch fix in the dedicated PR worktree, test, commit, push, reply, resolve, and re-query fresh threads.
4. Verify the latest head has passing required CI, no actionable/unresolved review threads, and a clean scoped diff.
5. Re-read this plan, record plan-vs-actual, and complete the Kanban card with evidence.

## Acceptance criteria
- PR remains open and is not merged by this worker.
- Required CI is green at the live head.
- All review/comment sources have no actionable unresolved finding.
- Any necessary change is tested, committed to the PR branch, pushed, and its review thread resolved.
