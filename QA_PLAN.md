# QA plan: hermes-agent PR #136

## Objective
Verify PR #136 on its actual head: extracted gateway slash-command handling, Kanban lifecycle heartbeat guidance, and the associated cache-budget behavior.

## Steps
1. Resolve the PR head and prepare an isolated verification checkout/worktree without changing unrelated branches.
2. Read the affected implementation and existing focused tests; run the relevant suites.
3. Boot and drive a real picker-capable gateway adapter/runtime path for bare `/reasoning` and `/fast`, observing that `GatewaySlashCommandsMixin` handles each command.
4. Construct an `AIAgent` prompt under a temporary `HERMES_HOME` and task context; verify lifecycle guidance includes the hourly heartbeat requirement and run the prompt cache-budget test.
5. Investigate and fix any reproducible defect on the PR branch if necessary, then repeat verification from a clean boot.
6. Re-read this plan and report plan-vs-actual plus rubric evidence to the Kanban card.

## Acceptance criteria
- Actual PR head is tested, not a stale/default branch.
- Bare gateway slash commands reach extracted mixin behavior.
- Worker prompt includes hourly heartbeat lifecycle guidance and budget test passes.
- Focused tests pass with named output; residual risk is stated.
