# QA plan: live-config CI repair PR #138

## Objective
Independently verify PR #138 from its remote head in this fresh worktree: GatewayRunner command routing, session-scoped `/fast`, choice-picker behavior, fail-closed unknown extension `MEDIA:` delivery, and the specified five-module wrapper regression suite. Do not merge.

## Steps
1. Fetch and inspect the live PR metadata/head and exact diff against `live-config`; check repository state and test entry points.
2. Map the changed runtime surfaces and test fixtures, then boot real gateway-side routing probes that observe `GatewaySlashCommandsMixin` execution for `/fast`, `/reasoning`, session-scoped `/fast`, and choice-picker interaction.
3. Drive the outbound unknown-extension `MEDIA:` path with both a valid safe file and an invalid/unvalidated path, observing delivery only after validation.
4. Run the named five-module `scripts/run_tests.sh --files` wrapper suite under CI-like settings; inspect any failure per root-cause discipline.
5. Re-read this plan, compare all steps against actual evidence, and report the five-axis QA rubric with residual risks. No merge or code change unless a real regression is found and repair is necessary.

## Acceptance criteria
- The checked-out commit is PR #138's current remote head, based on `live-config`.
- Runtime probes demonstrate the intended mixin path and fail-closed media validation behavior.
- The targeted modules pass via the project wrapper with named output.
- Completion records boot, intended path, tests, UI-capture applicability, and residual risk; plan-vs-actual is explicit.
