# Default-notify zero-subscription gate

Branch: `wt/t_8b08b46a` off `live-config` at `6c35913d86293c96694f8f0f9070bb74ec2788b2`.

The notifier retains its read-only zero-subscription fast path unless a configured default-notify target has a connected platform adapter. That exact condition permits the existing auto-subscribe block to writable-open the board and create its first subscription. No notifier-loop refactor or unrelated delivery behavior changes.

Validation: the existing live default-notify integration test is red on the base and green after the guard change; the three zero-sub gate tests stay green. The named-profile notifier test already sends exactly one delivery on the base and only has a stale message-text assertion failure, so it is deliberately outside this patch.
