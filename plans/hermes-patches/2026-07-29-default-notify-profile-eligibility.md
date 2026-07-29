# Default-notify owner-profile eligibility

- Task: t_bdf2f99f
- Base: PR #176 (wt/t_8b08b46a)
- Scope: default-notify auto-subscription eligibility only.
- Fix: use the stamped notifier profile adapter resolution for both the zero-subscription gate and row insertion.
- Regression: default Signal plus secondary Discord must not create a default-owned Discord subscription.
