# Fault-tolerance options

This page documents the `fault_tolerance` configuration section: what Batcher does when nodes and devices fail underneath a running job.

Both mechanisms are on by default and sized so a healthy fleet never notices them. {doc}`/user-guide/operate/running/unstable-nodes` is the task-oriented walkthrough, and this page is the field reference.

```python
from batcher import Config

ft = Config().fault_tolerance
print(ft.retry_budget_fraction, ft.retry_budget_floor, ft.quarantine.failure_threshold)
# 0.1 16 3.0
```

Tighten the retry budget by deriving a new config:

```python
import dataclasses

base = Config()
cfg = base.replace(
    fault_tolerance=dataclasses.replace(base.fault_tolerance, retry_budget_fraction=0.05),
)
print(cfg.fault_tolerance.retry_budget_fraction)
# 0.05
```

Or change one field by its dotted name, for the length of a block:

```python
from batcher.config import get_option, option_context

with option_context("fault_tolerance.quarantine.failure_threshold", 5.0):
    print(get_option("fault_tolerance.quarantine.failure_threshold"))
# 5.0
```

## Top level

| Field | Default | Meaning |
|-------|---------|---------|
| `retry_budget_fraction` | `0.1` | Share of attempted work a job may spend on retries before the next failure is raised instead of retried. `0.0` disables the budget. |
| `retry_budget_floor` | `16` | Retries authorized regardless of job size, so a short job is not failed by one flaky node. |
| `fail_on_untrusted_results` | `True` | Fail the run when a device reports a fault that corrupts data already computed on it, rather than retrying past it. |

These fields are the {py:class}`FaultToleranceConfig <batcher.config.FaultToleranceConfig>` dataclass, with the nested quarantine section below.

A per-task retry limit bounds a task, not a job. The budget caps retries across the whole job, so a fleet broken in a way no probe catches fails fast with the first clear error instead of retrying for hours.

`fail_on_untrusted_results` guards against a double-bit or uncontained ECC fault, where the device kept running and returned a wrong number. Retrying past one would write the corruption out, so the run fails instead.

## Quarantine

Which nodes and devices stop being scheduled, learned from task outcomes rather than telemetry. This catches the failures hardware can't report, such as a mismatched driver, a half-deployed container image, or a disk returning `EIO`.

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Learn which targets are bad from task outcomes. |
| `failure_threshold` | `3.0` | Decayed failure weight at which a target stops being scheduled. |
| `half_life_s` | `300.0` | How long a recorded failure keeps half its weight. |
| `cooldown_s` | `60.0` | How long a first quarantine lasts before the target goes back on probation. |
| `max_cooldown_s` | `900.0` | Ceiling on the per-offense doubling of the cooldown. |
| `max_blocked_fraction` | `0.34` | Share of the fleet that may be quarantined at once. |

These fields are the {py:class}`QuarantineConfig <batcher.config.QuarantineConfig>` dataclass.

`max_blocked_fraction` is the safety valve. When every node fails every task, such as with an expired credential, the cap stops the ledger from condemning the whole cluster, and the run reports that the failures have gone systemic.

::::{dropdown} How failures are weighted
Failures are weighted by cause. A failure that blames the placement, such as a device fault or a filesystem error, counts fully. One that doesn't, such as an accelerator running out of memory or a throttled model endpoint, counts nothing, because the next node the retry lands on would hit it too.

The cause is read from the exception's type and message, walking the cause chain, because the real error usually arrives wrapped by an SDK, an HTTP client, or Ray. Object-store throttling is matched by code as well as by phrase: `SlowDown` on S3, `RateLimitExceeded` on Google Cloud Storage, `ServerBusy` and `TooManyRequests` on Azure. An unrecognized failure is treated as the workload's own and is not retried, so a deterministic bug doesn't spend the recovery budget across the fleet.
::::

## See also

- {doc}`/user-guide/operate/running/unstable-nodes` for the task-oriented walkthrough.
- {doc}`/user-guide/operate/running/gpu-fleets` for power budgets, placement, and device health.
- {doc}`accelerator` for the device-health thresholds these build on.
- {doc}`/architecture/fault-tolerance` for how recovery works underneath.
