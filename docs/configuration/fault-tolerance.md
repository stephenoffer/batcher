# Fault-tolerance options

This page documents the `fault_tolerance` configuration section: what Batcher does when nodes
and devices fail underneath a running job. The rest of the `distributed` section tunes a
cluster that mostly works; this one tunes what happens when it doesn't.

Every default is chosen so that turning nothing on changes nothing. The quarantine thresholds
are permissive enough that a healthy fleet never trips them, and the retry budget is generous
enough that only a systematically broken run exhausts it.
{doc}`/user-guide/operate/running/unstable-nodes` is the task-oriented walkthrough; this page is the
field reference.

```python
import dataclasses
from batcher import Config

base = Config()
cfg = base.replace(
    fault_tolerance=dataclasses.replace(base.fault_tolerance, retry_budget_fraction=0.05),
)
print(cfg.fault_tolerance.retry_budget_fraction)
# 0.05
```

## Top level

| Field | Default | Meaning |
|-------|---------|---------|
| `retry_budget_fraction` | `0.1` | Share of attempted work a job may spend on retries before the next failure is raised instead of retried. `0.0` disables the budget. |
| `retry_budget_floor` | `16` | Retries authorized regardless of job size, so a short job is not failed by one flaky node. |
| `fail_on_untrusted_results` | `True` | Fail the run when a device reports a fault that corrupts data already computed on it, rather than retrying past it. |

These fields are the
{py:class}`FaultToleranceConfig <batcher.config.FaultToleranceConfig>` dataclass, with the
nested quarantine section below.

A per-task retry limit bounds a task and bounds nothing about a job. `task_max_retries=2` over
a hundred thousand partitions authorizes two hundred thousand retries, and a fleet broken in
some way no probe catches will use every one of them. What you see then is a run that takes
hours at a fraction of its rate and fails with whatever error happened to be last, long after
the first one said exactly what was wrong. The budget is what turns that into a bounded loss.

`fail_on_untrusted_results` is not a performance trade. Almost every failure loses work, and
losing work is what a retry is for; a double-bit or uncontained ECC fault does something else,
because the device kept running and returned a wrong number. Retrying past one produces a job
that completes successfully and writes out the corruption.

## Quarantine

Which nodes and devices stop being scheduled, learned from task outcomes rather than from
telemetry. Telemetry catches the failures hardware knows how to report; these thresholds catch
the ones it doesn't, such as a driver that no longer matches its runtime, a half-deployed
container image, or a disk that has started returning `EIO`.

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | Learn which targets are bad from task outcomes. |
| `failure_threshold` | `3.0` | Decayed failure weight at which a target stops being scheduled. |
| `half_life_s` | `300.0` | How long a recorded failure keeps half its weight. |
| `cooldown_s` | `60.0` | How long a first quarantine lasts before the target goes back on probation. |
| `max_cooldown_s` | `900.0` | Ceiling on the per-offense doubling of the cooldown. |
| `max_blocked_fraction` | `0.34` | Share of the fleet that may be quarantined at once. |

These fields are the {py:class}`QuarantineConfig <batcher.config.QuarantineConfig>` dataclass.

`max_blocked_fraction` is the safety valve on the whole mechanism. When the cause is systemic,
such as an expired credential or a model file that returns 404, every node fails every task.
Without a cap the ledger condemns the entire cluster in the first minute and turns a degraded
job into a dead one. Past the cap only the worst offenders stay quarantined, and the run
reports that the failures have gone systemic.

Failures are weighted by cause. A failure that blames the placement, such as a device fault or
a filesystem error, counts fully. One that doesn't, such as an accelerator running out of
memory or a throttled model endpoint, counts nothing at all: quarantining a node over the
workload's own behavior would take out the next node the retry lands on too.

## See also

- {doc}`/user-guide/operate/running/unstable-nodes` for the task-oriented walkthrough.
- {doc}`/user-guide/operate/running/gpu-fleets` for power budgets, placement, and device health.
- {doc}`accelerator` for the device-health thresholds these build on.
- {doc}`/architecture/fault-tolerance` for how recovery works underneath.
