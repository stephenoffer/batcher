# Running on unstable nodes

This page describes how Batcher keeps a job alive on a GPU cluster whose nodes and devices
fail underneath it, and what to configure when the defaults don't suit your fleet.

At fleet scale a node rarely fails by disappearing. It stays up and is wrong: a GPU reporting uncorrectable memory errors, a spill directory remounted read-only, a driver that no longer matches its CUDA runtime. The scheduler still sees a healthy node with a free slot, so one bad machine can walk the whole queue onto itself.

## What Batcher already does

Four mechanisms run by default:

- **Driver errors.** Batcher reads the driver's Xid log, the only report of a double-bit ECC fault or a GPU that fell off the bus. A device with a recent fatal Xid stops being scheduled, and the log line names the repair it needs.
- **Kernel faults.** Batcher reads the kernel log for faults that aren't about the GPU, such as a worker killed by the out-of-memory killer.
- **Outcomes.** A node that failed its last several tasks stops being scheduled whatever its telemetry says, and returns once it proves itself on real work.
- **Retry budget.** A bounded share of a job may be spent retrying, so a broken fleet fails fast with the first real error.

The defaults are visible on the config:

```python
from batcher import Config

ft = Config().fault_tolerance
print(ft.quarantine.failure_threshold, ft.quarantine.half_life_s, ft.fail_on_untrusted_results)
# 3.0 300.0 True
```

:::{dropdown} Why every fault signal expires
The kernel ring buffer holds a node's history, not its present, so a fatal Xid from before the last device reset is still in it. A quarantine keyed on "the buffer contains a fatal code" would never release a repaired device, and the fleet would shrink with nothing in any log to explain it. So every signal is windowed: recorded failures decay over a half-life, and a quarantine expires into probation. Only a success clears one.
:::

## Check the fleet before you trust it

{py:func}`bt.accelerator_problems() <batcher.accelerator_problems>` returns everything wrong with this node and its cluster as complete sentences, ready to paste into an alert. On a healthy node it returns an empty list:

```python
import batcher as bt

for problem in bt.accelerator_problems():
    print(problem)
```

An empty list can also mean nothing could be read, such as in a container without the host kernel log or on a node without `pynvml`. {py:func}`bt.accelerators() <batcher.accelerators>` tells the two apart by listing what was detected:

```python
inventory = bt.accelerators()
print(sorted(inventory)[:3])
# ['backend', 'devices', 'power']
```

## Set the collective timeout

A multi-GPU collective's default failure mode is to wait forever. When one rank dies, the others sit holding their GPUs, and no recovery mechanism runs because nothing has reported a failure.

Batcher sets `TORCH_NCCL_ASYNC_ERROR_HANDLING` and its older spelling on the GPU tasks it launches, which turns that hang into an ordinary task failure. It never overwrites a value you set. If you launch your own workers, set it there too.

## Tune the quarantine

The defaults suit a fleet where a node failing three tasks in five minutes is unusual. A large spot cluster wants a shorter half-life, so a node isn't held against its past. A fleet running hours-long stages wants a lower threshold, so a bad node is taken out after fewer losses.

```python
import dataclasses
from batcher import Config, set_config

base = Config()
quarantine = dataclasses.replace(
    base.fault_tolerance.quarantine,
    failure_threshold=2.0,
    half_life_s=120.0,
)
set_config(
    base.replace(
        fault_tolerance=dataclasses.replace(base.fault_tolerance, quarantine=quarantine),
    )
)
```

Don't raise `max_blocked_fraction` to fix a fleet that keeps failing. When every node fails every task, the cause is a credential, an image, or a model file, not the fleet.

## When a device corrupts rather than loses

A double-bit or uncontained ECC fault doesn't lose work. The device kept running and returned a wrong number, so every task that already succeeded on it is suspect. Batcher refuses to retry past such a fault and fails the run with a message saying so, because a retry would finish successfully and write the corruption out. Turn this off with `fault_tolerance.fail_on_untrusted_results` only where something downstream verifies the results.

Corruption is one of three verdicts Batcher reaches before it decides whether to retry at all, as the figure shows:

![How a distributed run recovers when a map or reduce task raised or its worker stopped answering. Batcher classifies the failure before retrying, and only one of three verdicts retries. Lost data, such as a Ray error that is not a task error because an actor, worker or node died, an unreachable peer, or a spill file on an ephemeral disk, is recomputed. A deterministic bug, such as a UDF exception, a bad cast, a schema mismatch or a broken runtime environment, is re-raised, because every retry would re-run it and burn the job-wide budget. Untrusted results, such as an uncontained ECC fault where the device kept running and answered wrongly, stop the run, because work already finished on that device is as suspect as the task that failed. A recompute then costs one of three things: re-reading the source partition and re-running the map by default, fetching an off-node replica when shuffle_replication is above 1, or migrating the data while the worker is still alive when there was advance notice such as spot metadata, SIGTERM or a Slurm deadline. Recovery brings its own hazard, a worker presumed dead that is not, so each round carries a higher epoch and a reducer discards any batch arriving under a stale one.](/_static/diagrams/fault_recovery.svg)

## Requirements and limitations

- The Xid and node-fault readers need a readable `/dev/kmsg`: `CAP_SYSLOG`, or a container sharing the host's kernel log.
- Device health needs `pynvml` on each worker (`pip install 'batcher-engine[nvml]'`), or the AMD equivalent.
- An unreadable source reports nothing rather than assuming the worst, so a fleet never drains because a base image changed.
- Quarantine is remembered across the stages of one job, not across separate jobs.

## See also

- {doc}`/configuration/fault-tolerance` for the field-by-field reference.
- {doc}`/user-guide/operate/running/gpu-fleets` for power budgets, fabric-aware placement, and device residency.
- {doc}`/user-guide/operate/running/gpu-diagnosis` for a GPU stage that is slow rather than failing.
- {doc}`/user-guide/operate/running/troubleshooting` for errors that are not the fleet's fault.
- {doc}`/architecture/fault-tolerance` for how recovery works underneath.
