# Running on unstable nodes

This page describes how Batcher keeps a job alive on a GPU cluster whose nodes and devices
fail underneath it, and what you configure when the defaults are not what your fleet needs.

At fleet scale a node rarely fails by disappearing. It fails by staying up and being wrong: a
GPU that reports uncorrectable memory errors and keeps accepting work, a filesystem remounted
read-only under the spill directory, a driver that no longer matches its CUDA runtime, a
container image that half-deployed. In every one of those cases the scheduler still sees a
healthy node with a free slot, so it keeps placing work there. One bad machine then walks the
whole queue onto itself, and what you observe is not "a node was broken" but "the job never
finished".

## What Batcher already does

Four mechanisms run by default and need no configuration.

Batcher reads the driver's own error log. An Xid error is the only report of a double-bit ECC
fault, a GPU that has fallen off the bus, or a device whose micro-controller has halted. NVML
has no counter for any of them and `nvidia-smi` shows none of them. A device with a recent
fatal Xid stops being scheduled, and the log line names the repair the device needs rather
than only the fault it reported.

Batcher reads the kernel log for the faults that are not about the GPU at all. A worker killed
by the kernel's out-of-memory killer never raises, never unwinds, and never logs, so from the
orchestrator's side it is indistinguishable from a preemption or a network partition. The
kernel wrote down exactly what happened, in the one place nothing else reads.

Batcher learns from outcomes, not only from readings. A node that has failed the last several
tasks placed on it stops being scheduled whatever its telemetry says, and starts being
scheduled again once it proves itself on real work.

Batcher bounds how much of a job may be spent retrying, so a systematically broken fleet fails
quickly with the first real error instead of slowly with the last one.

## Evidence expires

Every fault signal here is windowed, and that is deliberate. The kernel ring buffer holds a
node's history, not its present, so a fatal Xid from before the last device reset is still
sitting in it. A quarantine keyed on "the buffer contains a fatal code" never releases a
device that has since been repaired, and a fleet then shrinks over its lifetime with nothing
in any log to explain it.

The same applies to the outcome ledger. Recorded failures decay over a half-life, a quarantine
expires into probation rather than into a permanent verdict, and a success is what clears one.

## Check the fleet before you trust it

`bt.accelerator_problems()` returns everything wrong with this node and its cluster as a list
of complete sentences, so a failing deployment check can be pasted into an alert without a
lookup table.

```python
import batcher as bt

for problem in bt.accelerator_problems():
    print(problem)
```

On a healthy node it prints nothing. On a fleet with something wrong it names the device or
the node, the condition, and what it costs, including the node-level faults that no GPU probe
can see and the repair each condemned device needs.

An empty list is not the same as a healthy fleet. Inside a container without the host kernel
log, and on a node without `pynvml`, nothing can be read and nothing is reported. Use
`bt.accelerators()` when you need to tell "nothing is wrong" from "nothing could be checked".

## Set the collective timeout

If your pipeline runs a multi-GPU collective, this is the highest-impact stability setting on
the cluster and it is one environment variable.

A collective's default failure mode is to wait forever. When one rank dies or one device
faults, the surviving ranks do not raise; they sit in the collective holding their GPUs until
something outside kills them. From the orchestrator's side nothing has failed at all: the task
is running, the actor is alive, and no progress is being made. Every recovery mechanism on this
page is downstream of a failure being reported, so none of them ever runs.

Batcher sets `TORCH_NCCL_ASYNC_ERROR_HANDLING` and its older spelling on the GPU tasks it
launches, which turns that hang into an ordinary task failure. It never overwrites a value you
set yourself. If you launch your own workers, set it there too.

## Tune the quarantine

The defaults suit a fleet where a node failing three tasks in five minutes is unusual. Two
situations call for a change.

A fleet with a high background failure rate, such as a large spot cluster, wants a shorter
half-life so a node is not held against its own past. A fleet where a failure is expensive,
such as one running hours-long inference stages, wants a lower threshold so a bad node is
taken out after fewer losses.

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

Do not raise `max_blocked_fraction` to solve a fleet that keeps failing. When every node fails
every task the cause is almost never the fleet; it is a credential, an image, or a model file,
and condemning more nodes replaces an error message with an outage.

## When a device corrupts rather than loses

Almost every failure loses work, and losing work is what a retry is for. A double-bit ECC
error and an uncontained ECC fault do something else: the device kept running and returned a
number, and the number is wrong. The tasks that already succeeded on that device are as
suspect as the one that failed.

Batcher refuses to retry past those, and fails the run with a message saying so. That is on by
default and it is not a performance trade: a job that retries past a corrupting fault finishes
successfully and writes the corruption out, which is worse than the crash it avoided. Turn it
off with `fault_tolerance.fail_on_untrusted_results` only where something downstream verifies
the results independently.

## Requirements and limitations

The Xid and node-fault readers need a readable `/dev/kmsg`, which means `CAP_SYSLOG` or a
container that shares the host's kernel log. Device health needs `pynvml` on each worker, or
the AMD equivalent. Where a source cannot be read, Batcher reports nothing rather than
assuming the worst, so a fleet never drains because a base image changed.

Quarantine is keyed on the worker's placement within a fleet, so it is remembered across the
stages of one job and not across separate jobs.

## See also

- {doc}`/configuration/fault-tolerance` for the field-by-field reference.
- {doc}`gpu-fleets` for power budgets, fabric-aware placement, and device residency.
- {doc}`gpu-diagnosis` for a GPU stage that is slow rather than failing.
- {doc}`troubleshooting` for errors that are not the fleet's fault.
- {doc}`/architecture/fault-tolerance` for how recovery works underneath.
