# Fault tolerance

This page describes how a distributed Batcher query survives failure, and which knob
controls each layer of recovery.

At scale something is always failing: a node is preempted, a task hits a transient error, a connection drops mid-shuffle. Batcher's distributed path makes those failures slow a query down rather than kill it, and a recovered result is identical to one that never failed.

Two invariants make recovery sound:

- **Mergeable algebra.** Stateful operators are `partial`, `combine`, and `finalize`
  with an associative, commutative `combine`, so a lost partition can be recomputed and
  merged back in *any* order without changing the result. Recovery never has to
  reconstruct an exact interleaving.
- **Deterministic, source-recomputable tasks.** A shuffle task is a pure function of
  its durable input partition. Rerunning it produces the same bytes, so a retry is
  always safe.

## Layered retries

The cheapest mechanism handles the common case, and heavier machinery engages only when it can't. Every knob lives in `config.distributed` ({doc}`../configuration/options`). The blocks on this page share one base config:

```python
# docs: run
import dataclasses

import batcher as bt

base = bt.Config()
```

A failure is classified before anything retries: retry here, retry elsewhere, or don't retry. Retries draw from a job-wide budget, so a fleet broken in a way no probe catches fails fast with the first real error. Both are configured in {doc}`../configuration/fault-tolerance`. The figure shows the classification and the three prices a recompute can cost.

![Recovery classifies a failure before it retries, and only one of the three verdicts is a retry. A task that raised, or whose worker stopped answering, is classified as lost data to be recomputed when it is a RayError that is not a RayTaskError, meaning an actor, a worker or a node died, and likewise for a RetryableShuffleError from an unreachable peer or a ResourceError from a spill file on an ephemeral disk. A deterministic bug, such as a UDF exception, a bad cast, a schema mismatch or a broken runtime environment, is re-raised instead, because every retry re-runs it, burns the job-wide budget, and reports a resource error for a Python bug. An uncontained ECC fault, where the device kept running and answered wrongly, leaves results untrusted and recovery refuses to continue at all, since work already finished there is as suspect as the task that failed. A recompute then costs one of three prices: re-read the source partition and re-run the map by default, usually the longest phase; fetch an off-node replica when shuffle_replication is above 1 and the copy was acknowledged before the bucket was advertised; or migrate while the worker is still alive, given advance notice from spot metadata, a SIGTERM or a Slurm deadline. Recovery introduces its own hazard, a worker presumed dead that is not, so each round carries a higher epoch and a reducer discards any batch arriving under a stale one.](/_static/diagrams/fault_recovery.svg)

### Ray-level task and actor retries

The first line of defense is the scheduler itself. Ray retries a transient task
failure, such as a flaky node or a dropped connection, before any app-level recovery
engages, because a shuffle task is deterministic and recomputed from a durable source.

```python
# docs: run
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        task_max_retries=2,  # rerun a failed shuffle task
        retry_on_transient=True,  # extend retries to application exceptions
        actor_max_restarts=1,  # respawn a crashed compute actor (map/inference pool)
        actor_max_task_retries=1,  # rerun the in-flight call on the respawned actor
    )
)
print(cfg.distributed.task_max_retries)
# 2
```

`task_max_retries` covers worker death, and `retry_on_transient` extends it to
transport-classified transient application errors. `actor_max_restarts` and
`actor_max_task_retries` cover the long-lived compute actors that back the map and
inference pools. A `0` anywhere restores Ray's no-retry default.

### Shuffle recompute on worker loss

Beneath Ray's retries sits the app-level recovery loop. When a shuffle worker is lost,
its output partition is recomputed from its durable source partition and re-fetched.
This is the lineage-recovery path the mergeable algebra makes safe.

```python
# docs: run
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        recovery_max_attempts=3,  # recompute -> retry rounds before failing loudly
        recovery_backoff_base_s=0.5,  # exponential backoff between rounds
    )
)
```

`recovery_max_attempts` bounds the rounds, so a still-broken shuffle fails with a clear error rather than looping, and the exponential backoff keeps a flaky network from being hammered.

### Detecting a dead peer

The Flight transport treats a peer as dead when the gap between batches in a fetch exceeds `flight_idle_timeout_s` (60 seconds by default), long enough that a GC pause isn't misread as death. `flight_keepalive_s` adds an HTTP/2 keepalive ping that notices a silently dropped connection sooner.

## Epoch fencing

A worker presumed dead may not be, and a recomputed partition must not be double-counted with a straggling original. Each recovery round runs under a higher *epoch*, and a reducer discards any batch tagged with a stale one. A zombie producer that wakes up late can't corrupt the result.

## Straggler mitigation

A degraded but live node never fails, so nothing recomputes it, yet it stalls a shuffle barrier. Speculative execution backs up a slow task and takes whichever copy finishes first. Shuffle tasks are deterministic, so both copies are identical.

```python
# docs: run
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        speculation_max_backups=1,  # one concurrent backup at a barrier
        speculation_straggler_factor=1.5,  # back up a task 1.5x slower than the median
        speculation_min_finished_frac=0.75,  # only once 75% of tasks have finished
    )
)
```

One backup chases the single worst straggler, which keeps a uniformly slow stage from spawning a backup per task. `0` turns speculation off.

## Credit-based backpressure

Backpressure guards against the most common failure of all, running out of memory. One credit is one in-flight `RecordBatch` slot, so a channel's credit window bounds its buffered memory, and a producer blocks when its peer's credits reach zero. Carbonite grants the window and clamps any request to `default_credits` times `credit_ceiling_factor`.

```python
# docs: run
cfg = base.replace(
    flow_control=dataclasses.replace(
        base.flow_control,
        default_credits=16,  # in-flight batch slots per channel (the default)
        credit_ceiling_factor=4,  # max window = default_credits x this
    )
)
print(cfg.flow_control.default_credits * cfg.flow_control.credit_ceiling_factor)
# 64
```

`config.distributed.adaptive_credits`, on by default, runs a TCP-like AIMD controller on top: it grows the window by `aimd_alpha` per round trip and shrinks it by `aimd_beta` under memory backpressure. Flow control never changes the merged output. Bulk batches move over Arrow Flight and never through the Ray object store.

## Resilience profiles

Rather than tune each knob, pick a `config.distributed.resilience` profile. `"default"` suits a stable on-demand cluster. `"spot"` hardens the budgets as a bundle for a preemptible one: more actor restarts, task retries and recompute attempts, a spaced backoff so a preemption wave isn't retried in a tight loop, the HTTP/2 keepalive, and a brief wait for the autoscaler to replace churned capacity. An explicit value beats the profile, and the profile beats the default. A preemptible environment is detected and switched to `"spot"` automatically.

```python
# docs: run
spot = base.replace(distributed=dataclasses.replace(base.distributed, resilience="spot"))
with bt.config_context(spot):
    print(bt.from_pydict({"a": [1, 2]}).to_pydict())
# {'a': [1, 2]}
```

## Draining before a node goes away

Everything above is reactive. It notices a worker after the worker is already gone, and
pays a recompute for the work that went with it. When the environment says in advance
that a node is about to be taken away, Batcher instead migrates that worker's shuffle
output to a survivor while the worker is still alive, which costs one copy rather than a
full re-read of the source. Batcher checks three kinds of advance notice, because a given
cluster offers only one of them:

Batcher watches three sources of advance notice: cloud spot metadata (AWS, Google Cloud, Azure and Alibaba Cloud), an orchestrator signal (`SIGTERM` from Kubernetes or Slurm, `SIGUSR1` as Slurm's early warning), and a wall-clock deadline such as `SLURM_JOB_END_TIME`. Draining begins `config.distributed.drain_lead_s` seconds before a known deadline. On a scheduler that publishes only a wall-clock limit, export the lease:

```bash
export BATCHER_DEADLINE_EPOCH_S=$(( $(date +%s) + 4 * 3600 ))
```

:::{dropdown} How each notice source works
Cloud metadata answers on a spot instance. Batcher polls the AWS `instance-action`
endpoint, the Google Cloud `preempted` flag, Azure Scheduled Events, and the Alibaba Cloud
spot `termination-time`, treating only `Preempt` and `Terminate` as reclamation so routine
host maintenance doesn't migrate the fleet. The AWS probe presents an IMDSv2 session token,
without which it is silently dead on any instance launched with `HttpTokens=required`.

Only one of those endpoints can answer on a given node, and on a neocloud, an HPC cluster or
on-prem hardware none of them can. So Batcher skips the platforms this node isn't. It reads
both the provider it detected from the environment and what the firmware says the node was
built as, so a GPU cloud reselling hyperscaler capacity keeps the endpoint that answers for
it. It also stops probing an endpoint that has been unreachable three times running. A
metadata service doesn't appear partway through a job, and the alternative was paying a
timeout per endpoint on every poll for the life of the worker. Reachability resets that
count rather than the answer: a spot node spends its whole life being told "not draining",
which still proves the endpoint is there.

A signal arrives from an orchestrator. `SIGTERM` is what Kubernetes sends on eviction and
what Slurm sends when a job hits its time limit. `SIGUSR1` is Slurm's early warning, sent
ahead of the limit when the job was submitted with `--signal=B:USR1@120`. Batcher chains
to whatever handler you already installed, so your own checkpoint hook still runs.

A wall-clock deadline is known in advance. This is the case a batch scheduler leaves you in:
a Slurm allocation is not reclaimed with a notice, it just ends at a time fixed when the
job was submitted, and every process in it is killed then. Batcher reads
`SLURM_JOB_END_TIME` and begins draining `config.distributed.drain_lead_s` seconds
before it. Because this is a local clock comparison it needs no metadata service, no
signal, and no cooperation from the scheduler, which is what makes it work on an on-prem
HPC cluster where the other two sources are silent.

Slurm is the only scheduler that publishes the moment an allocation ends. PBS, LSF, Grid
Engine and HTCondor publish a wall-clock *limit* instead, which is not a time. Export the
lease and they reach the same drain path:

```bash
export BATCHER_DEADLINE_SECONDS=$(( 2 * 3600 ))
```

The lease is measured from when the process started, read from `/proc`. That is exact for a
script that starts Python first. It over-states the remaining time by however long a job
script spends before that, which drains late, so prefer the absolute form below when your
launcher knows the moment.

Any launcher that knows the exact moment its lease expires can give that instead, as Unix
epoch seconds:

```bash
export BATCHER_DEADLINE_EPOCH_S=$(( $(date +%s) + 4 * 3600 ))
```

An allocation with a known deadline is treated as preemptible, so it selects the `"spot"`
profile automatically. A Slurm job submitted with no time limit is not: Slurm exports a
saturated sentinel rather than omitting the variable, and Batcher rejects a deadline more
than a year out, so an unlimited job is left on the default budgets.

Draining changes *where* a partial result lives, never what it holds. The output is
unchanged.
:::

## Capacity that is leaving

Batcher reads Ray's drain list and excludes nodes being scaled in or evicted from every fan-out, so a query mid scale-in is provisioned against the nodes that will still be there. Under a known deadline, each wait for the head, the autoscaler or a placement group shrinks to the time actually left.

:::{dropdown} Details
A node the autoscaler is scaling in, or whose pod Kubernetes is evicting, stays alive and
keeps advertising its full resources so the work already on it can finish. Sizing a *new*
fleet onto it is what costs: the placement group reserves bundles on a node being removed,
the actors land, and the shuffle pays a recompute for output that was never going to
survive. Batcher reads Ray's drain list and excludes those nodes from every fan-out
sizing, so a query mid scale-in is provisioned against the nodes that will still be there.

It never narrows to nothing. If every remaining node is draining, the fleet is placed
anyway, because running on capacity that is going away beats not running, and the recovery
machinery above exists for exactly that case.

**Waits under a deadline.**

The scheduler waits in three places before any work happens: for the head to answer, for
the autoscaler to deliver capacity, and for a placement group to become satisfiable. Each
is bounded, and each bound was chosen for a cluster with no horizon, where waiting two
minutes for capacity is free if the alternative is running under-provisioned.

Under a lease it is not free, it is the entire remaining budget. A Slurm allocation with 90
seconds left would spend 180 waiting for autoscaler nodes that arrive after the kill, and
die having computed nothing. So when a deadline is known, each wait shrinks to the time
actually left, minus `drain_lead_s` for the migration window. Giving up sooner falls back
to running on the capacity already present, which is what these waits already do when the
autoscaler stalls.

A wait cut short this way records nothing about the cluster. Running out of time is a fact
about the job, not about how far the autoscaler would have gone, and treating it as a
learned capacity ceiling would make every later query in the process skip a wait it never
actually probed.

Being killed is also what leaks an autoscaler floor. `request_resources` is sticky and
lives in the autoscaler rather than the driver, so a job killed before its teardown runs
leaves the cluster pinned at full size with nothing running against it. The drain hook
drops the floor, which is why it is armed for preemptible deployments.
:::

## Shuffle-output replication

:::{warning}
Leave `shuffle_replication` at 1. The practical limits below say why.
:::

Losing a mapper normally forces a recompute: re-read its source partition from object
storage and re-run the map, usually the longest phase of a query. Setting
`config.distributed.shuffle_replication` above 1 places a copy of each mapper's output
on an off-node survivor, so a reducer fetches the byte-identical bucket instead, at the
cost of one extra network copy.

An aggregate's mapper publishes pre-aggregated partial state, typically far smaller than its source, so copying it is cheaper than regenerating it. A join, sort or window mapper publishes rows, so its copy is larger. A replica is advertised only once
its copy has been acknowledged, and a source's replicas are retired when it's
recomputed, so a reducer can never read a stale replica under a superseded epoch.

Every Flight shuffle takes it: aggregate, join, sort, and window. A wide aggregate
reduces through a combiner tree, and each level's merged partials are copied off-node
before the next level is built on them, so losing a combiner costs a re-fetch rather
than discarding every level built so far. That is the cheapest copy in the shuffle,
because a level's output is several partials already merged into one.

## Practical limits

Fault tolerance applies to the distributed path, which needs the optional `[ray]` extra. Single-node execution has none of this machinery and none of its overhead.

- Shuffle output lives on the worker that produced it, in memory with a local-disk spill. A lost worker's buckets are recomputed.
- Keep `shuffle_replication` at its default of 1, which recovers exactly. A recorded 3-node run found that above 1 a worker loss dropped that worker's share of the rows instead of failing. The cause, replicas copying a shuffle stage nobody had published, is fixed and pinned in [`tests/integration/test_shuffle_replication.py`](https://github.com/stephenoffer/batcher/blob/main/tests/integration/test_shuffle_replication.py), but no cluster run since the fix is recorded, so replication above 1 is unverified on real hardware.
- Draining runs under the `"spot"` profile. A cluster whose signals Batcher can't see needs `BATCHER_SPOT=1`, an exported `BATCHER_DEADLINE_EPOCH_S`, or `resilience="spot"`.
- Recovery covers workers, not the driver. A job that must survive its driver runs as a streaming query with `checkpoint=`, which restarts from its last committed offset ({doc}`Streaming </user-guide/moving-data/streaming/index>`).

:::{dropdown} Signal handling inside Ray actors
The signal traps need the main thread. A worker that can't install them, the usual case inside a Ray actor, falls back to the metadata and deadline polls. Those run every 5 seconds (`PreemptionMonitor`'s `poll_interval_s`), each metadata probe bounded at 0.3 seconds, which fits inside the 30 seconds or more that cloud providers give before reclamation. The once-per-session `attached to Ray` INFO line reports `resilience` and `shuffle_replication` alongside the node count.
:::

## See also

- {doc}`Carbonite </architecture/internals/carbonite>`: the resource manager, memory envelope,
  and the credit model in detail.
- {doc}`Execution model <execution>`: pipelines, breakers, and the mergeable algebra
  that makes recovery sound.
- {doc}`Configuration options <../configuration/options>`: every fault-tolerance,
  memory, and flow-control field with its default.
- {doc}`Fault-tolerance options <../configuration/fault-tolerance>`: the quarantine
  thresholds and the job-wide retry budget.
- {doc}`Unstable nodes </user-guide/operate/running/unstable-nodes>`: the operator's walkthrough
  for a GPU fleet whose nodes and devices fail underneath a running job.
