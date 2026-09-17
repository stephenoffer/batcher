# Fault tolerance

This page describes how a distributed Batcher query survives failure, and which knob
controls each layer of recovery.

A distributed query runs across many workers, and at scale something is always failing.
A node is preempted, a task hits a transient error, or a network connection drops
mid-shuffle. Batcher's distributed path is built so those failures slow a query down
rather than kill it, and a recovered result is identical to one that never failed.

Two invariants make recovery sound:

- **Mergeable algebra.** Stateful operators are `partial`, `combine`, and `finalize`
  with an associative, commutative `combine`, so a lost partition can be recomputed and
  merged back in *any* order without changing the result. Recovery never has to
  reconstruct an exact interleaving.
- **Deterministic, source-recomputable tasks.** A shuffle task is a pure function of
  its durable input partition. Rerunning it produces the same bytes, so a retry is
  always safe.

## Layered retries

Recovery is defense in depth. The cheapest mechanism handles the common case, and
heavier machinery engages only when it can't. The knobs live in `config.distributed`,
documented in {doc}`../configuration/options`.

Every layer below is a mechanism for *repeating* work that was lost. Two things bound them,
because repetition alone is not recovery. A failure is classified first into one of three
verdicts: retry here, retry elsewhere, or do not retry. A failure local to one machine that
is retried in place walks the whole queue onto it. Retries are also drawn from a job-wide
budget, so a fleet broken in some way no probe catches fails quickly with the first real
error rather than slowly with the last one. Both are configured in
{doc}`../configuration/fault-tolerance`.

The figure below shows that classification, the three verdicts it reaches, and the three
prices a recompute can cost depending on what was arranged beforehand.

![Recovery classifies a failure before it retries, and only one of the three verdicts is a retry. A task that raised, or whose worker stopped answering, is classified as lost data to be recomputed when it is a RayError that is not a RayTaskError, meaning an actor, a worker or a node died, and likewise for a RetryableShuffleError from an unreachable peer or a ResourceError from a spill file on an ephemeral disk. A deterministic bug, such as a UDF exception, a bad cast, a schema mismatch or a broken runtime environment, is re-raised instead, because every retry re-runs it, burns the job-wide budget, and reports a resource error for a Python bug. An uncontained ECC fault, where the device kept running and answered wrongly, leaves results untrusted and recovery refuses to continue at all, since work already finished there is as suspect as the task that failed. A recompute then costs one of three prices: re-read the source partition and re-run the map by default, usually the longest phase; fetch an off-node replica when shuffle_replication is above 1 and the copy was acknowledged before the bucket was advertised; or migrate while the worker is still alive, given advance notice from spot metadata, a SIGTERM or a Slurm deadline. Recovery introduces its own hazard, a worker presumed dead that is not, so each round carries a higher epoch and a reducer discards any batch arriving under a stale one.](/_static/diagrams/fault_recovery.svg)

### Ray-level task and actor retries

The first line of defense is the scheduler itself. Ray retries a transient task
failure, such as a flaky node or a dropped connection, before any app-level recovery
engages, because a shuffle task is deterministic and recomputed from a durable source.

```python
# Illustrative: the fault-tolerance section of config.distributed.
import dataclasses
from batcher import Config

base = Config()
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        task_max_retries=2,  # rerun a failed shuffle task
        retry_on_transient=True,  # extend retries to application exceptions
        actor_max_restarts=1,  # respawn a crashed compute actor (map/inference pool)
        actor_max_task_retries=1,  # rerun the in-flight call on the respawned actor
    )
)
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
# Illustrative.
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        recovery_max_attempts=3,  # recompute -> retry rounds before failing loudly
        recovery_backoff_base_s=0.5,  # exponential backoff between rounds
    )
)
```

`recovery_max_attempts` bounds the recompute and retry rounds, so a still-broken
shuffle fails with a clear error rather than looping. The exponential backoff keyed on
`recovery_backoff_base_s` keeps a flaky network from being hammered in a tight loop. A
larger cluster with a higher background failure rate raises both.

### Detecting a dead peer

A worker can stop responding without an explicit failure. The Flight transport treats a
peer as dead when the gap between batches in a fetch exceeds `flight_idle_timeout_s`.
That timeout is generous, so a long GC pause isn't misread as death, but bounded, so a
truly dead peer is detected and its partition recomputed. Setting `flight_keepalive_s`
adds an HTTP/2 keepalive ping that notices a silently dropped connection faster than
the idle timeout alone.

## Epoch fencing

Recovery introduces a hazard: a worker presumed dead may not actually be dead, and a
recomputed partition must not be double-counted with a straggling original. Each
recovery round runs under a monotonically increasing *epoch*. A reducer accepts a
partition tagged with the current epoch and fences out any batch arriving under a stale
epoch, discarding it. A zombie producer can wake up after its work was reassigned. It
can't corrupt the result. Its late bytes are ignored. Fencing and the
deterministic-task invariant together make a recomputed partition safe to merge back in.

## Straggler mitigation

A node that is degraded but alive is worse than a dead one. It never failed, so
nothing recomputes it. It still stalls a shuffle barrier. Speculative execution backs
up a slow survivor and takes whichever copy finishes first. Because shuffle tasks are
deterministic, the two copies are identical, so the result is unchanged.

```python
# Illustrative.
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        speculation_max_backups=1,  # one concurrent backup at a barrier
        speculation_straggler_factor=1.5,  # back up a task 1.5x slower than the median
        speculation_min_finished_frac=0.75,  # only once 75% of tasks have finished
    )
)
```

The default is `speculation_max_backups=1`, so one backup chases the single worst
straggler and nothing else. Set it to `0` and the barrier becomes a plain wait. The
bound is what keeps a uniformly slow stage from spawning a backup per task and
oversubscribing the cluster.

## Credit-based backpressure

Backpressure is fault tolerance against the most common failure of all, running out of
memory. The shuffle uses credit-based flow control. One credit is one in-flight
`RecordBatch` slot, so a channel's credit window is a direct bound on its buffered
memory. A producer blocks when its peer's credits reach zero, so a fast stage can't
flood a slow one and blow up memory. Carbonite is the authority that grants the window
and clamps any request to `default_credits` times `credit_ceiling_factor`.

```python
# Illustrative: config.flow_control.
cfg = base.replace(
    flow_control=dataclasses.replace(
        base.flow_control,
        default_credits=16,  # in-flight batch slots per channel (the default)
        credit_ceiling_factor=4,  # max window = default_credits x this
    )
)
```

The grant is a starting point rather than a fixed window. `config.distributed.adaptive_credits`,
on by default, runs a TCP-like AIMD controller that grows the window by
`config.flow_control.aimd_alpha` per round trip and multiplicatively shrinks it by
`config.flow_control.aimd_beta` when it sees memory backpressure, so the shuffle backs
off under pressure instead of holding a fixed window. Flow control never changes the
merged output, which keeps the guarantee that a distributed result equals a single-node
one intact.

The data plane bypasses the Ray object store entirely. Bulk Arrow batches move over
Arrow Flight (`bc-transport`), and only small control-plane strings transit Ray. The
object store is where the serialization overhead and OOM risk of a shuffle would
otherwise come from.

## Resilience profiles

Rather than tune each knob, pick a `config.distributed.resilience` profile. `"default"`
keeps conservative budgets tuned for a stable on-demand cluster. `"spot"` hardens them
as a bundle for a churning preemptible cluster. It raises actor restarts, task retries
and recompute attempts to ride out repeated loss, spaces the recovery backoff so a
preemption *wave* isn't retried in a tight loop, turns on the HTTP/2 keepalive so a
dropped peer is noticed fast, lets a stage wait briefly for the autoscaler to replace
churned capacity, and sets `shuffle_replication` to 2. A profile applies *below* any value
you set explicitly, so an explicit override beats the profile, and the profile beats
the default. A preemptible environment is auto-detected and switched to `"spot"` when
`resilience` is left at `"default"`.

## Draining before a node goes away

Everything above is reactive. It notices a worker after the worker is already gone, and
pays a recompute for the work that went with it. When the environment says in advance
that a node is about to be taken away, Batcher instead migrates that worker's shuffle
output to a survivor while the worker is still alive, which costs one copy rather than a
full re-read of the source. Batcher checks three kinds of advance notice, because a given
cluster offers only one of them:

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

## Not scheduling onto capacity that is leaving

A node the autoscaler is scaling in, or whose pod Kubernetes is evicting, stays alive and
keeps advertising its full resources so the work already on it can finish. Sizing a *new*
fleet onto it is what costs: the placement group reserves bundles on a node being removed,
the actors land, and the shuffle pays a recompute for output that was never going to
survive. Batcher reads Ray's drain list and excludes those nodes from every fan-out
sizing, so a query mid scale-in is provisioned against the nodes that will still be there.

It never narrows to nothing. If every remaining node is draining, the fleet is placed
anyway, because running on capacity that is going away beats not running, and the recovery
machinery above exists for exactly that case.

## Waits under a deadline

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

## Shuffle-output replication

Losing a mapper normally forces a recompute: re-read its source partition from object
storage and re-run the map, usually the longest phase of a query. Setting
`config.distributed.shuffle_replication` above 1 places a copy of each mapper's output
on an off-node survivor, so a reducer fetches the byte-identical bucket instead, at the
cost of one extra network copy.

The mergeable algebra makes that trade affordable. A mapper publishes pre-aggregated
partial state, typically far smaller than the source that produced it, so copying it is
much cheaper than regenerating it. A replica is advertised only once
its copy has been acknowledged, and a source's replicas are retired when it's
recomputed, so a reducer can never read a stale replica under a superseded epoch.

Every Flight shuffle takes it: aggregate, join, sort, and window. A wide aggregate
reduces through a combiner tree, and each level's merged partials are copied off-node
before the next level is built on them, so losing a combiner costs a re-fetch rather
than discarding every level built so far. That is the cheapest copy in the shuffle,
because a level's output is several partials already merged into one.

## Requirements and limitations

Fault tolerance applies to the distributed path, which needs the optional `[ray]`
extra. Single-node execution has none of the machinery on this page and none of the
overhead.

- Shuffle output lives on the worker that produced it. `bc-transport`'s partition store
  holds it in memory and spills to that worker's local disk under pressure. There is no
  external shuffle service, so a lost worker's buckets are gone and are recomputed unless
  replication placed a copy elsewhere.
- `shuffle_replication` defaults to 1, meaning no replica. Only the `"spot"` profile
  raises it, which a preemptible environment selects automatically.
- Draining runs only under the `"spot"` profile, so a stable cluster starts no monitor
  and pays nothing. A preemptible or time-limited environment selects that profile
  automatically, but a cluster whose signals Batcher can't see needs `BATCHER_SPOT=1`,
  an exported `BATCHER_DEADLINE_EPOCH_S`, or an explicit `resilience="spot"`.
- The signal traps need the main thread. A worker that can't install them, which is the
  usual case inside a Ray actor, falls back to the metadata and deadline polls.

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
