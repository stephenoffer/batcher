# Fault tolerance

A distributed query runs across many workers, and at scale something is always
failing: a node is preempted, a task hits a transient error, a network connection
drops mid-shuffle. Batcher's distributed path is built so those failures slow a
query down rather than killing it, and so a recovered result is identical to one
that never failed. This page describes the design; the resource model underneath it
is in [Carbonite](../internals/carbonite.md), and the broader distributed picture is
in [the execution model](execution.md).

Two invariants make recovery sound:

- **Mergeable algebra.** Stateful operators are `partial → combine → finalize` with
  an associative, commutative `combine`, so a lost partition can be recomputed and
  merged back in *any* order without changing the result. Recovery never has to
  reconstruct an exact interleaving.
- **Deterministic, source-recomputable tasks.** A shuffle task is a pure function of
  its durable input partition. Rerunning it produces the same bytes, so a retry is
  always safe.

## Layered retries

Recovery is defense in depth: the cheapest mechanism handles the common case, and
heavier machinery engages only when it cannot. The knobs live in
`config.distributed` ([reference](../configuration/options.md)).

### Ray-level task and actor retries

The first line of defense is the scheduler itself. A transient task failure — a
flaky node, a dropped connection — is retried by Ray before any app-level recovery
engages, because a shuffle task is deterministic and recomputed from a durable
source.

```python
# Illustrative — the fault-tolerance section of config.distributed.
import dataclasses
from batcher import Config

base = Config()
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        task_max_retries=2,        # rerun a failed shuffle task
        retry_on_transient=True,   # extend retries to application exceptions
        actor_max_restarts=1,      # respawn a crashed compute actor (map/inference pool)
        actor_max_task_retries=1,  # rerun the in-flight call on the respawned actor
    )
)
```

`task_max_retries` covers worker death; `retry_on_transient` extends it to
transport-classified transient application errors. `actor_max_restarts` and
`actor_max_task_retries` cover the long-lived compute actors that back the map and
inference pools. A `0` anywhere restores Ray's no-retry default.

### Shuffle recompute on worker loss

Beneath Ray's retries sits the app-level recovery loop. When a shuffle worker is
lost, its output partition is recomputed from its (durable) source partition and
re-fetched — the lineage-recovery path the mergeable algebra makes safe.

```python
# Illustrative.
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        recovery_max_attempts=3,      # recompute -> retry rounds before failing loudly
        recovery_backoff_base_s=0.5,  # exponential backoff between rounds
    )
)
```

`recovery_max_attempts` bounds the recompute→retry rounds before a still-broken
shuffle fails with a clear error rather than looping; the exponential backoff keyed
on `recovery_backoff_base_s` keeps a flaky network from being hammered in a tight
loop. A larger cluster with a higher background failure rate raises both.

### Detecting a dead peer

A worker can stop responding without an explicit failure. The Flight transport
treats a peer as dead when the gap between batches in a fetch exceeds
`flight_idle_timeout_s` (generous, so a long GC pause is not misread as death, but
bounded, so a truly dead peer is detected and its partition recomputed). Setting
`flight_keepalive_s` adds an HTTP/2 keepalive ping that notices a silently-dropped
connection faster than the idle timeout alone.

## Epoch fencing

Recovery introduces a hazard: a worker presumed dead may not actually be dead, and a
recomputed partition must not be double-counted with a straggling original. Each
recovery round runs under a monotonically increasing *epoch*. A reducer accepts a
partition tagged with the current epoch and fences out — discards — any batch
arriving under a stale epoch. A zombie producer that wakes up after its work was
reassigned therefore cannot corrupt the result; its late bytes are ignored. Combined
with the deterministic-task invariant, fencing is what lets a recomputed partition
be merged in safely.

## Straggler mitigation

A node that is degraded-but-alive is worse than a dead one: it cannot be recomputed
because it never failed, yet it stalls a shuffle barrier. Speculative execution
backs up a slow survivor and takes whichever copy finishes first. Because shuffle
tasks are deterministic, the two copies are identical, so the result is unchanged.

```python
# Illustrative.
cfg = base.replace(
    distributed=dataclasses.replace(
        base.distributed,
        speculation_max_backups=1,           # one concurrent backup at a barrier
        speculation_straggler_factor=1.5,    # back up a task 1.5x slower than the median
        speculation_min_finished_frac=0.75,  # only once 75% of tasks have finished
    )
)
```

`speculation_max_backups=0` (the default) disables it, and the barrier behaves like
a plain wait. Speculation is bounded so it never oversubscribes the cluster.

## Credit-based backpressure

Backpressure is fault tolerance against the most common failure of all — running out
of memory. The shuffle uses credit-based flow control: one credit is one in-flight
`RecordBatch` slot, so a channel's credit window is a direct bound on its buffered
memory. A producer blocks when its peer's credits reach zero, so a fast stage cannot
flood a slow one and blow up memory. Carbonite is the authority that grants the
window and clamps any request to `default_credits x credit_ceiling_factor`.

```python
# Illustrative — config.flow_control.
cfg = base.replace(
    flow_control=dataclasses.replace(
        base.flow_control,
        default_credits=4,        # in-flight batch slots per channel
        credit_ceiling_factor=16, # max window = default_credits x this
    )
)
```

By default the window is the static grant. `config.distributed.adaptive_credits`
(on by default) turns on a TCP-like AIMD controller that grows the window by
`aimd_alpha` per round trip and multiplicatively shrinks it by `aimd_beta` when it
sees memory backpressure — so the shuffle backs off under pressure instead of
holding a fixed window. It is result-preserving (flow control never changes the
merged output), which keeps the single-node-equals-distributed guarantee intact.

The data plane bypasses the Ray object store entirely: bulk Arrow batches move over
Arrow Flight (`bc-transport`), which is where the serialization overhead and OOM
risk of an object-store shuffle would otherwise come from.

## Resilience profiles

Rather than tune each knob, pick a `config.distributed.resilience` profile.
`"default"` keeps conservative budgets tuned for a stable on-demand cluster.
`"spot"` hardens them as a bundle for a churning preemptible cluster: more actor
restarts and recompute attempts to ride out repeated loss, HTTP/2 keepalive on to
notice a dropped peer fast, and one speculative backup so a degraded node cannot
stall a barrier. The profile applies *below* any value you set explicitly
(`explicit override > profile > default`), and a preemptible environment is
auto-detected and switched to `"spot"` when `resilience` is left at `"default"`.

## See also

- [Carbonite](../internals/carbonite.md) — the resource manager, memory envelope,
  and the credit model in detail.
- [Execution model](execution.md) — pipelines, breakers, and the mergeable algebra
  that makes recovery sound.
- [Configuration options](../configuration/options.md) — every fault-tolerance,
  memory, and flow-control field with its default.
