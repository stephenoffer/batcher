---
name: run-a-distributed-job
description: Take a working single-node Batcher pipeline onto a Ray cluster — the collect(distributed=...) opt-in, cluster config, partitioning and the Arrow Flight shuffle, spill and credit-based flow control, failure/retry behavior, and how to prove the distributed result equals the single-node one. Invoke when scaling a pipeline out, sizing a cluster, or debugging a distributed run.
---

# Run a distributed job

Scaling out is a **deployment change, not a rewrite**. The plan you already have runs
unchanged; you flip one keyword. Everything here assumes the pipeline works and is correct on
one node — fix it there first, because a distributed run is strictly harder to debug and the
answer must be identical anyway. `docs/integrations/compute/ray.md` is the canonical page.

## Decide whether to distribute at all

Distribution buys scale-out and larger-than-memory. It costs a shuffle, actor startup, and
a Ray dispatch. Below the crossover it is a **pessimization**, and `docs/benchmarks/results/scaling.md`
says so plainly: TPC-H sf1 (6M rows) on a 9-node / 128-CPU cluster runs in **86 ms
single-node vs 92 ms distributed**, and an 80k-row filter cost ~2,150 ms of pure Ray fan-out
before `"auto"` learned to stay home (now ~67 ms).

Distribute when the input does not fit in one node's memory; the scan is I/O-bound across many
files and more NICs help; a GPU stage needs more than one GPU; or a `flat_map`-style expansion
would materialize far more rows than the input (~5.8× faster distributed over 120M rows,
because each partition reduces before anything leaves it). Do **not** distribute a query that
already runs in well under a second, a small join, or anything bottlenecked on a single Python
callback — `distributed="auto"` is size-aware and distributes only above
`distributed.distribute_min_rows` (default `1_000_000`), or when a GPU stage forces it.

## The opt-in

There is no distributed API. It is the same plan and the same verbs.

```python
# Needs a Ray cluster + `pip install 'batcher-engine[ray]'`.
import batcher as bt

result = (
    bt.read.parquet("s3://lake/events/*.parquet")
    .filter(bt.col("status") == "purchase")
    .group_by("region")
    .agg(revenue=bt.col("amount").sum())
    .collect(distributed=True)
)
```

`Dataset.collect` is the whole surface:

```
collect(distributed="auto", num_workers=None, spill=False, num_partitions=None,
        adaptive="auto", transport="auto", backend="cpu") -> pa.Table
```

`distributed` is `"auto"` (size-aware) / `True` / `False`; `num_workers` defaults to one per
node with an even CPU share; `num_partitions` is the reducer count; `spill=True` lets stateful
operators spill; `transport` is `"auto" | "flight" | "disk"`; `backend` is `"cpu"` or a GPU
backend (`distributed.gpu_min_rows`, default `10_000_000`, is roughly where GPU starts winning).
The same switch is on `ds.write(path, ..., distributed="auto", num_workers=None)` and on
`ds.iter_batches(batch_size=None, *, distributed=False, num_workers=None, transport="auto")` —
note the latter defaults to `False`, not `"auto"`.

**The `"auto"` gotcha.** `"auto"` distributes only when Ray is *already* initialized (it never
forces a `ray.init()`), **and** the cluster reports more than one node, **and** the estimated
input clears `distribute_min_rows` — or a GPU-bearing `map_batches` forces it. On a workspace
where nothing has attached yet, `"auto"` silently runs in-process; use `distributed=True` when
you mean it. (Unrelated 20M number: the *adaptive re-optimization* gate in `api/adaptive.py` is
off below 20M input rows. That is not the distribution threshold.)

## The invariant: distributed == single-node

Every stateful operator is built once as mergeable `partial → combine → finalize`, with
`combine` associative and commutative, so partials merge in any order. One implementation
serves one core, many cores, and many machines — there is no second semantics that could
disagree. A distributed result is identical to the single-node one; if it isn't, that is a
bug in Batcher, not a tuning problem. Verify it on your own pipeline before a big run,
comparing **order-independently** unless the query ends in an explicit `sort`:

```python
# Needs a cluster.
import batcher as bt

def pipeline() -> bt.Dataset:
    return bt.read.parquet("s3://lake/sample/*.parquet").group_by("region").agg(
        revenue=bt.col("amount").sum()
    )

def rowset(t) -> set:
    return {tuple(r.values()) for r in t.to_pylist()}

assert rowset(pipeline().collect()) == rowset(pipeline().collect(distributed=True, num_workers=2))
```

Sweep `num_partitions` over `1, 2, N` — a bug that appears at only one partition count is the
classic mergeability failure. The in-repo versions live in `tests/integration/` as
`test_distributed.py` (parametrized over both transports), `test_flight_shuffle.py`,
`test_distributed_spilling.py`, and `test_distributed_empty_partition.py`.

## Partitioning and shuffle

- **Ray schedules; it does not carry data.** Tasks, actors, placement groups, and small
  control-plane messages (paths, worker addresses, metrics) go through Ray; bulk Arrow batches
  move worker-to-worker over Arrow Flight (`bc-transport`). Never `ray.put` a `RecordBatch` to
  move it between stages — that is the object-store tax the design exists to remove.
- **Partition count is chosen, not hand-tuned.** There is no `spark.sql.shuffle.partitions`.
  The optimizer sizes fan-out from measured cardinalities (`optimizer.target_rows_per_task`
  = 4M, `target_bytes_per_task` = 256 MiB), clamped to `[4, 4096]` with 16 as the unknown-size
  fallback and capped by `distributed.max_shuffle_partitions` (`2048`); workers default to one
  per non-head node with an even CPU share. `num_partitions=` / `num_workers=` are the
  overrides — reach for them only after `ds.stats()` says the split is wrong. Any reducer
  count is result-correct.
- **`ds.repartition(num_files=None, *, by=None, target_size_mb=None)` is about output files**,
  not a shuffle hint; `ds.shuffle(seed=...)` is a row shuffle, not a redistribution.
- **`transport="auto"`** picks Flight on a genuine multi-node cluster and a disk shuffle on one
  node. The disk shuffle writes to a driver-local work dir, so forcing `transport="disk"`
  across nodes without `distributed.shared_filesystem=True` yields tasks that cannot find
  their input. Skew is handled by salting (`skew_join_salt`, `skew_join_fraction=0.1`) and
  recursive re-partitioning of an oversized bucket.

## Memory, spill, and credits

Per-node memory is a function of the *partition*, not the relation, because each partition
reduces before anything leaves it. When a partition still does not fit, aggregation, distinct,
sort, join build and partitioned windows spill: the query gets slower, it does not die.

```python
from batcher import Config, MemoryConfig, FlowControlConfig
from batcher.config import DistributedConfig
import batcher as bt

cfg = Config().replace(
    memory=MemoryConfig(spill_dir="/mnt/local_ssd/batcher", spill_compression="zstd"),
    flow_control=FlowControlConfig(default_credits=16),
    distributed=DistributedConfig(namespace="nightly-etl", resilience="spot"),
)
bt.set_config(cfg)          # or: with bt.config_context(cfg): ...
```

- `MemoryConfig`: `soft_limit=0.85`, `hard_limit=0.9`, `max_memory_bytes=None` (auto-sensed —
  `unbounded_memory=True` is how you opt out of spilling, not `None`), `spill_dir=None`
  (point it at fast local disk, never a network mount), `spill_remote_uri=None` (object-store
  spill), `spill_compression="auto"`, `spill_bucket_max_bytes=128 MiB`.
- `FlowControlConfig` is the credit-based backpressure on the Flight exchange — **1 credit =
  1 in-flight RecordBatch slot; the producer blocks at 0**. `default_credits=16`,
  `credit_ceiling_factor=4`, `credit_byte_budget=256 MiB`, `shuffle_fan_in=8`,
  `shuffle_fetch_fan_in=32`, AIMD via `aimd_alpha=1` / `aimd_beta=0.5`, `backpressure_high=0.7`
  / `low=0.4`. `distributed.adaptive_credits=True` adapts the window — leave it on.

## Cluster setup

`config.distributed` is where the cluster lives. `ray_address=None` (the default) attaches to
a running cluster when `RAY_ADDRESS` is set or a managed control plane is detected, and starts
a local Ray only when nothing is reachable — calling `ray.init()` yourself first on a workspace
that exports no `RAY_ADDRESS` strands the job on a local single-node Ray. The fields that
matter on day one: `runtime_env` (every worker needs `batcher` **and its compiled extension**
importable — a wheel mismatch fails at the first task with an import error naming the native
module), `namespace` (isolates a job's shuffle actors), `transport`, and `shuffle_token` (also
`BATCHER_SHUFFLE_TOKEN`) plus `distributed.tls` for TLS/mTLS on an untrusted network.

**Object store vs worker memory.** Bulk batches bypass plasma, so the Ray object store carries
only control-plane metadata — size it small and give the memory to workers.
`object_store_memory_bytes` applies **only when Batcher starts Ray locally** (Ray rejects it
when attaching). Per-worker memory is not a knob: it derives from the *worker node's* RAM as
`node_mem * memory.soft_limit / workers_per_node`, so a fat head node cannot mis-budget a thin
worker. Flight has **no port setting** — the shuffle server binds an ephemeral port and
advertises the Ray node IP, so open the node-to-node range rather than hunting a `flight_port`.

## On a GPU fleet

Seven controls exist for a cluster whose scarce resource is accelerators rather than cores, all
of them off or unbounded by default so a fleet that configures nothing behaves as before.
`config.accelerator` is where they live (`docs/configuration/accelerator.md`), and
`docs/user-guide/operate/running/gpu-fleets.md` is the walkthrough.

- **Power binds before slots.** `accelerator.energy.power_budget_watts` clamps GPU fan-out to
  what a rack can actually power — ten 700 W devices on a 10 kW circuit, not the sixteen its
  slots hold. Exceeding a real budget does not fail; the driver clamps every device in the
  zone, which reads as the whole rack getting slower for no visible reason.
- **A collective must not span a fabric.** Placement is already STRICT_PACK for a
  `gpu_collective` stage, but STRICT_PACK cannot make a node wider than its NVLink domain. A
  world size above the widest domain is logged with both numbers. Label nodes with
  `batcher.io/rack`, `batcher.io/fabric`, and `batcher.io/power-zone` so the topology is
  readable; unlabelled, every topology decision degrades to the node-level one it made before.
- **Residency constrains placement, not just storage.** `governance.ResidencyCatalog` states
  which regions a dataset may be *computed* in, and `dist...fabric.permitted_nodes` filters the
  fleet to the nodes every input permits. Start in `advisory` mode: `strict` refuses, and the
  first strict run of a large pipeline fails somewhere nobody predicted.
- **Energy is measured and mergeable.** `core.energy.measure_stage` records what a stage drew,
  marking whether the figure came from a device reading or a datasheet, and `merge_ledgers`
  folds per-worker ledgers into one figure equal to the single-node one. Report it with
  `observe.format_energy_report`; the idle share is the number to act on.

- **The wires decide as much as the devices.** `bt.accelerators()["fabric"]` carries the rail
  map (which NIC each device leaves through), the peer islands (which pairs copy without
  touching host memory), and what a byte off a device costs against a local one. Two conditions
  appear in `bt.accelerator_problems()` and nowhere else: devices piled onto one rail, so a
  cross-node stage uses a fraction of the port rate, and a node where no pair can copy
  directly. The GPU task's environment carries the rail-aligned NIC list to the collective
  library, so a UDF running its own NCCL group is not left to probe for it.
- **Compression is a trade against the link.** `distributed.flight_compression="auto"` picks
  the wire codec from the measured fabric: past roughly 25 Gb/s a compressor becomes the
  ceiling rather than the wire, and on a 400 Gb/s port compressing costs throughput. An
  explicitly named codec is never overruled.
- **A slow shuffle can name its slow peer.** `ShuffleSession.stats()` carries per-rail
  throughput and a straggler figure alongside the locality ratio. A fleet with one node on a
  renegotiated link reads exactly like a fleet that is uniformly busy in every other figure.

Two failure modes worth naming, because neither surfaces as an error: a device the driver has
clamped runs at a fraction of its rate with the job's own timings as the only symptom
(`ml.devices.device_feed_advice` separates that from a starved pipeline), and a device
reporting uncorrectable ECC errors returns wrong tensors (health checking quarantines it, but
it is opt-in because it needs `pynvml` on every worker).

## Failures and retries

- **A preempted worker recomputes its partition** from its durable input. That makes
  idempotency a requirement: a `map_batches` `fn` with an external side effect (vector-DB
  insert, REST POST) can apply it **twice**. Make sinks upserts on a stable key; a pure
  transform is already safe.
- Budgets, outermost first: Ray task/actor retries (`task_max_retries=2`,
  `retry_on_transient=True`, `actor_max_restarts=1`, `actor_max_task_retries=1`), then
  app-level **recompute from shuffle lineage** (`recovery_max_attempts=3`,
  `recovery_backoff_base_s=0.5`, exponential with equal jitter so a preemption wave does not
  retry in lockstep). Lineage is Spark-style recompute, not replication.
- **Epoch fencing** makes recompute safe: reducers accept only the current epoch and discard
  batches from a zombie producer, so a recomputed partition cannot be double-counted.
  `shuffle_replication=1` (the `spot` profile raises it to 2) puts each mapper's buckets on
  peers so a lost worker costs a re-fetch instead of a recompute round. The copy goes to
  another node, and — when the market labels say so — to a node that is not spot, because a
  reclamation takes an instance group rather than a machine. Dead peers are detected by
  `flight_idle_timeout_s=60.0` (+ optional `flight_keepalive_s`).
- Stragglers: speculation is **on** with one backup by default (`speculation_max_backups=1`);
  `speculation_straggler_factor=1.5`, `speculation_min_finished_frac=0.75`. Set it to `0` to
  turn it off.
- `resilience="spot"` hardens all of the above as a bundle for a churning cluster — prefer
  it to tuning six knobs by hand. A preemptible environment is auto-detected and switched to
  `"spot"` when `resilience` is left at `"default"`; explicit overrides still win.
- Bad input files: `distributed.on_read_error` is `"error"` (fail fast) or `"skip"`.
- Autoscaling is requested and then waited for, bounded by `autoscale_wait_s` (`"auto"`:
  a bounded wait on an autoscaling cluster, off on a fixed one), with early exits after
  `autoscale_stall_s=90` of flat capacity or `autoscale_startup_grace_s=12` of no growth at
  all. Jobs that consistently run on too few workers hit one of those, **not**
  `placement_timeout_s=60.0`, which bounds the gang reservation made *after* the fan-out is
  chosen.
- **A map or inference stage that has produced nothing for two minutes now says why.** The
  engine compares the pending ask against the live topology and reports one of: no node can
  host one task (naming the binding resource and the widest node's figure — waiting cannot fix
  this), the cluster is short of free capacity (the three numbers; another job is holding it),
  or nothing at all. A placement group that times out logs the same diagnosis instead of
  falling back silently. If you are diagnosing a hung distributed job, that warning is the
  first thing to read. The *shuffle* barrier still prints the older, unresolved warning.

## Where the fleet lands

Placement is decided per query and is result-identical, so it is a cost and latency question
rather than a correctness one. Two behaviours are worth knowing:

- **One availability zone when the fleet fits in one.** A shuffle moves nearly all its bytes
  worker to worker, and cross-zone bytes are billed in both directions. The engine reserves
  the fleet's bundles in the zone with the most free capacity that can host it, and does
  nothing on a single-zone cluster, on unlabelled nodes, or when no one zone has room. Turn it
  off with `distributed.zone_aware_placement=False` if the zone spread is deliberate.
- **The fan-out and the placement are recorded as decisions.** `ds.explain(analyze=True)` and
  the dashboard show the narrowing chain ("fan-out 8 -> 6: cluster_fill 40, clamp 6") and the
  reservation ("reserved 8 bundle(s) SPREAD in us-west-2b"). That is the answer to "why did my
  job use a tenth of the cluster", and it is the first place to look before tuning anything.

## Pre-flight checklist

1. Pipeline is correct and green single-node.
2. Distributed result verified equal to single-node on a sample, across ≥2 partition counts.
3. `ds.explain()` shows the pushdowns you expect; `ds.stats()` on the sample shows no
   surprise operator dominating.
4. Every `map_batches` callback is idempotent, or its sink is an upsert.
5. `runtime_env` pins the same Batcher wheel the driver runs.
6. `spill_dir` points at fast local disk; a memory ceiling is set if the node is shared.
7. `namespace` is unique to this job; `shuffle_token`/`tls` set if the network is shared.
8. Sized deliberately: `num_workers` / `num_partitions` left to default unless measured.

## Benchmark it

```bash
just bench-dist        # python benchmarks/run.py --benchmark distributed
```

It runs four TPC-H plans (`groupby-agg`, `groupby-2key`, `join+groupby`, `distinct`) both ways,
checks equivalence **first**, and refuses to time a divergent result — so a `bench-dist`
failure means a mergeability bug, not a slow run (it exits 0 with a skip if Ray is absent).
`benchmarks/scenarios/dist_bench.py --workers 4` and `benchmarks/cluster/vs_ray_daft.py` are
the standalone drivers.

Keep claims honest: on the same 16×8 cluster Batcher **loses `filter_count` to Daft at sf10
(0.92×) and sf100 (0.84×)**, and below one GPU's memory a single-GPU cuDF run beats the
distributed one.

## See also

- `docs/integrations/compute/ray.md`; `docs/architecture/execution.md`, `docs/architecture/fault-tolerance.md`;
  `docs/configuration/options.md` (every knob named above).
- `docs/architecture/deep-dives/distribution/shuffle-flight.md`, `docs/architecture/deep-dives/distribution/distributed-scheduling.md`, `docs/architecture/deep-dives/distribution/credit-flow-control.md`, `docs/architecture/deep-dives/memory/spilling.md`, `docs/architecture/deep-dives/operators/mergeable-algebra.md`.
- `docs/user-guide/operate/running/gpu-fleets.md` and `docs/configuration/accelerator.md` for a GPU fleet's
  power budget, fabric-aware placement, device health, and data residency.
- `docs/benchmarks/results/scaling.md`, `docs/benchmarks/comparisons/vs-spark.md` — the Spark page is an architectural argument and
  publishes **no** head-to-head timings; do not quote it as a measurement.
- Skills: `write-a-batcher-pipeline`, `optimize-a-slow-query`, `debug-a-batcher-query`,
  `build-an-ml-pipeline`, `migrate-from-spark`, and
  `add-distributed-operator` (only if you are changing the engine, not using it).
