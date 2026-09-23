# GPU execution

This page describes the two paths on which Batcher runs work on a GPU, and the scheduling that keeps the device busy on each.

A GPU idles as easily as a CPU core and costs far more while it does. The job of an engine running GPU work is to keep the device fed, and almost every way of failing at that is a *scheduling* failure rather than a kernel failure.

Both GPU paths are Python. The `bc-*` crates contain no GPU code at all, which follows from the Arrow-only data-plane contract. The two paths differ in what they put on the device and in who dispatches them.

| Path | What runs on the device | Where |
|---|---|---|
| GPU relational backend ({py:meth}`collect(backend="gpu") <batcher.Dataset.collect>`) | cuDF dataframe ops, with a torch scatter-reduce fallback | `core/gpu_plan/` translates, `dist/gpu/` schedules, in Ray tasks with `num_gpus=1` |
| GPU inference stage (`map_batches(..., num_gpus=...)`) | the user's torch model | a Python Ray actor |

The relational backend is an opt-in accelerator for relational shapes, described in the next section. An unsupported shape, an OOM, or a GPU-less cluster falls back to the CPU engine, so `backend="gpu"` is always safe to request.

## The relational backend

`collect(backend="gpu")` walks a plan's operator IR and replays it on a cuDF DataFrame, one case per operator and one per expression. A plan reaches the device only when *every* node in it translates, so coverage is what decides how much of a real query is accelerated rather than a list of features.

| Layer | What translates |
|---|---|
| Operators | filter, project, group-by aggregate, sort, distinct, limit, window, unnest, unpivot, row id, plus equi/semi/anti joins of two chains and unions of chains |
| Aggregates | sum, count, count(\*), mean, min, max, var, stddev, median, quantile, count-distinct, product, bool-and, bool-or |
| Window functions | row_number, rank, dense_rank, percent_rank, cume_dist, ntile, lag, lead, first_value, last_value, nth_value, forward and backward fill, and the aggregates over a whole partition, a running frame, or a moving frame |
| Expressions | arithmetic, comparison, boolean, cast, `CASE`, coalesce, nullif, greatest, least, `IN`, null and NaN tests, twenty unary math functions, and the string and date vocabularies |

Anything outside that set is *declined* rather than approximated, and the stage runs on the CPU engine instead. That distinction is the whole safety argument for the backend: a fallback costs time, and an approximation costs a wrong answer.

That safety rests on a partition rather than on a list of features, and a partition is a shape.

![Why the device tier needs machinery no other tier needs, and what that machinery buys. Every other execution tier consumes the same Rust bc_expr::Expr, so there is one definition of what a shape means and it cannot drift. The device tier cannot: cuDF has no Rust binding, so the tier is a second statement of the engine's semantics in another language. Every IR tag is therefore in exactly one of two sets. Translated, in SUPPORTED_OPS and exprs._HANDLERS: filter, project, aggregate, sort, distinct, limit, window, unnest, unpivot and row_id, with the expression vocabularies keyed beside them. Declined with a reason, in DECLINED_OPS and DECLINED_EXPRS: asof_join, range_join and sample are not translated, and the image, audio and geo expressions are Rust kernels with no dataframe equivalent to translate onto. There is no third state, because a tag in neither set fails test_gpu_vocabulary_contract, which makes a new operator a decision rather than an oversight. The plan then goes one way or the other whole: if every node translates it runs on the device under cuDF, one shard per device, and it is eligible only as a chain over a scan, a join of two chains, or a union of chains; if any node declines the whole plan runs on the CPU engine and returns the same rows more slowly. A decline costs time where an approximation would cost a wrong answer, which is why backend="gpu" is always safe to ask for.](/_static/diagrams/gpu_tier_decision.svg)

The translator is parameterized by dataframe library. It runs on cuDF on a GPU worker and on pandas in the test suite, against the CPU engine as the oracle, so the same code a device executes is checked on every commit without a device. That check is what surfaced the cases where a dataframe library's default quietly disagrees with the engine: a null group key is a group rather than a dropped row, the sum of an all-null group is null rather than `0.0`, a null predicate drops its row, `NaN` orders above every number rather than comparing false, `substr` is 1-based, `%` takes the sign of the dividend, and `round` breaks halves away from zero.

### Using more than one device

A single device's memory is the wrong ceiling for the queries a GPU is worth using for, so a chain is split across devices whenever its shape allows. Each device reads its own shard straight from storage. What happens to the shards afterwards depends on which of three shapes the chain is.

A chain with a **mergeable reducer** folds: each device reduces its shard, and the small per-group results are combined once.

Three reducers have a mergeable form. An `aggregate`, for the reductions whose partials fold: a mean is not itself mergeable, but the sum and count it is a ratio of are. A `distinct`, because deduplicating twice is deduplicating once. And a sort carrying a limit, because a global top-N is the top-N of the shards' top-Ns. Only the row-local operators (filter and project) may run *below* the reducer; everything else reads rows its shard does not have. Anything *above* it runs once on the folded result, which is what lets the ordinary analytical shape (group by, then sort, then limit) fan out at all.

`median`, `quantile`, `var`, `stddev` and `count-distinct` each need a group's whole value set, so a chain reducing with one of those stays on a single device. An aggregate that cannot shard is a scale ceiling; one that shards wrongly is a wrong number.

A **row-local** chain concatenates. Every shard's output is already its slice of the answer, in order, so reassembling the slices in shard order is the answer. That is why a filter over a very large scan isn't bounded by one device's memory, though it's still bounded by the driver's, since every surviving row is returned there.

A **join** splits its probe side, and every device reads the whole build side itself. That's correct only for the join types whose output is driven by left rows: inner, left, semi and anti. A `right` or `full` join must emit an unmatched build row exactly once and every shard sees the whole build side, so each would emit it. Whether the build side is small enough to give to every device isn't decided by the backend: the fan-out runs only when the planner already marked the join `broadcast`, so both backends answer that question the same way.

The decomposition is expressed as more plan IR (`plan/distribution/`) rather than as a second set of kernels, so partial and combine run through the same translator every other operator does, and the multi-device answer equals the single-device one by construction. The same module answers the optimizer's question. A plan that shards is bounded by its shard size rather than by one device's memory, which changes where Kyber routes it.

### Sharing a device between shards

The fan-out cuts several times more shards than there are devices, so each one is small, work balances across an uneven fleet, and a preempted shard's retry is a fraction of the query. A shard that then asks for a *whole* device undoes half of that: Ray runs one per device and queues the rest, so a fleet whose own shard count says each piece is a quarter of a device runs at a quarter of its capacity while every utilization counter reads full.

Each shard therefore asks for the fraction of a device it needs. The share is derived from the largest shard's estimated working set against one device's memory, rounded up to a packing quantum, and it is chosen from the *largest* shard rather than the average, because one fraction is granted to the whole fan-out and sizing it to the average is how the shard that most needed room is the one that doesn't get it. A broadcast join charges its replicated build side to every co-tenant, since four tasks on a device hold four copies of it rather than one between them.

Over-packing degrades rather than fails. A shard granted a share it turns out not to fit falls into the subdivision ladder below, exactly as an under-estimated shard always did, and its retry goes back with a whole device. Under-packing has no such ladder: the idle device stays idle, and nothing reports it.

Set `gpu_pack_shards` to `False` to keep the previous one-device-per-shard behavior, `gpu_task_fraction` to pin the share for a fleet the estimator can't see, `gpu_max_tasks_per_device` to cap co-tenancy, and `gpu_shard_expansion` for a chain that materializes more than one intermediate.

### Folding without holding every shard

The fan-out bounds *device* memory by dividing the input. The merge then has to avoid moving that bound to the host: concatenating every shard's output on the driver before combining a single row means a group-by over a million groups across a thousand shards materializes a billion rows in one process, and the shard count grows with the fleet, so the failure arrives precisely on the large clusters the fan-out exists for.

Partials are folded a wave at a time instead. A wave of `gpu_merge_wave` partials is combined, the result is kept, and the wave is discarded, so peak driver memory tracks the wave size and the group count rather than the shard count. The result is exact at any wave size, because the fold is associative and commutative over its own output: `plan/distribution/mergeable.py` carries the second form of each combine, the one that reads the columns the first application wrote. Anything above the fold, such as a mean's final division, runs exactly once. Set `gpu_merge_wave` to `0` to fold everything at once.

### When a device is lost, or too small

A fan-out that abandons the accelerated path because one shard failed is not much of a fan-out. Failures are handled where they happen:

- a shard that **did not fit** is subdivided and rerun on the device, halving further while it still does not fit. The shard count is fixed before the query runs, from an estimate, and estimates are wrong exactly where it matters: a skewed key, a wider row than the footer promised, a neighboring tenant on the device. Subdividing is exact because the stage is mergeable;
- a shard that failed for **any other reason** (a reclaimed spot node, a device that fell off the bus) is recomputed by the native CPU engine, which produces the identical mergeable partial. One dead device costs that shard's time rather than the query;
- a **straggler** gets a duplicate through the same backup barrier the CPU shuffle uses, and whichever copy lands first is kept.

A worker's input doesn't pass through the driver. Each task receives a partition descriptor (a manifest of splits with the projection and predicate already pushed into it) and reads from storage itself. The exception is a source that can't be split, such as an in-memory table, whose rows are on the driver by construction and are shipped from there.

## Keeping the device fed

The naive way to run a decode-then-inference pipeline is stage-at-a-time: decode the whole partition, then run the forward pass. The GPU idles through the entire decode.

`core/udf/stream.py` detects a linear `Scan -> map -> ... -> map` chain and runs it as a prefetch-pipelined stream, each stage on its own thread behind a bounded queue, so the CPU decode of morsel *k+1* overlaps the GPU forward of morsel *k*. Order is preserved by FIFO prefetch and in-order per-stage application, so the concatenated output is byte-identical to the non-overlapped run at any prefetch depth.

```text
   SEQUENTIAL STAGES                                942 img/s,  ~30% GPU
   ─────────────────
   CPU  ████████ decode the whole partition ████████
   GPU                                              ██ forward ██
                                                    ▲
                                       idle through the entire decode


   STAGE-OVERLAPPED                               2,504 img/s,   81% GPU
   ────────────────
   CPU  ██ dec m0 ██ ██ dec m1 ██ ██ dec m2 ██ ██ dec m3 ██
   GPU                ██ fwd m0 ██ ██ fwd m1 ██ ██ fwd m2 ██ ██ fwd m3 ██

   same hardware, same result. verified per batch, order preserved,
   single-node equal to distributed. the device just stops waiting.
```

This is an execution property of the engine rather than a feature of the inference operator, so any CPU-heavy chain feeding a compute stage inherits it, for any modality.

:::{warning}
A *higher* GPU utilization percentage isn't automatically better, and it cuts both ways, so it's worth stating plainly. A slower engine spreads the same GPU work over more wall-clock and reads as higher utilization. Throughput is the number that matters. Utilization only explains it.
:::

## Warm pools

A pool that respawns per execution reloads the model every time. Batcher keeps GPU inference pools warm across `collect()` calls within a session (`distributed.warm_inference_pools`, on by default), so the model loads once per *session*.

That's worth about 2x on iterative or repeated inference, and far more when the load dominates. A gpt2 FP16 load takes 7 to 10 s while generation takes about 1 s, so paying it once rather than per execution is most of the wall clock. Measured over 8xT4 with 2,048 prompts, Batcher generated at 814.8 prompt/s, finishing in 2.51 s with 100% text match.

Pools are keyed by UDF identity, healed when an actor dies to preemption, and freed at process exit or through `release_inference_pools()`. They are also freed on their own once a session goes quiet, after `distributed.warm_inference_idle_s` of no stage running on them, which is 120 seconds by default. Holding a device across a whole session is a deadlock rather than an optimization on a cluster with a second tenant, because a device has no oversubscription to fall back on: a finished query that keeps every GPU leaves the next pool pending forever, and nothing reports it. Back-to-back queries never wait for the rebuild, since taking the pool cancels the pending release, and `0` restores whole-session residency for a process that owns its cluster.

:::{important}
A warm pool only helps if the model is *loadable once*. That's why `map_batches(Model, num_gpus=1)` takes a class rather than a function. The class's `__init__` loads the model and `__call__` runs it. Passing a closure that loads the model per batch emits a `PerformanceWarning`, and it throws away the single biggest win on this page.
:::

## Batch sizing

There are two controllers, and they optimize different things. The following table compares them, and the tabs below explain how each works.

| | Latency controller | Throughput controller |
|---|---|---|
| For | online serving | offline batch |
| Optimizes | a per-batch latency setpoint | maximum rows/sec under a VRAM cap |
| Method | a PID over the *relative* latency error | a constrained hill-climb |
| Lives in | `ml/inference/pool.py` | `ml/autobatch.py` |

::::{tab-set}
:::{tab-item} Latency (online serving)
A PID over the *relative* per-batch latency error drives the batch size toward a latency setpoint. The live controller is `ml/inference/pool.py::_LatencyController`, which reads its gains from the shared `PIDConfig`. It's a port of `crates/bc-udf/src/batch_size.rs::BatchSizeController`, which states the same law in Rust. `bc-udf` isn't linked into `bc-py`, so the Rust controller isn't on a live path.

```rust
let error = (self.target_latency_ms - observed_latency_ms) / self.target_latency_ms;
self.integral = (self.integral + error).clamp(-INTEGRAL_CLAMP, INTEGRAL_CLAMP);
let derivative = error - self.prev_error;
let adjustment = (self.kp * error + self.ki * self.integral + self.kd * derivative)
    .clamp(-MAX_STEP_FRACTION, MAX_STEP_FRACTION);
self.current = (self.current * (1.0 + adjustment)).clamp(min, max);
```

The control law applies *multiplicatively* to the current size over the *relative* error, which makes it scale-free. It behaves the same at 100 rows and at 100,000, with a natural fixed point at `observed == target`. The integral clamp is anti-windup, and the step cap stops a single anomalous latency from swinging the size wildly.
:::

:::{tab-item} Throughput (offline batch)
A latency PID optimizes the wrong thing for a batch job. What you want is maximum rows/sec *subject to a VRAM cap*, which is a constrained hill-climb rather than a setpoint tracker. `ml/autobatch.py::ThroughputController` grows the batch by 1.5x while throughput keeps improving, shrinks by 0.7x on a VRAM breach, and settles at the best size seen. VRAM is a hard constraint with a *predictive* guard, so it grows only if the projected fraction stays under the cap and never has to hit an OOM to learn where the wall is. Given a hub and a model signature it warm-starts from the plateau a prior run learned, which changes only the starting size and never the result.
:::
::::

The PID's shipped gains come from {py:class}`PIDConfig <batcher.PIDConfig>`.

```python
from batcher.config import PIDConfig

pid = PIDConfig()
print(pid.kp, pid.ki, pid.kd, pid.integral_clamp, pid.max_step_fraction)
```

```text
0.4 0.05 0.1 5.0 0.5
```

:::{note}
`architecture.txt` describes a PID controller targeting a *GPU-utilization* setpoint of 80 to 90%, and an RL/PPO batch sizer. Neither exists. The PID targets per-batch **latency**, the GPU path uses the non-PID throughput hill-climb above, and utilization is measured but feeds `num_gpus` and in-flight-depth recommendations rather than a PID.
:::

## Zero config

The simplest possible call is where out-of-the-box utilization is won or lost.

```python
# docs: skip
import batcher as bt


class Classifier:
    def __init__(self):
        import torchvision, torch

        self.model = torchvision.models.resnet50(weights="DEFAULT").cuda().eval()

    def __call__(self, batch):
        import torch

        with torch.no_grad():
            return {"pred": self.model(batch["img"].cuda()).argmax(1).cpu().numpy()}


# No batch_size given. Batcher picks a VRAM-safe default.
ds = bt.read.images("s3://bucket/frames/", decode=True, size=(224, 224))
out = ds.map_batches(Classifier, num_gpus=1, batch_format="torch").collect()
```

Batcher starts the throughput hill-climb from a VRAM-safe 256 rows, streams it with stage overlap, and self-corrects on a CUDA OOM by halving the batch. That reaches 82% utilization at 2,451 img/s on 131k images across 8xT4, matching the hand-tuned `batch_size=128` path at 2,504 img/s and 81%, with no knobs.

## Autocast, and why it probes

Half precision isn't a free win. A conv or matmul forward gets tensor cores. An autoregressive generation loop is launch-bound and memory-bound and gets nothing, or worse. Half precision also isn't bit-identical, so applying it where it doesn't pay is a silent output change bought for nothing.

So `ml/gpu.py::autocast_call` doesn't blindly wrap. It times FP32 against autocast on a 64-row probe (`_AUTOCAST_PROBE_ROWS`), taking the best of three timings with CUDA synchronized so GPU work is actually measured, and keeps autocast only if the speedup clears `_AUTOCAST_MIN_SPEEDUP`, which is 1.15. The verdict is cached per callable, and any failure during the probe returns the output-preserving FP32 path. `torch.compile` follows the same principle in `ml/inference/pipelines.py` and is applied only to models containing a `Conv2d`, because it measured 0.92x on a small text transformer where dynamic sequence lengths force per-shape recompiles.

## Measuring the device

`ml/gpu.py` is the vendor-neutral measurement layer. `detect_backend()` returns `cuda`, `rocm`, `xpu`, `mps`, `tpu`, or `cpu`, and utilization sampling dispatches through NVML, ROCm SMI, or the XPU equivalent. On MPS and TPU there's no utilization API, so the loop is a no-op rather than a guess.

```python
from batcher.ml.gpu import detect_backend

print(detect_backend())
```

```text
cpu
```

Device attribution honors `CUDA_VISIBLE_DEVICES` and the ROCm equivalents, so a Ray-pinned actor averages only *its* devices rather than the whole node's. Getting this wrong makes a one-GPU actor on an eight-GPU node report 12% utilization when it's saturated.

The measurements feed a learned loop. Per-model peak VRAM and utilization are recorded in the `MetadataHub` and consumed by `recommend_num_gpus`, which packs two models onto one device below 50% utilization, by `recommend_inflight_depth`, which gives a starved device more submit-ahead slots, and by `max_actors_per_gpu`.

## Dirty data

Real corpora contain rows that fail to decode. `core/udf/call.py::_resilient_call` bisects a failing batch to isolate the bad rows and drops them against the `max_errored_rows` budget, and halves the batch on a CUDA OOM before retrying. Tolerance is per row rather than per block, so with about 1% corrupt rows injected across 200k, Batcher retains 198,000 rows. One bad image costs you one image, not the job.

## Checking the device against the CPU engine

The relational backend is a second statement of the engine's semantics in another language against another library, so it is checked against the CPU engine rather than trusted. Every query in a suite runs on both backends in one process, and the two results are compared on column names and column types exactly, on non-float values exactly, and on floats within the tolerance that reassociation allows. A difference is a defect, never a decline.

That comparison is the only thing that finds this tier's characteristic failure. Every defect it has shipped has been a column *type* with correct values, which no value comparison can see: a DATE returning `timestamp[ms]`, an integer `abs` widening to double, an empty result losing its string columns to `null`, and `COUNT(DISTINCT ...)` returning `int32`. The translator's own tests run on pandas, so a cuDF-only behavior reaches a cluster before it reaches a test.

There are two oracles behind that comparison, they cost very different things, and only one of them is on for every run.

![The device tier's two oracles, and what each one can see. A device result is a `pyarrow.Table` from cuDF rather than from the engine, and it is checked before it is returned. The schema contract, `enforce_schema_contract`, is on for every device run: it reads a field list with no rows, no second execution and no device, holding the result against the engine's own `available_schema`, and it sees types. The shadow re-run, `shadow_verify`, is off by default behind `distributed.gpu_shadow_verify`: it re-runs the plan on the CPU engine and compares schema first and then values, and it is the only oracle for values. A result both agree on stands and is returned unchanged, so a verified run differs from an unverified one only in cost; if the CPU oracle itself raises, that is reported as verifying nothing and never as a pass. A difference is a defect and never a decline: the CPU engine answers instead, and it is reported through `note_gpu_failure`, which logs at warning level, rather than through `note_suppressed`, which is for declines, because the tier's contract is that a device changes where a plan runs and never what it computes. Every defect on record has been a column type with correct values: a DATE returning `timestamp[ms]` on a device where pandas gave `date32`, an integer `abs` widening to double, and an empty cuDF string column arriving as `null`.](/_static/diagrams/gpu_shadow_verify.svg)

Measured on four `1xT4` workers with 8 CPUs each, warm, against Batcher's own CPU engine on the same data. TPC-H is scale factor 1 and ClickBench is an 8 million row subset of `hits`, both read from the same Parquet by both backends. These are Batcher against Batcher, not against another engine.

| Query | Shape | CPU engine | GPU | Speedup |
|---|---|--:|--:|--:|
| `cb-q34` | `GROUP BY URL`, high cardinality | 187.2 s | **13.02 s** | **14.4x** |
| `tpch-q1` | Filter to group-by to sort | 6.20 s | **0.43 s** | **14.4x** |
| `cb-q29` | 90 summed projections | 2.07 s | **0.24 s** | **8.6x** |
| `cb-q35` | Group by four derived integer keys | 29.59 s | **4.83 s** | **6.1x** |
| `cb-q26` | Filter, sort, limit on strings | 3.73 s | **0.75 s** | **5.0x** |
| `cb-q05` | `COUNT(DISTINCT SearchPhrase)` | 5.49 s | **1.30 s** | **4.2x** |

The first row is the interesting one. High-cardinality string grouping is where the CPU engine is weakest, so it is also where a device pays for itself most, and the two shapes at the top of this table are the ones a warehouse query hits most often.

The GPU wins 9 of the 43 ClickBench queries outright. It loses the rest to fixed cost rather than to arithmetic, which is why the requirement below about the cluster image matters more than any tuning knob on this page.

Run the same comparison yourself with `distributed.gpu_shadow_verify=True`, which re-runs each device result on the CPU engine and reports where they disagree. It doubles the work, so leave it off outside verification.

## Requirements and limitations

**Install cuDF in the cluster image.** When a worker cannot import cuDF, Batcher ships it to every GPU task as a Ray `runtime_env` pip requirement instead. Resolving that environment costs about 22 seconds on a worker that does not already have it, against 0.23 seconds for the same task without it, and it is paid again whenever a GPU task starts on a fresh worker. On a four-worker cluster that is enough to dominate a relational query and make the device look slower than the CPU engine when it is roughly ten times faster. Baking cuDF into the image removes the requirement entirely, because the runtime environment is then never attached.

The ceiling is arithmetic. A *single, maximally large, compute-bound* job runs at the device's FLOPs and no scheduling changes that: one T4 sustains about 400 img/s at 100% utilization on ResNet-50, and eight actors reach about 3,200 with no parallel penalty. Every mechanism described above works on the pipeline around the model, so once the pipeline isn't the bottleneck, going faster means fewer FLOPs through FP16 or quantization.

A GPU `fn` never runs in a process pool, because it has to keep a single process and CUDA context. The GIL is therefore a real constraint on a GPU stage whose Python glue is heavy.

Gang placement for an inference pool is best-effort. The actor pool reserves a placement group sized to its autoscaling ceiling, and a stage flagged `gpu_collective` in `plan/resource/bounds.py` gets `STRICT_PACK` so its actors are co-located, as {doc}`the wires between GPUs </architecture/deep-dives/distribution/gpu-fabric>` describes. When the cluster can't grant the reservation in time, the pool logs a warning and falls back to default scheduling, which runs correctly but may place actors unevenly. The relational fan-out described above uses many devices as independent single-device tasks that share nothing, which is a weaker requirement than a collective.

The relational backend has no device-to-device shuffle, so it distributes only what the algebra above covers. A chain that reduces with `median`, `quantile`, `var`, `stddev` or `count-distinct` runs on one device, because each needs a group's whole value set. A `right` or `full` join runs on one device, because broadcasting the build side would duplicate its unmatched rows, and a join the planner did not mark `broadcast` runs on one device because its build side does not fit. All are scale ceilings rather than wrong answers, and all would lift with a key-partitioning exchange between devices.

A sharded relational result returns to the driver through the Ray object store, one message per shard, and the driver holds all of it. Two things follow. Splitting a *reducing* chain moves only one row per group per shard, so the driver sees a small result however large the input was. Splitting a **row-local** chain moves every surviving row: the ceiling stops being one device's memory and becomes the driver's, which is a better ceiling but not an absent one. And it is a deviation from the data-plane rule that bulk Arrow travels by Arrow Flight rather than as Ray objects, inherited from the original single-task backend and widened by sharding. Lifting it is the same work as the device-to-device exchange above.

GPU tensors move between stages as Arrow through host memory. There's no device-to-device transport. That's a deliberate consequence of the Arrow-only invariant, and `docs/architecture/internals/rfcs/rfc-gpu-transport.md` proposes changing it. That document is an in-tree proposal, not a description of shipped behavior.

## Code map

Each concern below maps to the file that owns it, so the device placement and batching rules on this page can be read directly:

| Concern | File |
|---|---|
| Stage-overlapped streaming, chain detection | [`python/batcher/core/udf/stream.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/stream.py) |
| UDF dispatch | [`python/batcher/core/udf/execute.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/execute.py) |
| OOM halving and dirty-row bisection | [`python/batcher/core/udf/call.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/call.py) |
| Threads vs processes policy | [`python/batcher/core/udf/strategy.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/strategy.py) |
| Distributed actor pools, warm pools | [`python/batcher/dist/executors/map.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/executors/map.py) |
| Latency PID | [`python/batcher/ml/inference/pool.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/ml/inference/pool.py), mirrored in [`crates/bc-udf/src/batch_size.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-udf/src/batch_size.rs) |
| Throughput hill-climb | [`python/batcher/ml/autobatch.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/ml/autobatch.py) |
| Device detection, utilization, VRAM | [`python/batcher/ml/gpu.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/ml/gpu.py) |
| GPU-vs-CPU backend policy | [`python/batcher/kyber/gpu/policy.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/kyber/gpu/policy.py) |
| GPU relational backend routing | [`python/batcher/api/terminal/gpu_backend/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/api/terminal/gpu_backend) |
| Plan and expression translation to cuDF | [`python/batcher/core/gpu_plan/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/core/gpu_plan) |
| Mergeable split, shared by the optimizer and the backend | [`python/batcher/plan/distribution/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/plan/distribution) |
| Multi-device fan-out, shard recovery, worker-side reads | [`python/batcher/dist/gpu/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/dist/gpu) |
| Per-shard device share, Ray options for a fan-out | [`python/batcher/dist/gpu/resources.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/gpu/resources.py) |
| Co-tenancy admission, MIG preference, health derate | [`python/batcher/carbonite/accel/fractional.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/carbonite/accel/fractional.py) |
| The packing quanta both of those round against | [`python/batcher/_internal/device_share.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/_internal/device_share.py) |
| cuDF and torch scatter-reduce kernels | [`python/batcher/core/gpu_transform.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/gpu_transform.py) |

## See also

- {doc}`The wires between GPUs </architecture/deep-dives/distribution/gpu-fabric>`: the interconnect facts behind multi-GPU placement.
- {doc}`Architecture </architecture/index>`: why the GPU paths live in Python and not in the crates.
- {doc}`Execution engine </architecture/internals/execution>`: the UDF stage this pipelines.
- `docs/architecture/internals/rfcs/rfc-gpu-transport.md`, an in-repo RFC rather than a site page: the device-to-device transport this page doesn't have.
- {doc}`GPU guide </ml/inference/gpu>`: the knobs, from a user's side.
- {doc}`ML guide </ml/index>`: how to write these pipelines.
- {doc}`Batch inference tutorial </getting-started/tutorials/ml/batch-inference>`: the pipeline this page is underneath.
- {doc}`AI and GPU benchmarks </benchmarks/results/ai-and-gpu>`: the numbers on this page, in context.
- {doc}`Multimodal ingest benchmarks </benchmarks/results/multimodal-ingest>`: the decode side of the same pipeline.
- {doc}`Tensor columns </architecture/deep-dives/memory/tensor-columns>`: what crosses into the model.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: how the actors get placed.
