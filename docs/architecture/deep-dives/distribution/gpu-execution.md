# GPU execution

This page describes the two paths on which Batcher runs work on a GPU, and the scheduling that keeps the device busy on each.

A GPU idles as easily as a CPU core and costs far more while it does, so the job of an engine running GPU work is to keep the device fed. Most of that is scheduling rather than kernels.

Both GPU paths are Python. The `bc-*` crates contain no GPU code at all, which follows from the Arrow-only data-plane contract. The two paths differ in what they put on the device and in who dispatches them.

| Path | What runs on the device | Where |
|---|---|---|
| GPU relational backend ({py:meth}`collect(backend="gpu") <batcher.Dataset.collect>`) | cuDF dataframe ops, with a torch scatter-reduce fallback | `core/gpu_plan/` translates, `dist/gpu/` schedules, in Ray tasks with `num_gpus=1` |
| GPU inference stage (`map_batches(..., num_gpus=...)`) | the user's torch model | a Python Ray actor |

The relational backend is an opt-in accelerator for relational shapes. An unsupported shape, an OOM, or a GPU-less cluster falls back to the CPU engine, so `backend="gpu"` is always safe to request. On a machine with no GPU the same rows come back from the CPU engine:

```python
import batcher as bt

ds = bt.from_pydict({"k": [1, 1, 2], "v": [10, 20, 30]})
out = ds.group_by("k").agg(s=bt.col("v").sum()).sort("k").collect(backend="gpu")
print(out.to_pydict())  # {'k': [1, 2], 's': [30, 30]}
print(bt.Config().distributed.gpu_require)  # False
```

Set `distributed.gpu_require=True` to make an explicit `backend="gpu"` raise with the reason instead.

## The relational backend

`collect(backend="gpu")` walks a plan's operator IR and replays it on a cuDF DataFrame, one case per operator and one per expression. A plan reaches the device only when *every* node in it translates, so coverage is what decides how much of a real query is accelerated rather than a list of features.

| Layer | What translates |
|---|---|
| Operators | filter, project, group-by aggregate, sort, distinct, limit, window, unnest, unpivot, row id, plus equi/semi/anti joins of two chains and unions of chains |
| Aggregates | sum, count, count(\*), mean, min, max, var, stddev, median, quantile, count-distinct, product, bool-and, bool-or |
| Window functions | row_number, rank, dense_rank, percent_rank, cume_dist, ntile, lag, lead, first_value, last_value, nth_value, forward and backward fill, and the aggregates over a whole partition, a running frame, or a moving frame |
| Expressions | arithmetic, comparison, boolean, cast, `CASE`, coalesce, nullif, greatest, least, `IN`, null and NaN tests, twenty unary math functions, and the string and date vocabularies |

Anything outside that set is *declined* rather than approximated, and the stage runs on the CPU engine instead. A fallback costs time; an approximation would cost a wrong answer.

![Why the device tier needs machinery no other tier needs, and what that machinery buys. Every other execution tier consumes the same Rust bc_expr::Expr, so there is one definition of what a shape means and it cannot drift. The device tier cannot: cuDF has no Rust binding, so the tier is a second statement of the engine's semantics in another language. Every IR tag is therefore in exactly one of two sets. Translated, in SUPPORTED_OPS and exprs._HANDLERS: filter, project, aggregate, sort, distinct, limit, window, unnest, unpivot and row_id, with the expression vocabularies keyed beside them. Declined with a reason, in DECLINED_OPS and DECLINED_EXPRS: asof_join, range_join and sample are not translated, and the image, audio and geo expressions are Rust kernels with no dataframe equivalent to translate onto. There is no third state, because a tag in neither set fails test_gpu_vocabulary_contract, which makes a new operator a decision rather than an oversight. The plan then goes one way or the other whole: if every node translates it runs on the device under cuDF, one shard per device, and it is eligible only as a chain over a scan, a join of two chains, or a union of chains; if any node declines the whole plan runs on the CPU engine and returns the same rows more slowly. A decline costs time where an approximation would cost a wrong answer, which is why backend="gpu" is always safe to ask for.](/_static/diagrams/gpu_tier_decision.svg)

The translator is parameterized by dataframe library. It runs on cuDF on a GPU worker and on pandas in the test suite, against the CPU engine as the oracle, so the translation is checked on every commit without a device. It follows the engine's semantics where a dataframe library's default differs: a null group key is a group, the sum of an all-null group is null, a null predicate drops its row, `NaN` orders above every number, `substr` is 1-based, `%` takes the sign of the dividend, and `round` breaks halves away from zero.

### Using more than one device

A chain is split across devices whenever its shape allows, and each device reads its own shard straight from storage. What happens to the shards depends on the chain's shape:

- **A mergeable reducer folds.** Each device reduces its shard and the small per-group results combine once. That covers an `aggregate` whose partials fold (a mean via its sum and count), a `distinct`, and a sort with a limit. Filter and project may run below the reducer, and anything above it runs once on the folded result, so group-by, then sort, then limit fans out.
- **A row-local chain concatenates.** Each shard's output is its slice of the answer, in shard order.
- **A join splits its probe side**, and every device reads the whole build side. That applies to inner, left, semi and anti joins the planner already marked `broadcast`.

The decomposition is expressed as more plan IR (`plan/distribution/`) rather than a second set of kernels, so partial and combine run through the same translator and the multi-device answer equals the single-device one by construction. Kyber reads the same module, since a plan that shards is bounded by its shard size rather than one device's memory.

:::{dropdown} Sharing a device between shards
The fan-out cuts several times more shards than there are devices, so each one is small, work balances across an uneven fleet, and a preempted shard's retry is a fraction of the query. Each shard therefore asks for the fraction of a device it needs. The share is derived from the largest shard's estimated working set against one device's memory, rounded up to a packing quantum, and it is chosen from the *largest* shard rather than the average, because one fraction is granted to the whole fan-out and sizing it to the average is how the shard that most needed room is the one that doesn't get it. A broadcast join charges its replicated build side to every co-tenant, since four tasks on a device hold four copies of it rather than one between them.

A shard granted a share it turns out not to fit falls into the subdivision ladder below, and its retry goes back with a whole device.

Set `gpu_pack_shards` to `False` for one device per shard, `gpu_task_fraction` to pin the share for a fleet the estimator can't see, `gpu_max_tasks_per_device` to cap co-tenancy, and `gpu_shard_expansion` for a chain that materializes more than one intermediate.
:::

:::{dropdown} Folding without holding every shard
The fan-out bounds *device* memory by dividing the input, and the merge keeps that bound off the host. Partials are folded a wave at a time. A wave of `gpu_merge_wave` partials is combined, the result is kept, and the wave is discarded, so peak driver memory tracks the wave size and the group count rather than the shard count. The result is exact at any wave size, because the fold is associative and commutative over its own output: `plan/distribution/mergeable.py` carries the second form of each combine, the one that reads the columns the first application wrote. Anything above the fold, such as a mean's final division, runs exactly once. Set `gpu_merge_wave` to `0` to fold everything at once.
:::

### When a device is lost, or too small

Failures are handled per shard, so one bad shard never abandons the accelerated path:

- a shard that **did not fit** is subdivided and rerun on the device, halving further while it still does not fit. Subdividing is exact because the stage is mergeable;
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

:::{note}
Throughput is the number that matters. Utilization explains it, and a slower engine spreading the same GPU work over more wall-clock reads as higher utilization.
:::

## Warm pools

A pool that respawns per execution reloads the model every time. Batcher keeps GPU inference pools warm across `collect()` calls within a session (`distributed.warm_inference_pools`, on by default), so the model loads once per *session*.

That's worth about 2x on iterative or repeated inference, and more when the load dominates. A gpt2 FP16 load takes 7 to 10 s while generation takes about 1 s, so paying it once rather than per execution is most of the wall clock. Measured over 8xT4 with 2,048 prompts, Batcher generated at 814.8 prompt/s, finishing in 2.51 s with 100% text match.

Pools are keyed by UDF identity, healed when an actor dies to preemption, and freed at process exit or through `release_inference_pools()`. They are also released after `distributed.warm_inference_idle_s` of no stage running on them, so an idle session doesn't hold GPUs another tenant needs. Back-to-back queries never wait for the rebuild, since taking the pool cancels the pending release, and `0` keeps the pool for the whole session:

```python
dist = bt.Config().distributed
print(dist.warm_inference_pools, dist.warm_inference_idle_s)  # True 120.0
```

:::{important}
A warm pool only helps if the model is *loadable once*. That's why `map_batches(Model, num_gpus=1)` takes a class rather than a function. The class's `__init__` loads the model and `__call__` runs it. Passing a closure that loads the model per batch emits a `PerformanceWarning`.
:::

## Batch sizing

There are two controllers, and they optimize different things:

| | Latency controller | Throughput controller |
|---|---|---|
| For | online serving | offline batch |
| Optimizes | a per-batch latency setpoint | maximum rows/sec under a VRAM cap |
| Method | a PID over the *relative* latency error | a constrained hill-climb |
| Lives in | `ml/inference/pool.py` | `ml/autobatch.py` |

:::::{dropdown} How each controller works
::::{tab-set}
:::{tab-item} Latency (online serving)
A PID over the *relative* per-batch latency error drives the batch size toward a latency setpoint. The controller is `ml/inference/pool.py::_LatencyController`, which reads its gains from the shared `PIDConfig`.

```python
# docs: skip
error = (self._target - observed_ms) / self._target
self._integral = clamp(self._integral + error, -pid.integral_clamp, pid.integral_clamp)
derivative = error - self._prev
raw = pid.kp * error + pid.ki * self._integral + pid.kd * derivative
adjustment = clamp(raw, -pid.max_step_fraction, pid.max_step_fraction)
self._cur = min(float(self._max), max(float(self._min), self._cur * (1.0 + adjustment)))
```

The control law applies *multiplicatively* to the current size over the *relative* error, which makes it scale-free. It behaves the same at 100 rows and at 100,000, with a natural fixed point at `observed == target`. The integral clamp is anti-windup, and the step cap stops a single anomalous latency from swinging the size wildly.
:::

:::{tab-item} Throughput (offline batch)
A latency PID optimizes the wrong thing for a batch job. What you want is maximum rows/sec *subject to a VRAM cap*, which is a constrained hill-climb rather than a setpoint tracker. `ml/autobatch.py::ThroughputController` grows the batch by 1.5x while throughput keeps improving, shrinks by 0.7x on a VRAM breach, and settles at the best size seen. VRAM is a hard constraint with a *predictive* guard, so it grows only if the projected fraction stays under the cap and never has to hit an OOM to learn where the wall is. Given a hub and a model signature it warm-starts from the plateau a prior run learned, which changes only the starting size and never the result.
:::
::::
:::::

The PID's shipped gains come from {py:class}`PIDConfig <batcher.PIDConfig>`.

```python
from batcher.config import PIDConfig

pid = PIDConfig()
print(pid.kp, pid.ki, pid.kd, pid.integral_clamp, pid.max_step_fraction)  # 0.4 0.05 0.1 5.0 0.5
```

The PID targets per-batch latency; GPU utilization is measured and feeds the `num_gpus` and in-flight-depth recommendations rather than a controller.

## Zero config

The simplest call gets the batch sizing, stage overlap and OOM recovery with no knobs:

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

A conv or matmul forward gets tensor cores from half precision; a launch-bound generation loop doesn't, and half precision isn't bit-identical. So `ml/gpu.py::autocast_call` times FP32 against autocast on a 64-row probe (`_AUTOCAST_PROBE_ROWS`), best of three with CUDA synchronized, and keeps autocast only if the speedup clears `_AUTOCAST_MIN_SPEEDUP` (1.15). The verdict is cached per callable, and a probe failure keeps FP32. `torch.compile` in `ml/inference/pipelines.py` is applied to models containing a `Conv2d`, where it pays.

## Measuring the device

`ml/gpu.py` is the vendor-neutral measurement layer. `detect_backend()` returns `cuda`, `rocm`, `xpu`, `mps`, `tpu`, or `cpu`, and utilization sampling dispatches through NVML, ROCm SMI, or the XPU equivalent. On MPS and TPU there's no utilization API, so the loop is a no-op rather than a guess.

```python
from batcher.ml.gpu import detect_backend

print(detect_backend())
```

```text
cpu
```

Device attribution honors `CUDA_VISIBLE_DEVICES` and the ROCm equivalents, so a Ray-pinned actor averages only *its* devices rather than the whole node's.

The measurements feed a learned loop. Per-model peak VRAM and utilization are recorded in the `MetadataHub` and consumed by `recommend_num_gpus`, which packs two models onto one device below 50% utilization, by `recommend_inflight_depth`, which gives a starved device more submit-ahead slots, and by `max_actors_per_gpu`.

## Dirty data

Real corpora contain rows that fail to decode. `core/udf/call.py::_resilient_call` bisects a failing batch to isolate the bad rows and drops them against the `max_errored_rows` budget, and halves the batch on a CUDA OOM before retrying. Tolerance is per row rather than per block, so with about 1% corrupt rows injected across 200k, Batcher retains 198,000 rows. One bad image costs you one image, not the job.

## Checking the device against the CPU engine

The relational backend is a second statement of the engine's semantics in another language against another library, so it is checked against the CPU engine rather than trusted. Two oracles do it. The schema contract runs on every device result for the price of a field-list walk, holding column names and types to the engine's own static analysis. The shadow re-run, `distributed.gpu_shadow_verify=True`, re-executes on the CPU engine and compares values too. A difference is a defect: the CPU engine answers instead.

```python
print(bt.Config().distributed.gpu_shadow_verify)  # False
```

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

High-cardinality string grouping and filter-group-sort pipelines, the shapes at the top of the table, gain the most. Run the same comparison yourself with `distributed.gpu_shadow_verify=True`, which doubles the work, so leave it off outside verification.

## Practical limits

- **Install cuDF in the cluster image.** Otherwise Batcher ships it to each GPU task as a Ray `runtime_env` pip requirement, which costs about 22 seconds on a fresh worker against 0.23 seconds without it. Baking cuDF into the image removes that cost entirely.
- **Model FLOPs set the ceiling.** One T4 sustains about 400 img/s at 100% utilization on ResNet-50, and eight actors reach about 3,200 with no parallel penalty. Past that, going faster means fewer FLOPs, through FP16 or quantization.
- **One process per GPU stage.** A GPU `fn` keeps a single process and CUDA context, so heavy Python glue in a GPU stage is bound by the GIL.
- **Gang placement is best-effort.** An inference pool reserves a placement group sized to its autoscaling ceiling, and a `gpu_collective` stage gets `STRICT_PACK`, as {doc}`the wires between GPUs </architecture/deep-dives/distribution/gpu-fabric>` describes. If the reservation can't be granted in time, the pool logs a warning and uses default scheduling.
- **Single-device shapes.** The relational backend fans out what the mergeable algebra covers. `median`, `quantile`, `count-distinct`, `var` and `stddev` reducers, `right` and `full` joins, and joins the planner didn't mark `broadcast` run on one device.
- **Results return through the driver.** A sharded relational result returns via the Ray object store, one message per shard. A reducing chain moves one row per group per shard; a row-local chain moves every surviving row, so the driver's memory is its ceiling.
- **Host-memory hand-off.** GPU tensors move between stages as Arrow through host memory, per the Arrow-only invariant. `docs/architecture/internals/rfcs/rfc-gpu-transport.md` is an in-tree proposal for device-to-device transport.

## Code map

Each concern below maps to the file that owns it, so the device placement and batching rules on this page can be read directly:

| Concern | File |
|---|---|
| Stage-overlapped streaming, chain detection | [`python/batcher/core/udf/stream.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/stream.py) |
| UDF dispatch | [`python/batcher/core/udf/execute.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/execute.py) |
| OOM halving and dirty-row bisection | [`python/batcher/core/udf/call.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/call.py) |
| Threads vs processes policy | [`python/batcher/core/udf/strategy.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/udf/strategy.py) |
| Distributed actor pools, warm pools | [`python/batcher/dist/executors/map.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/executors/map.py) |
| Latency PID | [`python/batcher/ml/inference/pool.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/ml/inference/pool.py) |
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
