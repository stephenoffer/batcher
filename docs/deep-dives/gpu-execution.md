# GPU execution

This page describes the two paths on which Batcher runs work on a GPU, and the scheduling that keeps the device busy on each.

A GPU costs roughly forty times what a CPU core costs and idles just as easily. The whole job of an engine running GPU work is to keep the device fed, and almost every way of failing at that is a *scheduling* failure rather than a kernel failure.

Both GPU paths are Python. The `bc-*` crates contain no GPU code at all, which follows from the Arrow-only data-plane contract. The two paths differ in what they put on the device and in who dispatches them.

| Path | What runs on the device | Where |
|---|---|---|
| GPU relational backend (`collect(backend="gpu")`) | cuDF dataframe ops, with a torch scatter-reduce fallback | `core/gpu_plan/` translates, `dist/gpu/` schedules, in Ray tasks with `num_gpus=1` |
| GPU inference stage (`map_batches(..., num_gpus=...)`) | the user's torch model | a Python Ray actor |

The second is the larger workload. The first is an opt-in accelerator for relational shapes, described in the next section. An unsupported shape, an OOM, or a GPU-less cluster falls back to the CPU engine, so `backend="gpu"` is always safe to request.

## The relational backend

`collect(backend="gpu")` walks a plan's operator IR and replays it on a cuDF DataFrame, one case per operator and one per expression. A plan reaches the device only when *every* node in it translates, so coverage is what decides how much of a real query is accelerated rather than a list of features.

| Layer | What translates |
|---|---|
| Operators | filter, project, group-by aggregate, sort, distinct, limit, window, equi/semi/anti join, union |
| Aggregates | sum, count, count(\*), mean, min, max, var, stddev, median, quantile, count-distinct, product, bool-and, bool-or |
| Window functions | row_number, rank, dense_rank, percent_rank, cume_dist, ntile, lag, lead, first_value, last_value, nth_value, forward and backward fill, and the aggregates over a whole partition, a running frame, or a moving frame |
| Expressions | arithmetic, comparison, boolean, cast, `CASE`, coalesce, nullif, greatest, least, `IN`, null and NaN tests, twenty unary math functions, and the string and date vocabularies |

Anything outside that set is *declined* rather than approximated, and the stage runs on the CPU engine instead. That distinction is the whole safety argument for the backend: a fallback costs time, and an approximation costs a wrong answer.

The translator is parameterized by dataframe library. It runs on cuDF on a GPU worker and on pandas in the test suite, against the CPU engine as the oracle, so the same code a device executes is checked on every commit without a device. That check is what surfaced the cases where a dataframe library's default quietly disagrees with the engine: a null group key is a group rather than a dropped row, the sum of an all-null group is null rather than `0.0`, a null predicate drops its row, `NaN` orders above every number rather than comparing false, `substr` is 1-based, `%` takes the sign of the dividend, and `round` breaks halves away from zero.

### Using more than one device

A single device's memory is the wrong ceiling for the queries a GPU is worth using for. A chain with a **mergeable reducer** is split instead: each device reads its own shard straight from storage and reduces it, and the small per-group results are folded once.

Three reducers have a mergeable form. An `aggregate`, for the reductions whose partials fold: a mean is not itself mergeable, but the sum and count it is a ratio of are. A `distinct`, because deduplicating twice is deduplicating once. And a sort carrying a limit, because a global top-N is the top-N of the shards' top-Ns. Only the row-local operators (filter and project) may run *below* the reducer; everything else reads rows its shard does not have. Anything *above* it runs once on the folded result, which is what lets the ordinary analytical shape (group by, then sort, then limit) fan out at all.

`median`, `quantile`, `var`, `stddev` and `count-distinct` each need a group's whole value set, so a chain reducing with one of those stays on a single device. An aggregate that cannot shard is a scale ceiling; one that shards wrongly is a wrong number.

The decomposition is expressed as more plan IR (`plan/distribution/`) rather than as a second set of kernels, so partial and combine run through the same translator every other operator does, and the multi-device answer equals the single-device one by construction. The same module answers the optimizer's question. A plan that shards is bounded by its shard size rather than by one device's memory, which changes where Kyber routes it.

### When a device is lost, or too small

A fan-out that abandons the accelerated path because one shard failed is not much of a fan-out. Failures are handled where they happen:

- a shard that **did not fit** is subdivided and rerun on the device, halving further while it still does not fit. The shard count is fixed before the query runs, from an estimate, and estimates are wrong exactly where it matters: a skewed key, a wider row than the footer promised, a neighboring tenant on the device. Subdividing is exact because the stage is mergeable;
- a shard that failed for **any other reason** (a reclaimed spot node, a device that fell off the bus) is recomputed by the native CPU engine, which produces the identical mergeable partial. One dead device costs that shard's time rather than the query;
- a **straggler** gets a duplicate through the same backup barrier the CPU shuffle uses, and whichever copy lands first is kept.

No worker's input passes through the driver. Each task receives a partition descriptor (a manifest of splits with the projection and predicate already pushed into it) and reads from storage itself. The second path is the entire ML batch-inference workload.

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

Pools are keyed by UDF identity, healed when an actor dies to preemption, and freed at process exit or through `release_inference_pools()`.

:::{important}
A warm pool only helps if the model is *loadable once*. That's why `map_batches(Model, num_gpus=1)` takes a class rather than a function. The class's `__init__` loads the model and `__call__` runs it. Passing a closure that loads the model per batch emits a `PerformanceWarning`, and it throws away the single biggest win on this page.
:::

## Batch sizing

There are two controllers and they optimize different things. Conflating them is the mistake the design avoids. The table names each one, then the tabs below explain how each works.

| | Latency controller | Throughput controller |
|---|---|---|
| For | online serving | offline batch |
| Optimizes | a per-batch latency setpoint | maximum rows/sec under a VRAM cap |
| Method | a PID over the *relative* latency error | a constrained hill-climb |
| Lives in | `crates/bc-udf/src/batch_size.rs`, `ml/inference.py` | `ml/autobatch.py` |

::::{tab-set}
:::{tab-item} Latency (online serving)
A PID over the *relative* per-batch latency error drives the batch size toward a latency setpoint. It's implemented identically in `crates/bc-udf/src/batch_size.rs::BatchSizeController` and `ml/inference.py::_LatencyController`, and shipped to Rust as `EngineConfig` so the two can't drift.

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

The PID's shipped gains come from `PIDConfig`.

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
out = ds.ml.map_batches(Classifier, num_gpus=1, batch_format="torch").collect()
```

Batcher starts the throughput hill-climb from a VRAM-safe 256 rows, streams it with stage overlap, and self-corrects on a CUDA OOM by halving the batch. That reaches 82% utilization at 2,451 img/s on 131k images across 8xT4, matching the hand-tuned `batch_size=128` path at 2,504 img/s and 81%, with no knobs.

## Autocast, and why it probes

Half precision isn't a free win. A conv or matmul forward gets tensor cores. An autoregressive generation loop is launch-bound and memory-bound and gets nothing, or worse. Half precision also isn't bit-identical, so applying it where it doesn't pay is a silent output change bought for nothing.

So `ml/gpu.py::autocast_call` doesn't blindly wrap. It times FP32 against autocast on a 64-row probe (`_AUTOCAST_PROBE_ROWS`), taking the best of three timings with CUDA synchronized so GPU work is actually measured, and keeps autocast only if the speedup clears `_AUTOCAST_MIN_SPEEDUP`, which is 1.15. The verdict is cached per callable, and any failure during the probe returns the output-preserving FP32 path. `torch.compile` follows the same principle and is applied only to models containing a `Conv2d`, because it measured 0.92x on a small text transformer where dynamic sequence lengths force per-shape recompiles.

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

## Requirements and limitations

The ceiling is arithmetic. A *single, maximally large, compute-bound* job runs at the device's FLOPs and no scheduling changes that: one T4 sustains about 400 img/s at 100% utilization on ResNet-50, and eight actors reach about 3,200 with no parallel penalty. Every mechanism described above works on the pipeline around the model, so once the pipeline isn't the bottleneck, going faster means fewer FLOPs through FP16 or quantization.

A GPU `fn` never runs in a process pool, because it has to keep a single process and CUDA context. The GIL is therefore a real constraint on a GPU stage whose Python glue is heavy.

Multi-GPU collective placement doesn't work. `SchedulingEnvelope.gpu_collective` and `placement_strategy` exist in `plan/resource.py` and are read by `dist/executors/ray_runtime/scheduling.py`, but the actor pool sets only `num_gpus` and `accelerator_type` and there's no placement group behind them. This is about *inference* stages. The relational fan-out described above uses many devices, but as independent single-device tasks that share nothing, which is a weaker requirement than a collective.

The relational backend has no device-to-device shuffle, so it distributes only what the algebra above covers. A chain that reduces with `median`, `quantile`, `var`, `stddev` or `count-distinct` runs on one device, because each needs a group's whole value set. A `right` or `full` join runs on one device, because broadcasting the build side would duplicate its unmatched rows, and a join the planner did not mark `broadcast` runs on one device because its build side does not fit. All are scale ceilings rather than wrong answers, and all would lift with a key-partitioning exchange between devices.

GPU tensors move between stages as Arrow through host memory. There's no device-to-device transport. That's a deliberate consequence of the Arrow-only invariant, and `docs/internals/rfc-gpu-transport.md` proposes changing it. That document is an in-tree proposal, not a description of shipped behavior.

## Code map

Each concern below maps to the file that owns it, so the device placement and batching
rules on this page can be read directly:

| Concern | File |
|---|---|
| Stage-overlapped streaming, chain detection | `python/batcher/core/udf/stream.py` |
| UDF dispatch | `python/batcher/core/udf/execute.py` |
| OOM halving and dirty-row bisection | `python/batcher/core/udf/call.py` |
| Threads vs processes policy | `python/batcher/core/udf/strategy.py` |
| Distributed actor pools, warm pools | `python/batcher/dist/executors/map.py` |
| Latency PID | `crates/bc-udf/src/batch_size.rs`, `python/batcher/ml/inference.py` |
| Throughput hill-climb | `python/batcher/ml/autobatch.py` |
| Device detection, utilization, VRAM | `python/batcher/ml/gpu.py` |
| GPU-vs-CPU backend policy | `python/batcher/kyber/gpu/policy.py` |
| GPU relational backend routing | `python/batcher/api/terminal/gpu_backend.py` |
| Plan and expression translation to cuDF | `python/batcher/core/gpu_plan/` |
| Mergeable split, shared by the optimizer and the backend | `python/batcher/plan/distribution/` |
| Multi-device fan-out, shard recovery, worker-side reads | `python/batcher/dist/gpu/` |
| cuDF and torch scatter-reduce kernels | `python/batcher/core/gpu_transform.py` |

## See also

- {doc}`Architecture <../architecture/index>`: why the GPU paths live in Python and not in the crates.
- {doc}`Execution engine <../internals/execution>`: the UDF stage this pipelines.
- `docs/internals/rfc-gpu-transport.md` (an in-tree RFC, not a site page): the device-to-device transport this page does not have.
- {doc}`GPU guide <../ml/gpu>`: the knobs, from a user's side.
- {doc}`ML guide <../ml/index>`: how to write these pipelines.
- {doc}`Batch inference tutorial <../tutorials/batch-inference>`: the pipeline this page is underneath.
- {doc}`AI and GPU benchmarks <../benchmarks/ai-and-gpu>`: the numbers on this page, in context.
- {doc}`Multimodal ingest benchmarks <../benchmarks/multimodal-ingest>`: the decode side of the same pipeline.
- {doc}`Tensor columns <tensor-columns>`: what crosses into the model.
- {doc}`Distributed scheduling <distributed-scheduling>`: how the actors get placed.
