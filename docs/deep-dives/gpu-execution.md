# GPU execution

A GPU costs roughly forty times what a CPU core costs and idles just as easily. The whole
job of an engine running GPU work is to keep the device fed, and almost every way of failing
at that is a *scheduling* failure, not a kernel failure.

Batcher has two distinct GPU paths, and neither of them is in Rust. The `bc-*` crates
contain no GPU code at all.

| Path | What runs on the device | Where |
|---|---|---|
| GPU relational backend (`collect(backend="gpu")`) | cuDF dataframe ops, torch scatter-reduce fallback | Python, in a Ray task with `num_gpus=1` |
| GPU inference stage (`map_batches(..., num_gpus=…)`) | the user's torch model | a Python Ray actor |

The second is the one that matters. The first is a narrow acceleration for large group-bys;
the second is the entire ML batch-inference workload.

## Keeping the device fed

The naive way to run a decode → inference pipeline is stage-at-a-time: decode the whole
partition, then run the forward pass. The GPU idles through the entire decode.

`core/udf/execute.py` detects a linear `scan → map → … → map` chain and runs it as a
prefetch-pipelined stream, each stage on its own thread, so the CPU decode of morsel *k+1*
overlaps the GPU forward of morsel *k*.

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

   same hardware, same result — verified per batch, order preserved,
   single-node equal to distributed. the device just stops waiting.
```

It is an execution property of the engine rather than a feature of the inference operator, so
any CPU-heavy → compute chain inherits it, for any modality.

:::{warning}
A *higher* GPU utilization percentage is not automatically better, and this cuts against us
elsewhere so it is worth stating here. A slower engine spreads the same GPU work over more
wall-clock and reads as higher utilization. Throughput is the number that matters; utilization
only explains it.
:::

## Warm pools

Ray Data respawns its actor pool, and reloads the model, on every execution. Batcher keeps
GPU inference pools warm across `collect()` calls within a session
(`distributed.warm_inference_pools`, on by default), so the model loads once per *session*.

This is worth about 2× on iterative or repeated inference, and it is the single biggest
contributor to the 11× on LLM batch inference: a gpt2 FP16 load takes ~7 s while generation
takes ~1 s, so a per-execution reload *is* the cost. Measured over 8×T4, 2,048 prompts:
batcher 2.51 s versus Ray Data 27.98 s, with 100% text match.

Pools are keyed by UDF identity, healed when an actor dies to preemption, and freed at
process exit or via `release_inference_pools()`.

:::{important}
A warm pool only helps if the model is *loadable once*. That is why `map_batches(Model,
num_gpus=1)` takes a class, not a function: the class's `__init__` loads the model and
`__call__` runs it. Passing a closure that loads the model per batch emits a
`PerformanceWarning`, and it throws away the single biggest win on this page.
:::

## Batch sizing

There are two controllers, and they optimize different things. Conflating them is the mistake
the design avoids.

| | Latency controller | Throughput controller |
|---|---|---|
| For | online serving | offline batch |
| Optimizes | a per-batch latency setpoint | maximum rows/sec under a VRAM cap |
| Method | a PID over the *relative* latency error | a constrained hill-climb |
| Lives in | `crates/bc-udf/src/batch_size.rs`, `ml/inference.py` | `ml/autobatch.py` |

::::{tab-set}
:::{tab-item} Latency (online serving)
A PID over the *relative* per-batch latency error, driving the batch size toward a latency
setpoint. It is implemented identically in
`crates/bc-udf/src/batch_size.rs::BatchSizeController` and `ml/inference.py::_LatencyController`,
and shipped to Rust as `EngineConfig` so the two cannot drift.

```rust
let error = (self.target_latency_ms - observed_latency_ms) / self.target_latency_ms;
self.integral = (self.integral + error).clamp(-INTEGRAL_CLAMP, INTEGRAL_CLAMP);
let derivative = error - self.prev_error;
let adjustment = (self.kp * error + self.ki * self.integral + self.kd * derivative)
    .clamp(-MAX_STEP_FRACTION, MAX_STEP_FRACTION);
self.current = (self.current * (1.0 + adjustment)).clamp(min, max);
```

The control law is applied *multiplicatively* to the current size over the *relative* error,
which makes it scale-free (it behaves the same at 100 rows and 100,000), with a natural
fixed point at `observed == target`. The integral clamp is anti-windup; the step cap stops a
single anomalous latency swinging the size wildly.
:::

:::{tab-item} Throughput (offline batch)
A latency PID optimizes the wrong thing for a batch job. What you want is maximum rows/sec
*subject to a VRAM cap*, which is a constrained hill-climb, not a setpoint tracker.
`ml/autobatch.py::ThroughputController` grows the batch by 1.5× while throughput keeps
improving, shrinks by 0.7× on a VRAM breach, and settles at the best size seen. VRAM is a hard
constraint with a *predictive* guard: it only grows if `vram_fraction × grow <= vram_cap`, so it
does not have to hit an OOM to learn where the wall is.
:::
::::

The PID's shipped gains:

```python
from batcher.config import PIDConfig

pid = PIDConfig()
print(pid.kp, pid.ki, pid.kd, pid.integral_clamp, pid.max_step_fraction)
```

```text
0.4 0.05 0.1 5.0 0.5
```

:::{note}
`architecture.txt` describes a PID controller targeting a *GPU-utilization* setpoint (80–90%),
and an RL/PPO batch sizer. Neither exists. The PID targets per-batch **latency**; the GPU path
uses the non-PID throughput hill-climb above; utilization is measured but feeds `num_gpus` and
in-flight-depth recommendations, never a PID.
:::

## Zero config

The simplest possible call is where out-of-the-box utilization is won or lost:

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

# No batch_size. Ray Data hard-errors here; Batcher picks a VRAM-safe default.
ds = bt.read.images("s3://bucket/frames/", decode=True, size=(224, 224))
out = ds.ml.map_batches(Classifier, num_gpus=1, batch_format="torch").collect()
```

Ray Data raises `ValueError: You must provide batch_size to map_batches when requesting GPUs`.
Batcher picks a VRAM-safe default (256 rows), streams it with stage overlap, and self-corrects
on a CUDA OOM by halving the batch. That reaches 82% utilization at 2,451 img/s on 131k images
across 8×T4, matching the hand-tuned `batch_size=128` path (2,504 img/s, 81%), with no
knobs.

## Autocast, and why it probes

Half precision is not a free win. A conv/matmul forward gets tensor cores; an autoregressive
generation loop is launch- and memory-bound and gets nothing, or worse.

So `ml/gpu.py::autocast_call` does not blindly wrap. It times FP32 against autocast on a
64-row probe, best-of-3, and keeps autocast only if it is at least 1.15× faster. The verdict
is cached per callable. `torch.compile` is applied on the same principle: only to models
containing a `Conv2d`, because it measured ~0.9× on text transformers.

## Measuring the device

`ml/gpu.py` is the vendor-neutral measurement layer. `detect_backend()` returns
`cuda | rocm | xpu | mps | tpu | cpu`, and utilization sampling dispatches through NVML,
ROCm SMI, or the XPU equivalent. On MPS and TPU there is no utilization API, so the loop is a
no-op rather than a guess.

```python
from batcher.ml.gpu import detect_backend

print(detect_backend())
```

```text
cpu
```

Device attribution honors `CUDA_VISIBLE_DEVICES` (and the ROCm equivalents), so a Ray-pinned
actor averages only *its* devices rather than the whole node's. Getting this wrong makes a
one-GPU actor on an eight-GPU node report 12% utilization when it is saturated.

The measurements feed a learned loop: per-model peak VRAM and utilization are recorded in the
`MetadataHub` and consumed by `recommend_num_gpus` (pack two models onto one device below 50%
utilization), `recommend_inflight_depth` (a starved device gets 4× the submit-ahead slots),
and `max_actors_per_gpu`.

## Dirty data

Real corpora contain rows that fail to decode. `_resilient_call` bisects a failing batch to
isolate the bad rows and drops them against a `max_errored_rows` budget, and halves the batch
on a CUDA OOM. With ~1% corrupt rows injected across 200k, Batcher retains 99% of the data and
Ray Data retains 0%. One bad image should cost you one image, not the job.

## Costs and limits

The honest ceiling: a *single, maximally large, compute-bound* job reaches roughly parity with
Ray Data (2,504 vs 2,383 img/s). Both engines saturate the same device at the same FLOPs, and
no scheduling cleverness changes arithmetic. One T4 sustains ~400 img/s at 100% utilization on
ResNet-50; eight actors reach ~3,200 with no parallel penalty. Every win described above comes
from the pipeline, and once the pipeline is not the bottleneck there is nothing left to win.

A GPU `fn` never runs in a process pool, because it must keep a single process and CUDA context, so
the GIL is a real constraint on a GPU stage whose Python glue is heavy.

Multi-GPU collective placement does not work today. `SchedulingEnvelope.gpu_collective` and
`placement_strategy` exist as declared seams, but the actor pool sets only `num_gpus` and
`accelerator_type`; there is no placement group. See `docs/internals/rfc-gpu-transport.md`,
which is explicitly a proposal.

And GPU tensors move between stages as Arrow through host memory. There is no device-to-device
transport. That is a deliberate consequence of the Arrow-only invariant, and it is the second
thing the RFC proposes changing.

## Code map

| Concern | File |
|---|---|
| Stage-overlapped streaming | `python/batcher/core/udf/stream.py` |
| UDF dispatch, OOM/dirty-row resilience | `python/batcher/core/udf/execute.py` |
| Threads vs processes policy | `python/batcher/core/udf/strategy.py` |
| Distributed actor pools, warm pools | `python/batcher/dist/executors/map.py` |
| Latency PID | `crates/bc-udf/src/batch_size.rs`, `python/batcher/ml/inference.py` |
| Throughput hill-climb | `python/batcher/ml/autobatch.py` |
| Device detection, utilization, VRAM | `python/batcher/ml/gpu.py` |
| GPU-vs-CPU backend policy | `python/batcher/kyber/gpu/` |

## See also

:::{seealso}
- [Architecture](../architecture/index.md): why the GPU paths live in Python and not in the crates
- [Execution engine](../internals/execution.md): the UDF stage this pipelines
- `docs/internals/rfc-gpu-transport.md` (an in-tree RFC, not a site page): the device-to-device transport this page does not have
- [GPU guide](../ml/gpu.md): the knobs, from a user's side
- [ML guide](../ml/index.md): how to write these pipelines
- [Batch inference tutorial](../tutorials/batch-inference.md): the pipeline this page is underneath
- [AI and GPU benchmarks](../benchmarks/ai-and-gpu.md): the numbers on this page, in context
- [Multimodal ingest benchmarks](../benchmarks/multimodal-ingest.md): the decode side of the same pipeline
- [Tensor columns](tensor-columns.md): what crosses into the model
- [Distributed scheduling](distributed-scheduling.md): how the actors get placed
:::
