# GPU batch inference vs Ray Data — where the devices were idle, and why

Companion to `BENCHMARK_RESULTS.md`, which covers the CPU relational suites. This one
covers the GPU path: `benchmarks/cluster/gpu_inference.py` (a fused CPU-normalize +
GPU-forward stage) and `gpu_pipeline.py` (a two-stage JPEG-decode pipeline), both
correctness-gated against Ray Data on the identical model, weights, and data.

## Environment

| | |
|---|---|
| Cluster | 4 x `1xt4-8cpu-32gb` workers (one T4, 8 vCPU each) + a CPU head node |
| Model | torchvision ResNet-50, seeded weights, `batch_size=64` |
| Data | 16,384 images as 32 parquet shards on shared storage, read distributed |
| Gate | every reported time follows a 100.000% prediction-agreement check |
| Versions | ray 2.56.0, torch 2.11.0+cu130, torchvision 0.26.0 |

## Result

| engine | time | throughput | vs Ray Data |
|---|--:|--:|--:|
| Batcher | 6.41 s | **2,556 img/s** | **4.28x** |
| Ray Data | 27.43 s | 597 img/s | 1.00x |

Sustained per-GPU utilization, sampled by an NVML probe pinned to each node across 98,304
images (six back-to-back collects, so the window is steady state rather than start-up):

| | throughput | per-GPU utilization | fleet mean |
|---|--:|---|--:|
| before | 530 img/s | 66 / 0 / 0 / 0 % | 16.6% |
| after | **2,787 img/s** | **95 / 93 / 94 / 93 %** | **93.7%** |

## The three things that were wrong

### One actor ran the whole stage while three GPUs sat at exactly 0.0%

Both actor-pool drivers assigned work to "the first actor with a free in-flight slot",
which fills actor 0 to its submit depth before actor 1 receives anything. An inference
stage's partitions are few and wide, so the partition count is routinely at or below
`len(actors) x depth`, and then the tail of the pool never runs.

The lever that made it worst is the one meant to help: `recommend_inflight_depth` raises
the depth precisely when a GPU looks starved, so the deeper it went the fewer actors got
work. At depth 2 two of four GPUs were idle; at depth 4 a single actor ran everything.

It was invisible from the outside because throughput still looked plausible — the one
working device runs the autocast fp16 path, so a single T4 sustained 530 img/s, which is
close enough to Ray Data's four-device 597 that nothing pointed at the pool.

### The measurement that should have caught it read a peak as a rate

`gpu_stats` reported the utilization sampled *right after a forward pass* — at the instant
the device is busiest. That read 86% for a stage whose sustained figure was 13%.
`recommend_num_gpus` packs a stage onto a fraction of a device only below 50%, and
`recommend_inflight_depth` deepens on the same signal, so the peak reading held both levers
shut. The measurement was keeping its own fix from firing.

Utilization is now a time-weighted mean over the actor's working window (`SustainedUtilization`),
excluding the model load before any work arrives and the idle tail after the last partition.
VRAM stays a peak, because it is a capacity constraint rather than a rate.

### 78% is not "fixed", and the loop had no way to say so

With the first two fixed, one actor per device held 78% and stopped there: 78% is above
every threshold, so the loop declared success. The idle fifth is the UDF's own CPU work
(decode, normalize, the H2D copy), which nothing but a second actor on the device can fill
— measured, a second actor took the same GPUs to 92%.

Packing toward a *target* needs one more fact than utilization: the density that produced
it. The same 78% means "add an actor" at one actor per device and "leave it alone" at two,
and a loop that cannot tell those apart alternates between them forever. `actors_per_device`
is now recorded beside utilization and peak VRAM, and the request is sized to land the
device at 90% — bounded by what measured VRAM allows. The target is deliberately below
saturation, which is also what makes it a fixed point: 92% on two actors computes
`ceil(2 x 0.9 / 0.92) == 2`.

Re-packing also required keying a resident pool by its **resource request**, not only by
its pipeline. Otherwise the pool *grows* to the new replica count and the previous run's
whole-GPU actors stay alive holding every device while their half-GPU replacements wait
forever — the query does not fail, it hangs with the cluster fully reserved.

## What this does not fix

**The first execution of a new pipeline.** The loop measures, then adapts: run 0 uses one
actor per device, and the packed configuration is in force from run 1. A model's VRAM
footprint is not knowable before it is loaded once, so a cold start cannot pack safely
without a declared `model_memory_gb`. Steady state is reached after a single measured run
and holds (93-95% across every subsequent run).

**A pipeline that is not GPU-bound.** `gpu_pipeline.py` decodes full-size JPEGs on the CPU
ahead of the model. Batcher runs it at 855 img/s and 28% GPU; Ray Data at 747 img/s and
50%. The higher utilization is the *worse* result: both engines are limited by ~32 cores of
JPEG decode, and Batcher spends less GPU time per image because its forward pass runs in
fp16. On this shape utilization is not a goodness metric, and no scheduling change can
raise it — the device has no more work to be given.

**The benchmark's own `mean_gpu_pct` on a short query.** It averages over the whole timed
window, so as the query gets faster the fixed start-up cost becomes a larger *fraction* and
the reported mean *falls* even as the device gets busier: 56% at 9.0 s, 49% at 6.4 s, while
per-node NVML over the compute itself reads 93-95%. Read the per-node figure, or lengthen
the window, before concluding anything from that number.

## Reproducing

```bash
cd benchmarks/cluster
BENCH_GPU_PARQUET=/mnt/cluster_storage/gpuimgs BENCH_GPU_N=16384 \
  BENCH_GPU_SHARDS=32 BENCH_GPU_BATCH=64 BENCH_RUNS=3 python3 gpu_inference.py
```

The first run of a fresh dataset reads it cold from shared storage and is bounded by that
read (19.6 s against 6.4 s warm), so time it twice.
