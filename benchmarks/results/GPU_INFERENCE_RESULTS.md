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

| engine | time | throughput | GPU utilization | vs Ray Data |
|---|--:|--:|---|--:|
| Batcher | 6.01 s | **2,725 img/s** | **87% mean / 100% peak, 4/4 nodes** | **4.53x** |
| Ray Data | 27.21 s | 602 img/s | 40% mean / 94% peak, 4/4 nodes | 1.00x |

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

## The first iteration

Run 0 of a new pipeline used to be the unpacked configuration, because the loop measures and
*then* adapts — which for a single-shot job is the only configuration it ever gets. It now
packs from a measurement of its own: the pool reserves the packable fraction up front (Ray
fixes an actor's `num_gpus` at creation, so a whole-GPU reservation can never add a second
actor later), starts at one actor per device, reads the model's footprint once it has loaded,
and fills each device from that. A stage that *declared* `model_memory_gb` keeps the existing
declared-size path.

Measured from a fresh process, per-node NVML, four T4s, 16,384 images:

| run | throughput | per-GPU utilization |
|---|--:|---|
| 0 (actor spawn + model load + cold read) | 284 img/s | 16 / 16 / 16 / 16 % |
| 1 | **2,918 img/s** | **93 / 91 / 94 / 93 %** |
| 2 | 2,855 img/s | 93 / 92 / 94 / 95 % |

The density is chosen *during* run 0 and held: runs 1+ rebuild nothing. Run 0's own window
still reads 16% because a first run spawns actors, loads a model on every device and reads
its input cold — around 50 of its 57 seconds. That is start-up, not idle GPU, and it is the
part no scheduler removes.

Three things had to be fixed before the density would hold rather than walk:

* **The VRAM cap read a device's usage as one actor's.** Two actors at 12% each read 24%,
  which looks like a single 24% actor — so the cap fell as the pool grew, the recommendation
  fell with it, and the pool was rebuilt (a model reload on every device) run after run. The
  density walked 2 -> 3 -> 2 across three consecutive runs.
* **A warm actor's utilization window was never closed.** It outlives its run, so the window
  kept accumulating the idle time *between* runs: a pool that had just held four T4s at 92.7%
  reported 25.7%, and the loop doubled a pool that was already saturated.
* **The loop chased the target instead of stopping at the goal.** Every step is a rebuild, and
  the step past a satisfied device measured 2,602 img/s at 77/94/95/75% against 2,787 at
  95/93/94/93%. A device at or above 80% is now left alone.

## What this does not fix

**A pipeline that is not GPU-bound.** `gpu_pipeline.py` decodes full-size JPEGs on the CPU
ahead of the model. Batcher runs it at 855 img/s and 28% GPU; Ray Data at 747 img/s and
50%. The higher utilization is the *worse* result: both engines are limited by ~32 cores of
JPEG decode, and Batcher spends less GPU time per image because its forward pass runs in
fp16. On this shape utilization is not a goodness metric, and no scheduling change can
raise it — the device has no more work to be given.

**A short query still under-reports on the benchmark's own meter.** It averages over the whole
timed window, so fixed start-up is charged against it. That is now much smaller — the pool is
packed from run 0, so a timed run rebuilds nothing and the meter reads 87% against the 93-95%
per-node NVML figure over the compute itself. It was 49-56% when every timed run paid a
re-pack.

## The relational GPU backend is a different story

`collect(backend="gpu")` runs TPC-H through the cuDF relational path. It is now *runnable* —
it used to hang — but it is nowhere near the inference path, and the numbers should be read
before anyone plans against it. TPC-H sf1, four T4s, 16 partitions, against DuckDB:

| | |
|---|---|
| correctness | every query that returned was correct (the harness gates it) |
| errored | **7 of 22** (q2, q7, q8, q9, q11, q20, q21) |
| pathological | **7 of 22** at 500-2,100x DuckDB (q4, q13, q15, q16, q17, q18, q22) |
| competitive | 8 of 22 within 0.74-1.28x (q1, q3, q5, q6, q10, q12, q14, q19) |
| device use | **one** GPU at 12-14%; the other three at 0.0% |

Two of those are understood. The errors are a projection applied to a staged intermediate
whose schema no longer carries the column (`dist/fleet/source.py` selects `p_partkey` from a
shuffle bucket that does not have it). The single-device use is that sf1 is far too small for
this path: q1 takes 1.3 s against DuckDB's 47 ms, almost all of it fan-out and shuffle, with
the kernels themselves a rounding error — so the oversubscribed shard fan-out never has enough
work to reach a second device. Neither is a scheduling problem the packing loop can solve.

**A deadlock was fixed to get this far.** Ray gives a task `num_cpus=1` by default, and a GPU
relational shard took that default; the shuffle fleet takes its workers in a placement group,
so on a cluster fanned out to one worker per core the group held every CPU and the GPU shards
waited for a core that never came free. The query did not fail — it hung, with all four
devices idle and `ray status` reporting the cluster fully reserved. q1 hung indefinitely at 32
partitions and completed in 11.5 s at 8; it now returns in 11.7 s at 32.

## Reproducing

```bash
cd benchmarks/cluster
BENCH_GPU_PARQUET=/mnt/cluster_storage/gpuimgs BENCH_GPU_N=16384 \
  BENCH_GPU_SHARDS=32 BENCH_GPU_BATCH=64 BENCH_RUNS=3 python3 gpu_inference.py
```

The first run of a fresh dataset reads it cold from shared storage and is bounded by that
read (19.6 s against 6.4 s warm), so time it twice.
