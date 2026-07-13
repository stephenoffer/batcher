# Performance and memory

Batcher is built to stay fast on a laptop and survive on a cluster. The levers a
user actually reaches for are few: cache a result you reuse, size the morsels, give
the engine a memory budget so it spills instead of dying, and read back what the
query did. Every knob lives on one frozen `Config`, applied process-wide with
`set_config` or scoped to a block with `config_context`.

This page is about making *your* query faster. For how Batcher compares against DuckDB,
Polars, Ray Data, and Daft, see the [benchmarks](../benchmarks/index.md), which include
the workloads where Batcher still loses.

## Setup

```python
import batcher as bt

events = bt.from_pydict(
    {
        "region": ["us", "eu", "us", "eu", "us", "eu"],
        "status": ["active", "active", "churned", "active", "active", "churned"],
        "amount": [10.0, 5.0, 99.0, 7.0, 3.0, 8.0],
    }
)
```

## Result caching

`cache()` marks a dataset's result to be stored in memory the first time it is
computed. A later terminal on the *same* cached dataset returns the stored result
instead of re-running the plan. It is the Spark/Polars `cache` pattern. Use it when
an expensive upstream (a filter, a join, an aggregation) feeds several downstream
queries.

```python
hot = events.filter(bt.col("status") == "active").cache()

first = hot.to_pydict()   # computed once, then stored
second = hot.to_pydict()  # cache hit — no re-execution
print(first == second)
# True
print(sorted(first["region"]))
# ['eu', 'eu', 'us', 'us']
```

The cache is process-wide and memory-bounded by `memory.result_cache_max_bytes`
(256 MiB by default). It holds results LRU and yields their memory back to running
queries under pressure, so caching never grows the process without bound. And it
marks *this* result: a further transform on a cached dataset is a new, uncached one.

## Reusing a cached dataset

Because a cached dataset is a reusable handle, run as many terminals on it as you
like. Each is served from the one materialized result.

```python
print(hot.count())
# 4
print(hot.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict())
# {'region': ['eu', 'us'], 'total': [12.0, 13.0]}
```

## Morsel-driven execution

The engine's unit of work is a *morsel*: a small Arrow `RecordBatch`, 16,384 rows by
default, sized to fit cache so scheduling stays granular and parallelism stays even
across cores. You rarely change it. When you do, `execution.morsel_rows` and
`execution.morsel_bytes` are the levers, and a morsel splits at whichever bound trips
first, so wide rows (large strings, embeddings, blobs) stay memory-bounded even at a
fixed row count. The setting is result-invariant. A morsel batches data; it never
changes the output.

```python
from batcher.config import Config, ExecutionConfig, config_context

small_morsels = Config().replace(execution=ExecutionConfig(morsel_rows=4096))
with config_context(small_morsels):
    out = events.group_by("region").agg(total=bt.col("amount").sum()).sort("region").to_pydict()
print(out)
# {'region': ['eu', 'us'], 'total': [20.0, 112.0]}
```

`execution.adaptive_morsel_sizing` (on by default) shrinks the per-morsel target
under memory pressure so the streaming working set stays bounded when memory is
tight, and leaves it at the configured target otherwise. For per-batch UDF and
inference workloads, the `pid` section tunes the controller that grows or shrinks
the batch toward a target latency.

## Adaptive re-optimization

Every cost estimate is a guess until the query runs. At a pipeline breaker (a sort,
an aggregate, a join build) the engine has *measured* the real size of what it just
processed. When an estimate was off by more than `optimizer.reoptimize_error` (2x by
default), it re-plans the rest of the query on the measured numbers before
continuing. This is the part static optimizers cannot match. `collect(adaptive=...)`
controls it: `"auto"` (the default) turns it on only when a join's input size is a
pure estimate, and `True`/`False` force it. The result is identical whichever way it
runs.

```python
dim = bt.from_pydict({"region": ["us", "eu"], "tier": ["gold", "silver"]})
joined = (
    events.join(dim, on="region")
    .group_by("tier")
    .agg(total=bt.col("amount").sum())
    .sort("tier")
)
print(joined.collect(adaptive=True).to_pydict())
# {'tier': ['gold', 'silver'], 'total': [112.0, 20.0]}
```

## Out-of-core spilling

Stateful operators (aggregation, distinct, sort, join build, windowed-by-partition)
spill to disk when they would exceed the memory envelope, so a query that does not
fit in memory slows down rather than dying. Spilling is a property of the runtime
primitive, not a separate plan: the result is bit-identical to the in-memory run.

You do not ask an operator to spill; you set a memory budget and the engine decides.
Setting `memory.max_memory_bytes` is what opts the in-memory engine into spilling,
and the data plane receives a per-operator budget of `max_memory_bytes x hard_limit`.
A deliberately tiny budget forces the out-of-core path here so the example runs
anywhere. In production you set it to the real ceiling, honoring a container or cgroup
limit.

```python
from batcher.config import MemoryConfig

big = bt.from_pydict({"k": [i % 50 for i in range(2000)], "v": list(range(2000))})


def totals(ds: bt.Dataset) -> dict:
    return ds.group_by("k").agg(total=bt.col("v").sum()).sort("k").to_pydict()


in_memory = totals(big)

tiny_budget = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
with config_context(tiny_budget):
    spilled = totals(big)

print(in_memory == spilled)
# True — the out-of-core result is identical to the in-memory one
print(len(spilled["k"]))
# 50
```

On a big job the local (NVMe) spill tier overflows to `memory.spill_remote_uri` (any
fsspec URL) once local disk fills. A skewed aggregate bucket that overflows
`memory.spill_bucket_max_bytes` is re-partitioned and reduced one piece at a time, so
a large or skewed query degrades gracefully instead of running out of memory.

## Competitive benchmarks

Every benchmark below is **correctness-gated**: the engines must return the identical
result (a sorted row multiset within float tolerance) before any timing is trusted. A
fast wrong answer is a bug, not a win. Reproduce with:

```bash
python benchmarks/run.py --benchmark operators               # single-node operator mix (sf1)
python benchmarks/run.py --benchmark operators --scale 10    # at 60M rows (sf10)
```

### Single-node operator mix (TPC-H `lineitem`, vs DuckDB and Polars)

Batcher wins the analytical core (grouped aggregation, filters, ordering, the whole
window family) at both 6M (sf1) and 60M (sf10) rows. Each cell is
`batcher / fastest-competitor` wall time, so **below 1.0 means batcher is faster**
(e.g. `0.40×` = 2.5× faster than the next engine).

| operator | sf1 (6M) | sf10 (60M) |
|-------------------------------|:-------:|:--------:|
| group-by sum, one key         | 0.45×   | 0.64×    |
| group-by, two keys            | 0.53×   | 0.89×    |
| filter → count                | 0.32×   | 0.12×    |
| filter → project              | 0.85×   | 0.57×    |
| sort → top-N (`LIMIT`)        | 0.69×   | 0.76×    |
| `DISTINCT` (two columns)      | 0.66×   | 1.00×    |
| window `rank()`               | 0.56×   | 0.40×    |
| window running `sum()`        | 0.36×   | 0.32×    |
| window `lag()`                | 0.54×   | 0.50×    |

The window family is the widest margin: at 60M rows, `rank() OVER (PARTITION BY …)`
runs ~2.5× faster than DuckDB and ~13× faster than Polars, and the whole family
scales: the win holds or grows from 6M to 60M rows. (Daft is 30–100× slower on
windows.) Two-key aggregation and DISTINCT hash their composite keys directly rather
than through a row encoder, so they keep winning as row counts grow.

### Out-of-core resilience (the memory-bound regime)

The harder test is a tight memory budget where **both** engines must spill to disk:
the PB-scale regime in miniature. Batcher stays alive and competitive, and
high-cardinality dedup that loses in memory can flip to a win once the comparison is
out-of-core (batcher's hash-partitioned distinct spill vs DuckDB's):

| operator (60M rows, both engines spilling) | batcher vs DuckDB |
|--------------------------------------------|:-----------------:|
| high-cardinality `DISTINCT` (two int cols) | 0.70× (win)       |
| group-by, two keys                         | ~1.0× (tie/win)   |
| `COUNT(DISTINCT)`                           | 1.6× (completes)  |

The guarantee that matters on a big job is *completion*. Aggregation, distinct, sort
(single- or multi-key), join build, and partitioned windows all spill, and a skewed
partition is recursively re-partitioned so peak memory stays bounded regardless of the
key distribution. The query finishes rather than OOMing.

### vs Ray Data

On Ray Data's own distributed home turf, batcher's in-process native engine is
**50–450× faster** on the SQL operator mix: Ray Data carries a large fixed per-operation
cost (task scheduling plus the block/pandas bridge, ~300–4500 ms) that batcher, native
and in-process, does not pay.

But the fairer test is Ray Data's *bread-and-butter data plane*: streaming
`map_batches` ETL / batch inference, multimodal-style file I/O, and the last-mile
training-data loader. The results below are measured on a single 96-core node (188 GB),
each **correctness-gated** (row count + checksum identical across engines). Every cell
is Ray Data's wall time ÷ batcher's, so **`>1` means batcher is faster**.

**Map-heavy ETL / batch inference** (`read_parquet → map_batches → agg/write`, 20 M rows):

| workload | what it exercises | batcher vs Ray Data |
|-------------------------------------------|------------------------------------|:------:|
| CPU feature transform (`map_batches`, NumPy) | GIL-releasing per-batch inference | **2.3× faster** |
| Pure-Python per-row UDF (`map_batches`) | tokenizer / RAG-chunking proxy | **1.4–2.2× faster** |
| Row-exploding `flat_map` | 1→N fan-out | **3.5× faster** |
| Class-based load-once model (`map_batches(Model)`) | the batch-inference pattern | **1.3× faster** |
| `batch_format="numpy"` / `"pandas"` transform | framework-native preprocessing | **1.0–1.6× faster** |
| Chained multi-stage map (`map → map → filter → group_by`) | pipelined stages | **3.2× faster** |
| Read 2 000 small files → map → agg | small-file ingest | **1.3× faster** |
| `map_batches → write_parquet` (directory of shards) | full ETL out to storage | **1.5–1.8× faster** |
| Metadata `count()` over a directory | no scan needed | **170–950× faster** |

Batcher wins these because a CPU-bound `map_batches` runs across a **warm, shared process
pool** (Ray Data's actor-pool role) that reads its input from RAM-backed shared memory
zero-copy, with no per-worker pickling of the batch, while a GIL-releasing NumPy/torch `fn`
fans across threads with no IPC at all. Reads and multi-file writes are parallelized
(Parquet decode/encode releases the GIL), and a `read → map → write` overlaps compute
with I/O on a background thread the way Ray Data's streaming executor pipelines stages.

**Distributed-training data ingest** (`iter_torch_batches`, 10 M rows × 32 features,
`batch_size=1024`). Ray Data's own docs note it is ~20 % slower than a native PyTorch
`DataLoader` here; batcher is faster still:

| ingest configuration | batcher vs Ray Data |
|------------------------------------------------------|:------:|
| plain stream to tensors | **3.0× faster** |
| with `local_shuffle_buffer_size` | **2.1× faster** |
| with in-stream `map_batches` normalization | **2.4× faster** |
| DDP `streaming_split` across 4 ranks | **3.5× faster** |

The loader (`iter_torch_batches`, `streaming_split` for data-parallel sharding, plus
`stream_loader` / `shard_stream_loader` for exact-shuffle and out-of-core-from-shards)
streams Arrow → tensor in bounded memory with background prefetch, and keeps the shuffle
streaming, where a global `random_shuffle` forces Ray Data to materialize.

**Lazy / metadata control plane.** Ray Data pays a fixed scheduling + execution cost on
even trivial queries; batcher answers them from Parquet metadata and a lazy plan. Same
10 M-row dataset, warm (repeated) call:

| operation | batcher | Ray Data | vs Ray |
|-----------------------------------|--------:|---------:|-------:|
| `schema` | ~0 ms | ~0 ms | tie (cold: 0.03 vs 4 ms) |
| `count()` (whole dataset) | 0.05 ms | 76 ms | **~1400×** |
| `head(10)` | ~0 ms | 170 ms | **>100 000×** |
| `limit(100).collect()` | 71 ms | 173 ms | **2.4×** |
| `filter(pred).count()` | 47 ms | 695 ms | **15×** |

`count()` is answered from footer row counts and cached; `head(10)` streams and stops
after ten rows (Ray Data's `limit` does not short-circuit, it schedules a task). And a
`filter(...).count()` is compiled to a `COUNT(*)` aggregate, so **projection pushdown
reads only the predicate's column** and the count fuses into a single `count_if` pass
rather than materializing every matching row. That is the difference between reading one
column and all of them.

**Data connectors: reads and directory writes** (20 M rows / 64 files; both engines
write a *directory* of shards, Ray Data's default output, for a fair comparison):

| connector | read | directory write |
|-----------|:----:|:---------------:|
| Parquet | **21×** | **4.4×** |
| CSV | **14×** | **4.4×** |
| JSON | **5×** | **1.7×** |

Reads win because batcher decodes files concurrently in-process (Parquet/CSV/JSON decode
releases the GIL) with no task-scheduling or object-store hop. Directory writes shard the
encode across cores, and where the encoder is GIL-bound (JSON via pandas), across
*processes*: each worker encodes and writes one part, no result copy. That took the JSON
directory write from **12.9 s → 1.0 s**. It holds per-worker in the distributed path too,
so multi-node writes shard the same way.

**Broad operation sweep** (20 M rows, each result fingerprinted in Arrow, with no Python
materialization on either side):

| operation | batcher vs Ray Data |
|-----------------------------------|:------:|
| `sort` (full) | **3.6×** |
| `sort` → `head(n)` (top-N) | **>100×** (streams, stops at N) |
| `top_k(100)` | **9×** |
| `group_by` low-cardinality → agg | **29×** |
| `distinct` | **13×** |
| `value_counts` | **34×** |
| `sample(frac)` → count | **16×** |
| selective `filter` → count | **14×** |
| `union` → count | **>1000×** (metadata) |
| `join` → count | **18×** |
| `write_csv` (single file / directory) | **1.3× / 3.6×** |
| `count()` / `head()` | **10³–10⁵×** |

Ray Data's `group_by`, `distinct`, and `value_counts` all pay for an all-to-all shuffle
plus the block/pandas bridge; batcher's are native, in-process, morsel-parallel hash
aggregations. A `sort → head(n)` keeps only the running best N rows (streaming top-N), so
it never sorts the whole relation.

**Lazy schema/metadata after a transform chain.** Because a `Dataset` is a lazy plan,
`schema`, `columns`, `dtypes`, and a derivable `count()` are answered by *inferring over
the plan*, never by executing it. After `select → filter → with_columns → rename → drop`
(and even after a `join → group_by`), batcher returns the schema in **under 1 ms**. Ray
Data must run the pipeline (or a block of it) whenever an opaque `add_column`/`map` is in
the chain: **~200 ms vs ~1 ms, a 100×+ gap**, in exactly the inner loop of interactive,
exploratory work.

**Scale and mode.** Single-node `collect()` materializes, so it is fastest up to memory
limits and wins these workloads through ~60 M rows. Past that, the same mergeable
operators run **distributed** (`collect(distributed=True)`) or **streaming**
(`iter_batches()` / `iter_torch_batches()`), which keep per-node memory bounded. A
row-exploding `flat_map → agg` that materializes 480 M rows on one node instead runs
~**5.8× faster** distributed, where each partition reduces before anything leaves it.
Full numbers and the reproduction scripts are in `benchmarks/BENCHMARK_RESULTS.md`.

### Multimodal & physical-AI ingest (images, point clouds, audio)

Robotics and physical-AI training start by turning a corpus of media files (camera
frames, LiDAR point clouds, audio clips) into model-ready tensors. This is the ingest hot
path, where batcher competes with Ray Data and Daft directly. Every number below is a
single 96-core node, best-of-3 warm, **correctness-gated** (frame/point count and output
shape identical across engines), reproducible from `benchmarks/scenarios/`.

**Image decode + resize** (2 000 JPEG frames, `640×480 → 224×224`, the vision-model
preprocessing step; `benchmarks/scenarios/image_decode.py`):

| engine | img/s | batcher advantage |
|-----------|------:|:-----------------:|
| **batcher** | ~5 700 | — |
| Daft | ~2 400 | **2.4× faster** |
| Ray Data | ~940 | **6.1× faster** |

**Point-cloud / LiDAR loading** (20 000 frames of `4096×3` points streamed to torch via
`iter_torch_batches`; `benchmarks/scenarios/point_cloud_load.py`):

| engine | frames/s | batcher advantage |
|-----------|---------:|:-----------------:|
| **batcher** | ~21 000 | — |
| Ray Data | ~9 000 | **2.4× faster** |

**Audio decode** (encoded clips → PCM waveforms, the Whisper/wav2vec/CLAP first step;
`benchmarks/scenarios/audio_decode.py`) decodes natively (`col(bytes).audio.decode()`,
symphonia) fanned across every core, versus a per-clip `soundfile` loop under the GIL.

batcher wins the whole class for the same reasons.

Decode is native and parallel. Image, audio and video decode run in the Rust data plane
over Arrow, and each media kernel fans out *per row* across the cores, so a corpus smaller
than one 16 384-row morsel still uses the whole machine instead of a single core. Image
decode adds SIMD JPEG (zune, with a 1/2·1/4·1/8 DCT-scaled path for large frames → small
model inputs) and SIMD resize (`fast_image_resize`).

Tensors then cross the boundary already shaped, with no re-type UDF. A decoded image or a
loaded `.npy` frame becomes a canonical `fixed_shape_tensor` column whose shape rides in
Arrow field metadata: zero-copy across the FFI and straight into a `(N, …)` torch tensor.
Because the reader emits the type directly, the decode stays a pure native `with_columns`
on the fully-parallel path, rather than being forced through a slower per-batch
`map_batches` re-type, which alone roughly halved image-ingest throughput.

Files are read concurrently in one wide wave (media decode releases the GIL), and the
loader streams Arrow → tensor in bounded memory with background prefetch.

### Distributed shuffle & data movement (automatic)

A distributed `group_by` / `join` / `sort` moves its data over Carbonite's Arrow Flight
transport, never the Ray object store, and picks the cheapest path for each fetch **with
no configuration**. Same-process buckets are read in place. Same-node, cross-process
buckets (the common case when a node runs several worker actors) are read **zero-copy
from shared memory**, a memory-mapped Arrow IPC file, which measured **≈23× faster than a
loopback Flight hop** (1.2 → 27 GB/s point-to-point on a real cluster), and **7.5×**
through a full concurrent gather. It is on by default and self-limiting: a mapper skips
the shared-memory copy under memory pressure (falling back to Flight), so it never risks
OOM on a tight or churning spot node, and a miss is always bit-identical to the network
path.

Cross-node, a single reducer saturates its NIC (~22 Gbps on a T4 node), so throughput
scales with the **whole cluster**: measured aggregate all-to-all shuffle of **2.0 → 6.9 →
15.2 GB/s at 2 → 4 → 8 nodes**. The mergeable `partial → combine → finalize` algebra and
credit-based flow control keep per-node memory bounded as the cluster grows, so the same
path holds from a few nodes to thousands, including autoscaling spot fleets. None of this
is a knob you set; it adapts to the cluster shape on its own.

### GPU data-plane backend (vs Ray Data + cuDF)

A supported relational query can run on the GPU (cuDF) instead of the CPU engine by
passing `backend=` to `collect()`. It is the same query and the same result; only *where*
it runs changes.

```python
# docs: skip
ds = bt.read.parquet("s3://warehouse/events/")
q = ds.group_by("country").agg(revenue=bt.col("amount").sum())

q.collect(backend="cpu")   # the native engine (default)
q.collect(backend="gpu")   # force the cuDF GPU backend for any supported shape
q.collect(backend="auto")  # let Kyber decide GPU vs CPU by estimated size
```

`backend="auto"` is the adaptive choice. Kyber's cost policy sends a query to the GPU
only when the estimated input is large enough to amortize the device overhead
(host↔device transfer, cuDF import, task dispatch) *and* fits the cluster's GPU memory. A
working set that exceeds one GPU is sharded across many; one that exceeds them all stays
on the spillable CPU engine. Below the crossover a small query stays on the (already fast,
morsel-parallel) CPU engine, and anything unsupported or a GPU-less cluster transparently
falls back. The GPU worker reads its shard straight from storage; the source is never
funnelled through the driver.

The crossover itself is **learned**, not fixed. Each GPU or CPU group-by run records its
(estimated rows, wall time) to the MetadataHub; Kyber fits a cost line per backend and
solves for their intersection, so the threshold self-corrects to the hardware. A faster
GPU or a wider table moves it on its own (Core measures, Kyber consumes). Until enough runs
are seen it uses the measured default (`distributed.gpu_min_rows`, ~10 M rows on an 8×T4).

Measured on an 8×T4 cluster, a `read_parquet → group_by → sum` at **100 M rows**, each
result correctness-gated against the CPU engine:

| engine | wall time | note |
|-----------------------------------|----------:|------|
| batcher `backend="gpu"` (warm) | ~2.3 s | steady-state after cuDF is loaded |
| batcher `backend="gpu"` (cold) | ~7.1 s | first query pays one-time cuDF import |
| Ray Data + cuDF (`map_batches`, `num_gpus=1`) | ~42.6 s | per-block cuDF + object-store + combine |

That is **6× faster cold and ~18× warm than the hand-written Ray Data + cuDF path**. Ray
Data has no GPU aggregate, so a user rebuilds one from `map_batches` and pays its per-block
bridge on top. At small sizes the picture inverts (at 4 M rows the GPU loses ~5× to the CPU
engine), which is exactly why `auto` gates on size. See
`benchmarks/gpu_backend/relational_vs_raydata.py`.

### GPU batch inference & ML workloads (vs Ray Data)

The relational GPU path above is one use of the device. The larger one is ML: batch
inference, embeddings, LLM generation, and training-data ingest through `map_batches`.
Batcher runs these as **stage-overlapped streaming** (a CPU decode/prep stage keeps the GPU
forward fed instead of idling it) with **session-warm model pools** (the model loads once per
session, not once per job). Measured distributed over **8×T4**, each result correctness-gated
(prediction / generated-text / tensor agreement across engines) before any timing is trusted
(`benchmarks/cluster/gpu_*.py`):

| workload (8×T4) | batcher | Ray Data | batcher vs Ray |
|-----------------|--------:|---------:|:--------------:|
| **LLM batch inference** (gpt2 generate, 2048 prompts) | 814 prompt/s (2.5 s) | 73 prompt/s (28 s) | **11.1×** |
| **training-data ingest** (`iter_torch_batches`, zero-copy DLPack) | 1.06 M rows/s | 281 K rows/s | **3.0×** |
| **batch inference** (ResNet-50 classify, iterative) | 2576 img/s @ 78% util | 1257 @ 41% | **2.05×** |
| **batch embeddings** (2048-d feature vectors) | 2502 img/s @ 80% util | 1267 @ 41% | **1.98×** |
| **zero-config GPU** (`map_batches(Model, num_gpus=1)`, no `batch_size`) | 2451 img/s @ 82% util | *hard-errors* | Ray refuses |

The wins are the general `map_batches`-inference mechanism, not one model:

- Stage-overlap streaming lifted a two-stage decode → ResNet-50 pipeline from **942 → 2504
  img/s**, and GPU utilization from **~30% → 81%**. The device stays fed instead of idling
  through the CPU decode. Result-identical to the materializing path; single-node == distributed.
- Session-warm pools (`distributed.warm_inference_pools`, on by default) load the model once
  per session. Ray Data respawns its actor pool and reloads the model on every `collect()`. That
  is where the 2×–11× shows up, in the realistic notebook / batch-inference-service /
  many-datasets pattern, and it grows with model size: the LLM's multi-GB load is paid once,
  not per run.
- Zero config is where out-of-the-box utilization is won or lost. `map_batches(Model,
  num_gpus=1)` with no `batch_size` hard-errors on Ray Data (`must provide batch_size`).
  Batcher picks a VRAM-safe default, streams it with stage overlap, and self-corrects on a CUDA
  OOM by halving the batch, reaching **82% util at 2451 img/s** with zero knobs.

On a *single maximally-large* compute-bound job both engines saturate the same GPUs at the same
FLOPs (131k images: batcher 2504 vs Ray 2383 img/s ≈ parity, both ≥78% util). That is the honest
ceiling; 2× there needs fewer FLOPs (FP16 / quantization), not a faster data plane. Full
methodology and per-scale numbers: `benchmarks/BENCHMARK_RESULTS.md`.

## Reading a query plan

`explain()` runs the optimizer and renders the optimized plan with per-operator
cardinality estimates, without executing. It is how you confirm a predicate landed at
the scan, or that a join was reordered the way you expected.

```python
print(events.filter(bt.col("status") == "active").select("region", "amount").explain())
# A text rendering of the optimized plan, annotated with estimated row counts.
```

Where `explain()` shows the *planned* shape, `stats()` runs the query and reports
what the engine *measured*: rows in/out, wall time, peak bytes, spill, and the
operator that dominated wall time.

```python
run = events.group_by("region").agg(total=bt.col("amount").sum()).stats()
print(run.rows)
# 2
print(run.bottleneck is not None)
# True — the operator that took the most wall time
```

For a quick per-column read of the data itself (counts, null fraction, approximate
distinct count) before a load, `profile()` executes a one-row-per-column summary.

```python
print(events.profile().columns)
# ['column', 'count', 'null_count', 'null_fraction', 'approx_distinct']
```

## Tuning checklist

Reach for these in order; most workloads need none of them.

- A result reused across queries: `cache()` the shared upstream.
- Bounded or container memory: set `memory.max_memory_bytes` to the real ceiling, so
  stateful operators spill instead of OOMing.
- Wide rows (blobs, embeddings): lower `execution.morsel_bytes` to keep the working set
  bounded, and leave `morsel_rows` alone.
- A query slower than expected: `explain()` to check the plan, then `stats()` to find the
  operator that dominated wall time.
- A cluster shuffle under memory pressure: the credit-based backpressure in `flow_control`
  and `distributed`. See [Fault tolerance](../architecture/fault-tolerance.md).

Every field, with its default and meaning, is in
[Configuration options](../configuration/options.md).

## Next steps

- [Configuration options](../configuration/options.md): the full `Config` reference.
- [Fault tolerance](../architecture/fault-tolerance.md): how a distributed query
  survives task, worker, and node failures.
- [Aggregations](aggregations.md): the breakers that spill and re-optimize.
