# vs Ray Data

This is the widest margin Batcher has, and it holds on Ray Data's own home turf rather
than only on SQL, where Ray Data is weakest. The reason is structural. Ray Data pays a
fixed per-operation cost of roughly 300–4,500 ms even on a warm cluster (task scheduling plus
the block/pandas bridge), and Batcher, in-process and native over Arrow, pays none of it. On
small and medium work, that fixed cost *is* the runtime.

:::{important}
Nothing here is a scheduling trick against a straw man. Ray Data attaches to a live Ray
cluster in every comparison, which is where it is designed to be strongest, and every
workload is correctness-gated (row counts and checksums, or prediction agreement for the
model runs) before a time is recorded. A run whose result disagrees with the oracle reports
`FAILED` and contributes no number to any table below.
:::

:::{note}
The tables on this page come from three different machines: a 16-core node for reads and
writes, a 96-core node for the `map_batches` data plane, and an 8×T4 Ray cluster for the
model work. Ratios hold *within* a table. A number lifted out of one and set against a
number from another says nothing. [Methodology](methodology.md) lists the hardware per
family.
:::

## Scorecard

| Workload | Result |
|---|---|
| Text embeddings (MiniLM) | **47× faster** |
| Fraud feature aggregation (tabular) | **139× faster** |
| Parquet read → aggregate | **20.8× faster** |
| Audio feature extraction | **12.5× faster** |
| LLM batch inference (gpt2) | **11.1× faster** |
| Image generation (diffusion) | **8.6× faster** |
| Training ingest (`iter_torch_batches`) | **3.0× faster** |
| `map_batches` CPU transform | **2.35× faster** |
| Batch inference (ResNet-50, iterative) | **2.05× faster** |
| `batch_format="pandas"` transform | **1.02×**, a tie |
| One maximally large, compute-bound GPU job | ~parity, and it should be |
| Batcher's own `backend="gpu"` aggregate | **Ray Data is ahead**, 0.9× |

## Ray Data's data plane

The fair test is not SQL. It is streaming `map_batches` ETL, batch inference, and last-mile
training ingest. Single node, 96 cores, 188 GB; 20M rows across 96 files. Ratio is
`ray_ms / batcher_ms`, so **above 1 means Batcher is faster**.

| Workload | Batcher | Ray Data | vs Ray |
|---|---:|---:|---:|
| `flat_map` (1→4 row expansion → count) | 455 ms | 1,586 ms | **3.5×** |
| `chained_map` (map → map → filter → group-by) | 1,807 ms | 5,733 ms | **3.17×** |
| `cpu_map` (per-batch NumPy transform → sum) | 1,011 ms | 2,372 ms | **2.35×** |
| `map_write_dir` (map → write a Parquet directory) | 1,250 ms | 2,099 ms | **1.68×** |
| `numpy_format` (`batch_format="numpy"`) | 2,002 ms | 2,883 ms | **1.44×** |
| `class_inference` (`map_batches(Model)`, load once) | 2,067 ms | 2,672 ms | **1.29×** |
| `many_files_map` (2,000 files → map → sum) | 2,356 ms | 2,982 ms | **1.27×** |
| `pandas_format` (`batch_format="pandas"`) | 1,663 ms | 1,702 ms | 1.02× |
| `py_map` (pure-Python per-row UDF → sum) | 1,123–1,808 ms | ~2,400 ms | **1.3–2.2×** |

`pandas_format` is a tie, and it is the honest shape of the result: when the UDF boundary
forces a pandas conversion, most of the wall clock is that conversion and the engine
underneath barely matters.

What produces the wins elsewhere: a CPU-bound UDF runs on a warm shared process pool that
reads its input from RAM-backed shared memory zero-copy, so there is no per-worker pickle;
a GIL-releasing NumPy or torch function fans across threads with no IPC at all; and a
`read → map → write` overlaps compute with I/O off-thread.

## Reading and writing

20M rows across 64 files, single node (16 cores, 30 GB). Both engines write a *directory*
of shards, which is Ray Data's default output, so the comparison is like-for-like.

| Operation | Batcher | Ray Data | vs Ray |
|---|---:|---:|---:|
| `read_parquet` + sum | 72 ms | 1,502 ms | **20.8×** |
| `read_csv` + sum | 98 ms | 1,394 ms | **14.3×** |
| `read_json` + sum | 302 ms | 1,588 ms | **5.3×** |
| `write_parquet` (dir) | 317 ms | 1,396 ms | **4.4×** |
| `write_csv` (dir) | 326 ms | 1,430 ms | **4.4×** |
| `write_json` (dir) | 1,016 ms | 1,709 ms | **1.68×** |

Reads win because Batcher decodes files concurrently in-process (Parquet, CSV, and JSON decode
all release the GIL) with no per-file task scheduling and no object-store hop.

The JSON write was an embarrassment before it was a win. A `to_pylist()` plus a per-row
`json.dumps` took **over 65 seconds** for a single file, and a directory write took 12.9 s,
which is 7.7× *behind* Ray Data. Encoding across processes and streaming the result took
that to 1.0 s.

## The lazy control plane

A metadata question should not execute a query. 10M rows across 64 files, warm, on the
96-core node:

| Operation | Batcher | Ray Data | vs Ray |
|---|---:|---:|---:|
| `count()` | 0.05 ms | 76 ms | **~1,400×** |
| `head(10)` | ~0 ms | 170 ms | **>100,000×** |
| `filter(pred).count()` | 47 ms | 695 ms | **15×** |
| `limit(100).collect()` | 71 ms | 173 ms | **2.4×** |

`filter(pred).count()` was the one loss in this group, at 2,187 ms and 3.2× behind Ray
Data, because it scanned all 32 columns. Compiling `.count()` to a `COUNT(*)` aggregate let
projection pushdown prune the scan to the predicate's column and fuse the count into
`count_if`: 2,187 → 47 ms, and from 3.2× behind to 15× ahead.

## GPU and AI workloads

Distributed over 8×T4 with real models, each run gated on prediction agreement:

| Workload | Model | vs Ray Data |
|---|---|---:|
| Text embeddings | sentence-transformers MiniLM | **47×** |
| Audio feature extraction | torchaudio mel + ResNet-18 | **12.5×** |
| LLM batch inference | HF gpt2 | **11.1×** |
| Image generation (diffusion) | diffusers ddpm-cifar10 | **8.6×** |
| Video-clip inference | ResNet-18 per frame | **3.6×** |
| Training-data ingest | `iter_torch_batches` (DLPack) | **3.0×** |
| Batch inference | ResNet-50 | **2.05×** |
| Batch embeddings (image) | ResNet-50 features | **1.98×** |
| Fractional-GPU packing | EfficientNet-B0, 2 per GPU | **1.96×** |
| Zero-config GPU | `map_batches(Model, num_gpus=1)` | Ray Data hard-errors |

The large ratios are all the same mechanism seen from different angles. A model that loads
once per *session* instead of once per *job* turns a 7-second gpt2 load from the dominant
cost into a rounding error, which is why the LLM number is 11× and scale-independent: Ray
Data reloads the model on every execution. MiniLM loads in ~2 s and embeds nearly instantly,
so the reload is essentially the entire Ray Data runtime, and the ratio reaches 47×.

The zero-config row is not a speed claim. `ds.map_batches(Model, num_gpus=1)` with no
`batch_size` raises `ValueError: You must provide batch_size to map_batches when requesting
GPUs` on Ray Data. Batcher picks a VRAM-safe default, streams it with stage overlap, and
halves the batch on a CUDA OOM: 2,451 img/s at 82% utilization with no knobs.

[AI and GPU](ai-and-gpu.md) has the detail.

## Tabular feature engineering

The batch fraud-detection path, distributed across the 8×T4 Ray cluster. The dominant cost
is per-account aggregation over transaction history, and it is relational, so Batcher runs
it in the Rust engine as a mergeable `partial → shuffle → combine` while Ray Data has no
relational optimizer and a known-weak group-by shuffle. 20M transactions, 200k accounts:

| Engine | Throughput | Wall |
|---|---:|---:|
| **Batcher** | **77.0 M rows/s** | **260 ms** |
| Ray Data (`groupby().aggregate(...)`) | 0.6 M rows/s | 36,301 ms |

**139×**, with per-account means agreeing to 4.3e-14. The full pipeline (aggregate, join the
features back onto every transaction, score) runs 3.8 M rows/s against Ray Data's 0.7 M
rows/s, a **5.3×**.

## Dirty data

Real corpora contain rows that fail to decode. On the 8×T4 cluster, with ~1% corrupt rows
injected across 200k rows and each engine's tolerance flag enabled:

| Engine | Granularity | Rows kept |
|---|---|---:|
| **Batcher** (`max_errored_rows`) | per row | **198,000 / 200,000 (99%)** |
| Ray Data (`max_errored_blocks=-1`) | per block | 0 / 200,000 (0%) |

Both engines survive. Neither crashes. But with corruption spread roughly one row in a
hundred, *every* Ray Data block contains a bad row, so block-granular tolerance discards the
entire dataset. One bad image should cost you one image.

## Where Ray Data is not behind

:::{warning}
Two results here go the other way, and one of them is Batcher losing to itself. A single
maximally large compute-bound GPU job reaches parity, and Batcher's own opt-in
`backend="gpu"` relational path is **slower than Ray Data** on a distributed group-by sum.
Both are published because a benchmark page that only contains wins is a brochure.
:::

**A single maximally large compute-bound GPU job.** At 131k images both engines saturate the
same devices at the same FLOPs: 2,504 img/s against 2,383, both above 78% utilization. That
is the honest ceiling, and no scheduling cleverness moves it. Beating it needs fewer FLOPs
(FP16, quantization), which is a model-side decision.

**Batcher's own GPU relational backend loses to its CPU engine, and to Ray Data.** On a
distributed group-by sum (20M rows, 8×T4):

| Engine | Throughput | Wall |
|---|---:|---:|
| Batcher CPU (native Rust mergeable aggregate) | 69 M rows/s | 289 ms |
| Ray Data (`groupby().aggregate()`) | 0.7 M rows/s | 30.4 s |
| Batcher GPU (`backend="gpu"`, distributed) | 0.6 M rows/s | 33.9 s |

A group-by sum is memory-bound, so the GPU's compute advantage does not apply, and the GPU
path still pays task dispatch, per-shard read, and a host→device transfer for a reduction
that is trivial once the bytes have moved. `backend="gpu"` stays opt-in for exactly this
reason. It is published here because it is the kind of result a benchmark page normally
leaves out.

## Distributed

Even distributed-against-distributed, with both engines on the same live cluster, Batcher
beats Ray Data on every pipeline at every scale tested. See [scaling](scaling.md).

## Reproduce

:::{dropdown} The four scripts behind this page
```bash
python benchmarks/run.py --benchmark operators --tier multi
python benchmarks/cluster/vs_ray_daft.py 10
python benchmarks/cluster/fraud_scoring.py
python benchmarks/cluster/gpu_pipeline.py
```
:::

## See also

- [AI and GPU](ai-and-gpu.md): the ten GPU workload families in full.
- [Scaling](scaling.md): the distributed cluster runs.
- [Ray integration](../integrations/ray.md): how the two systems actually compose, which is
  the answer for most people who read this page.
- [Distributed scheduling](../deep-dives/distributed-scheduling.md) and
  [shuffle over Flight](../deep-dives/shuffle-flight.md): why the data plane bypasses the
  Ray object store.
- [Mergeable algebra](../deep-dives/mergeable-algebra.md): the `partial → combine →
  finalize` behind the 139× fraud aggregation.
- [Batch inference](../ml/inference.md): the warm pool the large ratios come from.
- [Methodology](methodology.md): hardware per family; the rows are not comparable to each
  other.
