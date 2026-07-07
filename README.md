# Batcher

**One data engine that runs on your laptop and scales to a cluster — and tunes
itself while your query runs.**

Most data tools make you choose: fast on a single machine, or able to scale across
many — rarely both. So teams outgrow their tool and rewrite the pipeline, or run one
system for SQL, another for DataFrames, and a third for ML and pay to keep the seams
from leaking. Batcher is a single engine for all of it: quick on small data, steady
at large scale, for SQL, DataFrame, and ML workloads. The same code runs on one core
or a thousand, so going bigger is a config change, not a rewrite.

```python
import batcher as bt

revenue = (
    bt.read("s3://events/*.parquet")
    .filter(bt.col("status") == "active")
    .group_by("region")
    .agg(total=bt.col("amount").sum())
    .sort("total", descending=True)
)
print(revenue.to_pydict())   # nothing runs until here
```

## What makes it different

Most engines decide how to run your query before they have looked at any data, then
stick with that plan even when the data turns out different — which is the usual
reason a job stalls or runs out of memory. Batcher watches the data as it flows and
adjusts the plan as it goes, so a query that starts on a bad guess corrects itself
instead of failing. No other engine does this *during* a query.

## How it compares

| Tool | Where it stops | What Batcher does instead |
|------|----------------|---------------------------|
| **DuckDB** | fast, but single-node and plans once | scales out, and re-optimizes mid-query |
| **Polars** | fast, but single-node | the same code runs from one core to a cluster |
| **Spark** | scales, but heavy on small jobs | runs in-process locally — no cluster to spin up |
| **Ray Data** | scales, but no cost-based optimizer | a learned, cost-based optimizer |

These are capability differences. The speed is real too, and measured
correctness-first: the benchmark harness refuses to time a query whose result doesn't
match DuckDB, so a fast wrong answer never counts as a win.

## Benchmarks

Numbers, not adjectives — and every one is **correctness-gated**: the harness runs each query
on every engine, checks they return the *identical* result (a sorted row multiset within float
tolerance), and only then trusts the timing. A fast wrong answer is a bug, not a win. Setup:
TPC-H `lineitem` (6M rows at scale 1, 60M at scale 10), read once into Arrow and shared
byte-identically across engines; a 9-node / 128-CPU cluster; 8×T4 GPUs for the ML runs. Full
per-scale tables: [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md).

**Analytical SQL, single-node (vs DuckDB / Polars).** Each cell is `batcher / fastest-competitor`
wall time — **below 1.0 means Batcher is faster** (`0.40×` = 2.5× faster). The margin *holds or
grows* from 6M to 60M rows, so it scales rather than just starting fast:

| operator | sf1 (6M) | sf10 (60M) |
|--------------------------------|:-------:|:--------:|
| group-by sum, one key          | 0.45×   | 0.64×    |
| filter → count                 | 0.32×   | **0.12×** |
| sort → top-N (`LIMIT`)         | 0.69×   | 0.76×    |
| window `rank()`                | 0.56×   | **0.40×** |
| window running `sum()`         | 0.36×   | **0.32×** |

At 60M rows `rank() OVER (…)` is **~2.5× faster than DuckDB** and **~13× faster than Polars**.
Under a tight budget where *both* engines spill, Batcher stays alive and competitive — a
high-cardinality `DISTINCT` even flips to a **1.4× win** out-of-core.

**Distributed data plane (vs Ray Data).** In-process and native, Batcher pays none of Ray Data's
per-operation task-scheduling + block/pandas-bridge cost (~300–4500 ms fixed, even on a cluster):

| operation | batcher | Ray Data | speedup |
|-----------------------------|--------:|---------:|:-------:|
| group-by sum | 14 ms | 1,824 ms | **127×** |
| global sum | 4 ms | 1,804 ms | **440×** |
| sort → top-20 (`LIMIT`) | 15 ms | 4,569 ms | **306×** |

Even on Ray Data's *own* streaming `map_batches` home turf, Batcher leads: `map_batches` transform
**2.3×**, row-exploding `flat_map` **3.5×**, chained multi-stage map **3.2×**, Parquet read **21×**,
`iter_torch_batches` training-data ingest **3.0×**.

**GPU batch inference (8×T4, vs Ray Data)** — stage-overlap streaming keeps the device fed and
session-warm pools load the model once per session, not once per job:

| GPU workload | batcher | Ray Data | vs Ray |
|--------------|--------:|---------:|:------:|
| **LLM batch inference** (gpt2 generate) | 814 prompt/s | 73 prompt/s | **11.1×** |
| batch inference (ResNet-50) | 2576 img/s @ 78% util | 1257 @ 41% | **2.05×** |
| batch embeddings (2048-d vectors) | 2502 img/s @ 80% util | 1267 @ 41% | **1.98×** |
| zero-config `map_batches(Model, num_gpus=1)` | 2451 img/s @ 82% util | *hard-errors* | Ray refuses |

Stage-overlap alone lifted a decode → ResNet-50 pipeline from **942 → 2504 img/s** and GPU
utilization from **~30% → 81%** — same result, the device just stops idling through the CPU decode.

**Why the wins happen** — they're structural, not tuning: an in-process native engine over Arrow
(no per-operation scheduler / object-store hop), composite-key hashing that makes aggregation and
`DISTINCT` scale, warm model pools + stage-overlapped streaming for GPU work, and adaptive
re-optimization that re-tunes the plan on measured cardinalities mid-query.

## Install

Prebuilt wheels ship for Linux, macOS, and Windows on Python 3.10+ — no Rust needed.
Batcher is on PyPI as `batcher-engine` and imported as `batcher` (the bare `batcher`
name belongs to an unrelated project):

```bash
pip install batcher-engine
```

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3]})
print(ds.select(doubled=bt.col("x") * 2).to_pydict())
# {'doubled': [2, 4, 6]}
```

Optional features are extras, e.g. `pip install "batcher-engine[ray,cloud]"`.

New here? The [documentation](https://stephenoffer.github.io/batcher/) has a
quickstart, guides, and runnable examples.

## Under the hood

You write Python; the heavy lifting happens in Rust over [Apache
Arrow](https://arrow.apache.org/). Python builds and optimizes your query, and Rust
runs it — so you get Python's ergonomics with native speed. The same engine powers
one core and a whole cluster, which is why a result is identical whether it ran on
your laptop or a hundred machines. The full design (and the math behind the
optimizer) is in the
[documentation](https://stephenoffer.github.io/batcher/) and `architecture.txt`.

> **Status:** young but working, not yet 1.0. Batcher runs SQL and DataFrame
> workloads, single-node and distributed, and is benchmarked for correctness and
> speed against DuckDB and Polars. Expect APIs to change, and some operators and
> large-scale paths are still landing.

## Build from source

Requires a [Rust toolchain](https://rustup.rs):

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
maturin develop           # build the Rust engine into the venv
pytest
```

Or install an unreleased revision directly:
`pip install "git+https://github.com/stephenoffer/batcher.git"`.

## Layout

- `python/batcher/` — the Python API
- `crates/` — the Rust engine
- `docs/`, `architecture.txt` — design and documentation

Apache-2.0 licensed.
