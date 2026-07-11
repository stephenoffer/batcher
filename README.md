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

**Single-node, head-to-head on the same in-memory Arrow (TPC-H sf1).** Each engine runs the
*identical* zero-copy Arrow input Batcher runs on, so this is a like-for-like **execution**
comparison. Cells are **how many times faster Batcher is** (higher = Batcher faster; a `<1`
value means the competitor is faster, shown as e.g. `0.5× (2× slower)`). Every row is
correctness-gated against DuckDB:

| operator | vs DuckDB | vs Polars | vs PyArrow | vs Ray Data | vs Spark |
|-------------------------|:--------:|:--------:|:----------:|:-----------:|:--------:|
| group-by sum (1 key)    | **2.1×** | **2.6×** | 0.7× (1.5× slower) | **306×** | **28×** |
| group-by sum (2 keys)   | **1.5×** | **1.9×** | 0.7× (1.5× slower) | **191×** | **21×** |
| global sum              | **8×**   | **2.2×** | **7×**    | **2700×**   | **197×** |
| filter → count          | **37×**  | **17×**  | **320×**  | **430×**    | **125×** |
| join → group-by         | **2.6×** | **1.4×** | **7×**    | **135×**    | **25×** |
| sort → top-N (`LIMIT`)  | **1.2×** | **29×**  | **345×**  | **477×**    | **24×** |
| filter → project        | **3.1×** | **1.3×** | **23×**   | **22×**     | **20×** |
| window `rank()`         | **2.7×** | **16×**  | n/a¹      | n/a¹        | **19×** |
| window running `sum()`  | **3.2×** | **12×**  | n/a¹      | n/a¹        | **17×** |
| window `lag()`          | **2.1×** | **27×**  | n/a¹      | n/a¹        | **13×** |

¹ PyArrow (Acero) and Ray Data have no window functions. On TPC-H sf1, Batcher likewise beats
DuckDB-on-Arrow on **all 21 runnable queries** (1.3×–4.3× faster) and Spark on every query
(5×–33×).

**How it scales, kept honest.** Against Polars, PyArrow, Spark, and Ray Data, Batcher leads at
every scale we measured. Against DuckDB the story is scale-dependent, and we report it straight:

| scale (single node) | Batcher vs DuckDB (same-input execution) |
|---------------------|-------------------------------------------|
| **sf1** — 6M rows, in memory   | wins **all 21** TPC-H queries (1.3–4.3×) |
| **sf10** — 60M rows, in memory | wins 15 of 21; trails on 6 aggregate/join-heavy queries (1.2–3×) |
| **sf100** — 600M rows, scanned | DuckDB leads 2–11×; Batcher OOMs on the deepest join trees (q3/q4/q5) |

The gap grows with scale for three structural reasons, none of them a tuning knob: DuckDB
**decompresses its native store on the fly** (fewer bytes off memory — Batcher's Arrow-only
contract has no compressed form to read), its **vector-at-a-time engine with selection vectors**
edges Batcher's batch-at-a-time kernels as rows grow, and it **streams** where Batcher's model
materializes each operator's output — which is what OOMs the largest single-node sf100 joins.
Batcher's answer at that scale is **distribution**: the same mergeable operators shard across a
cluster (one partition per node, bounded per-node memory), which is the regime it is built for and
where it beats Ray Data 50–450× (below). Closing the single-node scale gap to DuckDB is honest,
open work — vectorized kernels, dictionary-aware grouping, and streaming between operators.

**Distributed data plane (vs Ray Data).** In-process and native, Batcher pays none of Ray Data's
per-operation task-scheduling + block/pandas-bridge cost (~300–8000 ms fixed, even on a cluster) —
**22×–2700× faster** across the operator mix above. Even on Ray Data's *own* streaming
`map_batches` home turf it leads: `map_batches` transform **2.3×**, row-exploding `flat_map`
**3.5×**, chained multi-stage map **3.2×**, Parquet read **21×**, `iter_torch_batches`
training-data ingest **3.0×**.

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
