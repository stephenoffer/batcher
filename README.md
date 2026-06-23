# Batcher

**One data engine from your laptop to your cluster — and it re-optimizes itself
while the query is still running.**

Pick a data tool today and you're picking its ceiling. The fast single-node engines
don't scale out. The ones that scale are slow to start on small jobs, or they ship
without a real optimizer. So teams outgrow their engine and rewrite the pipeline, or
they run one stack for SQL, another for DataFrames, and a third for ML — and pay to
keep the seams from leaking.

Batcher is one engine for all of it: sub-second queries on a laptop and PB-scale jobs
on a cluster, batch and streaming, across SQL, DataFrame, and ML/multimodal
workloads. The same code runs on one core or a thousand, so scaling out is a
deployment change, not a rewrite.

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

## Why it's different

Every other engine plans your query once, before it has seen a single row, and then
commits to that plan no matter what the data turns out to be. When the estimate is
wrong (a filter that should cut 90% of rows but cuts 5%, a join whose "small" side
turns out huge), they run the bad plan to the end. DuckDB's optimizer is static.
Spark AQE re-plans, but only at stage boundaries.

Batcher re-optimizes *during* the query. At every pipeline breaker it has **measured**
the data it just processed: real row counts, real memory, real timings. It re-plans
the rest of the query on those numbers instead of guesses, so a query that starts on
a bad estimate corrects itself mid-flight. That's the moat, and a static optimizer
can't retrofit it.

## How it compares

| Tool | Where it stops | What Batcher does instead |
|------|----------------|---------------------------|
| **DuckDB** | fast, but single-node and plans once | scales out, and re-optimizes mid-query |
| **Polars** | fast, but single-node and one backend | the same mergeable code from one core to a cluster |
| **Spark** | scales, but carries cluster overhead on small jobs | runs in-process locally — no cluster to spin up |
| **Ray Data** | scales, but has no cost-based optimizer | a learned, cost-based optimizer (Kyber) |

These are capability differences, not a benchmark brag. Speed is measured against
DuckDB and Polars correctness-first — the harness refuses to time a query whose
result doesn't match DuckDB, and every relational operator is differential-tested
against it. The numbers live in [`benchmarks/`](benchmarks/); run them yourself.

## How it's built

Python builds and optimizes the query plan but never touches a row; it ships the plan
as JSON and gets Arrow batches back, zero-copy. Rust does all the per-row work —
compiling pipelines to machine code (data-centric produce/consume, after HyPer/Umbra)
and scheduling them as 16K-row morsels. Three things follow from that split:

- **You don't rewrite to scale.** Stateful operators are written once as mergeable
  `partial / combine / finalize` primitives, so one core, many cores, and many
  machines run the same code with the same results and bounded memory.
- **Small queries stay small.** Single-node runs a pure in-process Rust engine with
  no cluster, no actors, no startup tax — the sub-second path isn't an afterthought.
- **Distribution doesn't add a second engine.** On a cluster, Ray handles scheduling
  only; the data moves over Arrow Flight and never touches the Ray object store, so a
  result is identical whether it ran on one node or a hundred.

The full design, and the math behind the optimizer and resource manager, is in
[`docs/`](docs/) and `architecture.txt`.

> Status: young but working, not yet 1.0. The engine runs SQL and DataFrame workloads
> single-node and distributed, and is benchmarked for correctness and speed against
> DuckDB and Polars. Expect APIs to change, and some operators and large-scale paths
> are still landing.

## Install

Batcher is distributed on PyPI as `batcher-engine` and imported as `batcher`
(the bare `batcher` name belongs to an unrelated project). Prebuilt wheels ship for
Linux, macOS, and Windows on Python 3.10+ — no Rust toolchain needed:

```bash
pip install batcher-engine
```

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3]})
print(ds.select(doubled=bt.col("x") * 2).to_pydict())
# {'doubled': [2, 4, 6]}
```

Optional features are extras, e.g. `pip install "batcher-engine[ray,cloud]"`. To
install an unreleased revision from source (requires a [Rust toolchain](https://rustup.rs)):
`pip install "git+https://github.com/stephenoffer/batcher.git"`.

## Build from source

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
maturin develop           # build the Rust engine into the venv
pytest
```

## Layout

- `python/batcher/` — public API and control plane (never touches a tuple in the hot path)
- `crates/` — the Rust engine (only `bc-py` links PyO3)
- `docs/`, `architecture.txt` — design and the mathematical foundations

Apache-2.0 licensed.
