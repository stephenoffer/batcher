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

These are capability differences, not a benchmark brag. Speed is measured
correctness-first: the benchmark harness refuses to time a query whose result doesn't
match DuckDB, and every operator is checked against it. The numbers live in
[`benchmarks/`](benchmarks/) — run them yourself.

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
