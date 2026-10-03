<p align="center">
  <img src="docs/_static/logo.png" alt="Batcher" width="96">
</p>

<h1 align="center">Batcher</h1>

<p align="center">
  One engine for SQL, DataFrames, streaming and models, from a laptop to a cluster.
</p>

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python"></a>
  <a href="https://stephenoffer.github.io/batcher/"><img src="https://img.shields.io/badge/docs-batcher-blue.svg" alt="Docs"></a>
</p>

<p align="center">
  <a href="https://stephenoffer.github.io/batcher/">Docs</a> ·
  <a href="https://stephenoffer.github.io/batcher/getting-started/quickstart.html">Quickstart</a> ·
  <a href="https://stephenoffer.github.io/batcher/getting-started/tour.html">Tour</a> ·
  <a href="https://stephenoffer.github.io/batcher/benchmarks/index.html">Benchmarks</a>
</p>

---

You write Python. A Rust engine runs it over Apache Arrow. Every operator is written once, as a
mergeable `partial → combine → finalize` step, so your laptop and a Ray cluster run the
same code and get the same answer. Scaling out is an argument, not a rewrite:

```python
import batcher as bt

revenue = (
    bt.read("s3://events/*.parquet")
    .filter(bt.col("status") == "active")
    .group_by("region")
    .agg(total=bt.col("amount").sum())
    .sort("total", descending=True)
)
revenue.collect()                                   # one machine
revenue.collect(distributed=True, num_workers=8)    # a Ray cluster, same plan
```

It's quick. On TPC-H sf1 it finishes in a quarter of DuckDB's time and about half of Polars',
both reading the same Arrow input. It also covers more than tables. Streams, lakehouse
tables, video frames and model inference all go through the same plan, and the optimizer
remembers what it measured, so a query you run every night gets a better plan over time.

## Install

```bash
pip install "git+https://github.com/stephenoffer/batcher.git"   # needs Rust 1.89+ and Python 3.11+
```

Prebuilt wheels ship as `batcher-engine` with the first tagged release. The `[ray]` and `[cloud]`
extras add cluster and object-store support. See [all install options](https://stephenoffer.github.io/batcher/getting-started/install/index.html).

## A quick tour

DataFrames and SQL build the same plan, so use whichever reads better:

```python
orders = bt.from_pydict({"region": ["eu", "us", "eu", "us"], "amount": [120.0, 80.0, 45.0, 300.0]})

orders.group_by("region").agg(total=bt.col("amount").sum())
bt.sql("SELECT region, SUM(amount) AS total FROM orders GROUP BY region", orders=orders)
```

Joins and window functions:

```python
users = bt.from_pydict({"uid": [1, 2, 3], "region": ["eu", "us", "eu"]})
spend = bt.from_pydict({"uid": [1, 1, 2, 3], "amt": [5.0, 7.0, 9.0, 2.0]})

totals = users.join(spend, on="uid").group_by("uid", "region").agg(total=bt.col("amt").sum())
totals.with_columns(rank=bt.rank().over(partition_by="region", order_by=[("total", True)]))
```

JSON is an expression, not a parsing step:

```python
logs = bt.from_pydict({"line": ['{"lvl":"ERR","msg":"disk full"}', '{"lvl":"INFO","msg":"ok"}']})
logs.select(lvl=bt.col("line").json.extract_string("$.lvl"))
```

The rest is folded up below.

<details>
<summary>Streaming</summary>

```python
clicks = bt.read.files_incremental("landing/clicks", "parquet", state_dir="state/seen")

counts = (
    clicks.with_watermark("ts", "10 minutes")
    .group_by("page", w=bt.window(bt.col("ts"), "1 minute"))
    .agg(n=bt.count())
)
counts.write.parquet("out/", trigger=bt.Trigger.processing_time("10s"), checkpoint="state/ck")
```

Point it at `bt.read.kafka(...)` instead and the query stays the same. Kinesis, Pulsar and
Pub/Sub work too.

</details>

<details>
<summary>Models and vector search</summary>

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(np.array([[0, 0], [4, 4]]), np.array([0, 1]))
rows = bt.from_pydict({"f0": [0.1, 4.0], "f1": [0.0, 3.9]})
rows.ml.predict(model, features=["f0", "f1"], output_column="label")

docs = bt.from_pydict({"doc": ["a", "b", "c"], "vec": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]})
docs.ml.similarity_to([1.0, 0.0], column="vec").sort("score", descending=True).limit(2)
```

</details>

<details>
<summary>Images and video</summary>

```python
shots = bt.read.images("s3://bucket/photos/")           # uri, bytes, width, height, format, ...
large = shots.filter(bt.col("width") >= 1024)            # answered from headers, no decode
large.select(aspect=bt.col("bytes").image.aspect_ratio())
```

Width and format come from the file header, so that filter throws away small images before
anything gets decoded.

</details>

<details>
<summary>Lakehouse tables</summary>

```python
bt.from_pydict({"id": [1, 2], "v": ["a", "b"]}).write.delta("lake/t", mode="append")
bt.from_pydict({"id": [2, 3], "v": ["B", "c"]}).write.delta("lake/t", mode="append", merge_on="id")
bt.read.delta("lake/t").sort("id").to_pydict()
# {'id': [1, 2, 3], 'v': ['a', 'B', 'c']}
```

</details>

<details>
<summary>Data quality</summary>

```python
raw = bt.from_pydict({"id": [1, 2, 3], "amount": [10.0, -1.0, 30.0]})
raw.dq.positive("amount").drop().to_pydict()
# {'id': [1, 3], 'amount': [10.0, 30.0]}
```

Swap `.drop()` for `.fail()` to raise, or `.quarantine()` to set the bad rows aside.

</details>

<details>
<summary>Polars and pandas interop</summary>

```python
import polars as pl

frame = pl.DataFrame({"city": ["Oslo", "Lima", "Oslo"], "temp": [3.5, 19.0, 5.5]})
bt.from_polars(frame).group_by("city").agg(avg=bt.col("temp").mean()).to_polars()
```

Data crosses over Arrow without a copy, so you can move one slow step of an existing pipeline
and leave the rest alone.

</details>

## Benchmarks

Speedups against each engine reading the same in-memory Arrow. The harness checks every result
against the other engine before it trusts a timing, so a fast wrong answer never counts.

| Suite | vs DuckDB | vs Polars |
|---|:--:|:--:|
| TPC-H sf1, 22 queries | 4.0× | 1.9× |
| ClickBench, 43 queries | 6.3× | 2.7× |
| Semi-structured JSON | 3.1× | over 60× |
| H2O.ai join | 1.7× | 2.0× |
| Operator mix, 46 kernels | 2.1× | 6.3× |

On GPUs, batch inference over 100,000 images on six T4 nodes ran 2.4× faster than Ray Data and
5.4× faster than Daft. ResNet-50 holds 2,504 images a second at 81% GPU utilization on 8×T4,
and MiniLM embeds 33,611 texts a second on the same machine.

The suites come from one sweep on a 48-core box, best of five. The
[benchmark docs](https://stephenoffer.github.io/batcher/benchmarks/index.html) and
[`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md) have the full tables and
the commands to reproduce them.

## How it works

<details>
<summary>One operator at every scale</summary>

Python builds and optimizes a lazy plan, then hands it to Rust as JSON. Rust runs it over Arrow
in morsels spread across every core. Aggregations merge partial states. Joins and windows
partition on their keys, and a sort range-partitions on its leading key. That's why one implementation covers one core and a cluster.
On a cluster, batches go worker to worker over Arrow Flight with credit-based flow control and
never touch the Ray object store.

</details>

<details>
<summary>An optimizer that learns</summary>

On big joined queries, Batcher re-plans at stage boundaries using row counts it measured
rather than guessed. Between runs it keeps column sketches, cost coefficients fitted to real
operator times, and a bandit over join strategies. Run a query often and its plan improves.
[More on this](https://stephenoffer.github.io/batcher/architecture/differentiators.html).

</details>

<details>
<summary>Spill and GPUs</summary>

Aggregations spill to disk when memory runs short, and so do joins and sorts. On GPUs, decode overlaps
with inference and model pools stay loaded between jobs, so the device isn't sitting idle
while the CPU catches up.

</details>

## Learn more

The docs go deeper on all of this. Good places to start:

- [Getting started](https://stephenoffer.github.io/batcher/getting-started/index.html): install it and run a first query
- [User guide](https://stephenoffer.github.io/batcher/user-guide/index.html): a page per capability
- [Machine learning](https://stephenoffer.github.io/batcher/ml/index.html): batch inference, embeddings, retrieval and training loaders
- [Migration guides](https://stephenoffer.github.io/batcher/getting-started/migration/index.html): coming from Spark, pandas, Polars, DuckDB, Ray Data or Daft
- [Cookbook](https://stephenoffer.github.io/batcher/cookbook/index.html) and [API reference](https://stephenoffer.github.io/batcher/api/index.html)
- [Architecture](https://stephenoffer.github.io/batcher/architecture/index.html): how the engine works inside

## Contributing

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
maturin develop      # build the Rust engine into the venv
pytest
```

The Python API lives in `python/batcher/` and the engine in `crates/`. [`MAP.md`](MAP.md) says
what every module is for. If your dev box gets rebuilt and loses its Rust toolchain,
`source tools/bootstrap_env.sh` puts it back.

Batcher is pre-1.0, so APIs can still change. Apache-2.0 licensed.
