# Batcher

```{raw} html
<div class="bt-hero">
  <p class="bt-hero-tagline">One data engine, from your laptop to your cluster.</p>
  <p class="bt-hero-sub">
    Batcher runs SQL, DataFrame, and ML workloads on a JIT-compiling Rust core, and
    re-optimizes the query while it is still running. Sub-second on a laptop,
    bounded-memory at petabyte scale &mdash; the same code either way.
  </p>
  <p class="bt-hero-cta">
    <a class="bt-btn bt-btn-primary" href="getting-started/index.html">Get started</a>
    <a class="bt-btn" href="getting-started/quickstart.html">Quickstart</a>
    <a class="bt-btn" href="https://github.com/stephenoffer/batcher">GitHub</a>
  </p>
</div>
```

Most engines plan a query once, before they have seen a single row, then commit to
that plan whatever the data turns out to be. Batcher does not. It measures the data
as it flows and re-plans the rest of the query on real numbers, so a query that
starts on a bad estimate corrects itself mid-flight.

The design splits in two. Python builds the plan, optimizes it, and decides how much
it should cost, but never touches a row. Every per-row operation runs in Rust over
Apache Arrow. Between them sits one boundary: a JSON plan going down, zero-copy Arrow
batches coming back.

```{raw} html
<div class="bt-points">
  <div>
    <h3>Adaptive re-optimization</h3>
    <p>Kyber re-plans at pipeline breakers using measured cardinalities. DuckDB
    optimizes once; Spark adapts only at stage boundaries.</p>
  </div>
  <div>
    <h3>One algebra, one to many machines</h3>
    <p>Stateful operators are written once as mergeable partial / combine / finalize
    primitives. One core or a thousand run the same code, with bounded memory and
    spill to disk.</p>
  </div>
  <div>
    <h3>JIT-compiled expressions</h3>
    <p>A Cranelift fast path compiles column expressions once per operator and reuses
    the machine code across batches, with a checked fallback to the interpreter.</p>
  </div>
  <div>
    <h3>Lazy, immutable API</h3>
    <p>A Dataset is a handle to a plan. Each operation returns a new one; nothing runs
    until a terminal call such as <code>collect</code> or <code>write.parquet</code>.</p>
  </div>
</div>
```

## A first query

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "category": ["a", "b", "a", "b", "a"],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "qty": [1, 2, 3, 4, 5],
    }
)

summary = (
    ds.with_columns(total=bt.col("price") * bt.col("qty"))
    .group_by("category")
    .agg(revenue=bt.col("total").sum(), orders=bt.count())
    .sort("revenue", descending=True)
)

print(summary.to_pydict())
# {'category': ['a', 'b'], 'revenue': [220.0, 200.0], 'orders': [3, 2]}
```

Files and object stores use the same API. Only the source changes.

```python
# docs: skip
ds = bt.read("s3://bucket/events.parquet")
ds.filter(bt.col("status") == "active").write.parquet("output/active.parquet")
```

## Where to go next

```{toctree}
:maxdepth: 2
:caption: Documentation

getting-started/index
tutorials/index
user-guide/index
migration/index
api/index
configuration/index
ml/index
learning-paths/index
architecture/index
internals/index
```
