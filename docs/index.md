# Batcher

Batcher is a native, JIT-compiling data engine with an adaptive optimizer. A
Python control plane builds and optimizes a query plan; a Rust data plane runs it
over Apache Arrow. The same engine targets sub-second queries on a laptop and
PB-scale jobs on a cluster, for SQL, DataFrame, and ML workloads.

The design splits cleanly in two. Python builds a plan, optimizes it, and decides
how much it should cost, but never touches a row of data. All per-row work runs in
Rust over Arrow record batches. The boundary between them is a JSON plan plus
zero-copy Arrow batches.

## What sets it apart

- **Adaptive re-optimization.** The optimizer (Kyber) re-plans during a query, at
  pipeline breakers, using measured cardinalities rather than static estimates.
  DuckDB optimizes once before execution; Spark adapts only at stage boundaries.
- **One algebra, single node to cluster.** Stateful operators are written once as
  mergeable `partial / combine / finalize` primitives. The same code runs on one
  core, many cores, or many machines, with bounded per-node memory and spill to
  disk.
- **JIT-compiled expressions.** A Cranelift fast path compiles column expressions
  once per operator and reuses the compiled code across batches, with a checked
  fallback to the interpreter.
- **Lazy and immutable API.** A `Dataset` is a handle to a plan. Every operation
  returns a new `Dataset`; no work runs until a terminal operation such as
  `collect`, `to_pydict`, or `write.parquet`.

## Quick example

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

Reading from files or object stores uses the same API; only the source changes:

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
