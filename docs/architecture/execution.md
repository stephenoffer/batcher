# Execution model

This page describes how Batcher turns an optimized plan into results: the lazy API, the
pipeline and breaker model, and the three execution paths.

Python builds and optimizes the plan. Rust runs every per-row computation over Apache Arrow. The two meet at a JSON plan IR plus zero-copy Arrow `RecordBatch`es, and nothing else crosses.

:::{dropdown} Where the plan crosses into Rust
Core hands the plan to the native engine in `core/executor.py`:

```python
out, metrics_json = _native.execute_plan_metered(plan.to_json(), sources, engine_cfg, query_id)
```

{doc}`Execution engine </architecture/internals/execution>` has the contributor's view: which crate runs which scale, the thresholds with their config names, and the metadata layer that answers some terminals without a scan.
:::

## Lazy evaluation

The API is lazy and immutable. Each operation returns a new {py:class}`Dataset <batcher.Dataset>` wrapping a `LogicalPlan`, and nothing computes until a terminal call:

```python
# docs: run
import batcher as bt

ds = bt.from_pydict({"x": [-1, 2, 3, 15], "y": ["a", "b", "c", "d"]})
result = ds.filter(bt.col("x") > 0).select("x", "y")  # builds a plan, runs nothing
print(type(result).__name__)
# Dataset
print(result.collect().to_pydict())  # the plan runs here
# {'x': [2, 3, 15], 'y': ['b', 'c', 'd']}
```

By `collect`, the optimizer sees the entire computation. It can push predicates and projections down, fuse operators and order joins before a batch is read, and it still has one plan to revise mid-query.

The terminals are {py:meth}`collect() <batcher.Dataset.collect>`, which returns a PyArrow `Table`; `to_pydict()`; `count()`; `iter_batches()`, which streams a result; and the `write` namespace, either {py:obj}`ds.write("out/") <batcher.Dataset.write>` or a typed form such as {py:meth}`ds.write.parquet(...) <batcher.api.io_namespace.writer.Writer.parquet>`. `explain()` shows the optimized plan without running it:

```python
# docs: run
plan = ds.filter(bt.col("x") > 10).select("y").explain()
assert "pushed[x > 10]" in plan
```

## Pipelines and breakers

Execution lowers a plan into pipelines and breakers. A pipeline is a maximal chain
of operators that streams a batch straight through without materializing, such as
scan, filter, project, and probe. A breaker is an operator that must collect its input
before it can produce output, such as a hash-join build, an aggregate, a sort, a
distinct, or a window.

![A streaming Scan-Filter-Project pipeline feeding two pipeline breakers: the HashJoin build, then the Aggregate.](/_static/diagrams/pipeline_breakers.svg)

Breakers do the real work. Data materializes there, spills there under memory pressure, shuffles there when a query is distributed, and gets re-optimized there once real numbers are known.

The unit of work flowing through a pipeline is the *morsel*, a `RecordBatch` of 16,384 rows or 1 MiB, whichever limit it reaches first. The row bound keeps the working set in cache. The byte bound fires on wide data such as images or embeddings:

```python
# docs: run
cfg = bt.active_config().execution
print(cfg.morsel_rows, cfg.morsel_bytes)
# 16384 1048576
```

## Execution paths

There is one set of operator semantics, exercised by three paths. The Tier-0 sequential
interpreter is the reference. It is deterministic and kept obviously correct, and the
other two paths are tested against it.

![One shared Expr and RelOp feeding three execution tiers. The Tier-0 sequential interpreter is the correctness oracle. The Tier-0 parallel path changes only scheduling and must equal the oracle. The Tier-1 Cranelift JIT must be bit-for-bit identical on its supported subset, and an unsupported expression falls back to the interpreter rather than diverging.](/_static/diagrams/execution_tiers.svg)

Tier-0 parallel reuses the same operator code and changes only the scheduling: it morselizes, runs on a rayon thread pool, and hash-shuffles into the breakers. Tier-1 is the Cranelift JIT. It compiles fixed-width numeric and temporal arithmetic, comparisons, boolean logic and `CASE` once per operator, caches the artifact process-wide, and reuses it across every morsel. Anything else falls back to the interpreter, so the JIT stays bit-for-bit identical to it.

The JIT compiles scalar expressions, never relational state. Hash tables, partial aggregates and sort buffers live in `bc-runtime`, so re-planning at a breaker throws away at most a compiled expression.

## One algebra, single node to cluster

Stateful operators are written once as mergeable primitives: `partial(batch)`
builds a partial state, `combine(states)` merges two of them, and `finalize(state)`
emits rows. Because `combine` is associative and commutative, partials merge in any
order. That single implementation serves one core (the sequential interpreter),
many cores (the parallel path builds partials and combines them), and many machines,
where the distributed path composes the same `partial`, `combine`, and `finalize`.
There is no separate distributed operator with its own semantics, so the rows, the
column names and the column types come back the same on a laptop and on a cluster.
[`tests/integration/test_distributed.py`](https://github.com/stephenoffer/batcher/blob/main/tests/integration/test_distributed.py) asserts that equality operator by operator against a Ray cluster.

## Adaptive re-optimization

At a stage boundary the engine has *measured* the data it just processed rather than
estimated it: real row counts, real operator times, and real peak memory. Core records
those numbers, and when an estimate was off by more than `optimizer.reoptimize_error`
(2.0 by default), Kyber re-plans the rest of the query on the measured values before
continuing.

This is stage-boundary re-optimization, the same granularity Spark AQE works at, and Batcher runs it single-node as well as distributed. `adaptive="auto"`, the default, spends the loop where it can pay for itself: on a single node, a joined query that clears 5 million rows or about 320 MB per pipeline breaker the loop would cut at, so the simplest joined shape qualifies at about 10 million rows. A separate cross-query loop feeds sketch-backed statistics from each run into the next, at any size.

`explain(analyze=True)` runs the query and prints each operator's estimate beside its measured row count:

```python
# docs: run
report = ds.filter(bt.col("x") >= 3).explain(analyze=True)
assert "actual=2" in report
```

## Memory and spilling

Carbonite owns the memory envelope. It throttles at `memory.soft_limit` (85% of the envelope by default) and spills to disk before the budget is exhausted. Aggregation, join, sort and window all have a spill path, so a query too large for memory gets slower rather than failing. Spilling changes where state lives, never what the query computes. This group-by runs under a 64 MiB envelope:

```python
# docs: run
import dataclasses

cfg = bt.active_config()
tight = dataclasses.replace(cfg, memory=dataclasses.replace(cfg.memory, max_memory_bytes=64 << 20))
big = bt.from_pydict({"k": [i % 1000 for i in range(200_000)], "v": list(range(200_000))})
with bt.config_context(tight):
    print(big.group_by("k").agg(s=bt.col("v").sum()).sort("k").limit(2).to_pydict())
# {'k': [0, 1], 's': [19900000, 19900200]}
```

## Distribution

Ray is an optional dependency used for scheduling and control-plane metadata only, and single-node execution never loads it. On a cluster, each worker hosts the same in-process Rust engine, and bulk Arrow batches move between workers over Arrow Flight (`bc-transport`) with credit-based flow control: one credit is one in-flight batch slot, and a producer blocks at zero. Batches bypass the Ray object store entirely. The radix-partition-and-spill machinery behind single-node out-of-core is also the distributed shuffle, so disk and network are two sinks for one mechanism.

```python
# docs: skip
result = big.group_by("k").agg(s=bt.col("v").sum()).collect(distributed=True, num_workers=4)
```

## See also

- {doc}`Execution engine </architecture/internals/execution>`: the tiers, the crate map, and the exact thresholds.
- {doc}`Architecture overview <overview>`: the two planes and the control-plane subsystems.
- {doc}`Fault tolerance <fault-tolerance>`: what happens when a worker or a task fails.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: how morsels are cut and scheduled.
- {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`: the compiled subset and its parity guard.
- {doc}`Configuration options <../configuration/options>`: every execution knob.
