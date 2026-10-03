# Query lifecycle

This page follows one {py:meth}`collect() <batcher.Dataset.collect>` from a Python object graph down to native code and back, exactly once per execution, with measurements returning along the way.

Nothing happens until a terminal call. `filter`, `select`, `join`, and `group_by` each return a new {py:class}`Dataset <batcher.Dataset>` wrapping a new `LogicalPlan`, so by the time the engine runs anything it has seen the entire computation:

```python
import batcher as bt

ds = bt.from_pydict({"g": ["a", "b", "a", "c"], "x": [1, 2, 3, 4]})
q = ds.filter(bt.col("x") > 1).group_by("g").agg(s=bt.col("x").sum())
print(type(q).__name__)          # Dataset  (nothing has executed yet)
print(q.sort("g").to_pydict())   # {'g': ['a', 'b', 'c'], 's': [3, 2, 4]}
```

:::{important}
The control plane never touches a row. Python builds and optimizes a plan, ships it as a JSON document, and receives Arrow buffers back. Every per-row and per-batch operation happens in Rust.
:::

## The six steps

```text
Dataset.collect()
  │
  ├─ 1. metadata shortcut   api/terminal/metadata_answer/   (can the footer answer it?)
  ├─ 2. optimize            kyber/                          logical plan → physical plan
  ├─ 3. admit               carbonite/                      does it fit the envelope?
  ├─ 4. lower               plan/physical.py::to_json       physical plan → JSON IR
  ├─ 5. execute             bc_py::execute_plan_metered     JSON IR + Arrow in, Arrow out
  └─ 6. feed back           metadata/MetadataHub            measured rows/times/bytes
```

:::{dropdown} The same six steps, drawn across the language boundary
```text
        ┌──────────────────── Python: the control plane ────────────────────┐
        │                                                                   │
 collect()  Dataset ──► LogicalPlan                                         │
        │                   │                                               │
        │        1  ┌───────▼────────┐  answered from a footer?             │
        │           │ metadata_answer├──────────────► rows  (no execution)  │
        │           └───────┬────────┘   only if every stat is EXACT        │
        │                   │ no                                            │
        │        2  ┌───────▼────────┐                                      │
        │           │ Kyber optimize │  logical → physical + ResourceBounds │
        │           └───────┬────────┘                                      │
        │        3  ┌───────▼────────┐                                      │
        │           │ Carbonite admit│  fits the envelope? spill if not     │
        │           └───────┬────────┘                                      │
        │        4  ┌───────▼────────┐                                      │
        │           │ to_json()      │                                      │
        └───────────┴───────┬────────┴──────────────────────────────────────┘
                            │
             JSON IR (a few KB, parsed once)  +  Arrow C Data Interface (zero-copy)
                            │
        ┌───────────────────▼───────────────────────────────────────────────┐
        │  5   bc-py::execute_plan_metered                                  │
        │        RelOp tree ──► bc-interp ──► morsels ──► result batches    │
        └───────────────────┬───────────────────────────────────────────────┘
                            │
       Arrow batches (zero-copy)  +  ExecMetrics: rows, ns, bytes, spill
                            │
        ┌───────────────────▼───────────────────────────────────────────────┐
        │  6   MetadataHub.record(...)                                      │
        │        └──────► Kyber reads it on the next run, and at the next   │
        │                 pipeline breaker in this one  ────────────────┐   │
        └───────────────────────────────────────────────────────────────┼───┘
                                                                        │
                        back to step 2 for the rest of the plan  ◄──────┘
```
:::

Steps 2 through 6 are sequenced in one place, `run_relational` in [`python/batcher/api/orchestration/run.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/api/orchestration/run.py). Every relational terminal routes through it, single-node, distributed, or an adaptive stage. `api` is the only layer allowed to import Kyber, Carbonite, and Core, which can't import each other.

### 1. The metadata shortcut

The cheapest execution is none. Before the engine is handed anything, the conductor asks whether the answer follows from statistics the source already declares, such as a Parquet footer, an ORC row count, or a lakehouse manifest. `count()`, a global `min`/`max`, `is_empty()`, and a null-count filter can all return without a scan:

```python
import batcher as bt

bt.from_pydict({"x": [1, 2, 3, 4]}).write.parquet("nums.parquet")
nums = bt.read.parquet("nums.parquet")
print(nums.count())                       # 4
print(nums.agg(n=bt.count()).explain())   # no aggregate operator: the count is folded to a constant
```

:::{important}
A shortcut fires only when every statistic on the path is `Provenance.EXACT`. A truncated string bound, a sketched distinct count, or a learned prior never answers an exact terminal.
:::

### 2 and 3. Optimize, then admit

Kyber pushes down predicates and projections, fuses operators, and chooses a join order, then lowers the result to a `PhysicalPlan` carrying per-operator `ResourceBounds` and provenance-tagged estimates. Carbonite decides whether the plan fits the memory envelope. If memory is the binding constraint, the verdict is a counter-offer that routes the query out of core instead of into an OOM. Any other binding constraint raises a `PlanError` before anything executes.

![The contract loop a terminal op drives, as a clockwise ring of four stations, each carrying the verb that keeps it in its lane. Kyber DECIDES, in kyber.optimize_full, and hands Carbonite a PhysicalPlan with a resource bound per operator. Carbonite PROTECTS, in carbonite.validate, and when the plan fits it reserves and runs it. Core MEASURES, in core.execute, and returns per-operator metrics: actual rows, time, peak bytes. The MetadataHub REMEMBERS, in collect_source_metadata, and the dashed edge from it back to Kyber is read on the next run, not this one. Admission has a third outcome besides pass and fail: a plan that will not fit in memory gets a counter-offer and is routed out-of-core, to spill and then run. Any other binding constraint raises instead, because spilling only ever answers a memory constraint.](/_static/diagrams/query_lifecycle.svg)

### 4 and 5. Lower and execute

`PhysicalPlan.to_json()` serializes the relational IR, and Core calls the one FFI entry point:

```text
out, metrics_json = _native.execute_plan_metered(plan.to_json(), sources, engine_cfg, query_id)
```

`sources[i]` is the relation bound to `Scan { source_id: i }`, a list of pyarrow `RecordBatch`es, and `query_id` makes the execution cancellable. Two very different things cross here:

::::{tab-set}
:::{tab-item} The plan
```text
plan.to_json()   a JSON document, a few kilobytes
                 parsed once per execute_plan call, never per batch
                 deserialized into a bc_ir::RelOp tree inside bc-py
```
:::

:::{tab-item} The data
```text
sources[i]       pyarrow RecordBatches, bound to Scan { source_id: i }
                 crossed through the Arrow C Data Interface
                 no serialization, no copy, no Python object per row
```
:::
::::

### 6. Feed back

`execute_plan_metered` returns per-operator row counts, elapsed and CPU nanoseconds, peak and result bytes, and spill figures alongside the data. Core records them into the `MetadataHub` keyed by a structural plan signature, and Kyber reads them on the next run, or at the next pipeline breaker of an adaptive run. That loop is what makes a query shape improve the more it runs.

## What the user can see

`explain()` shows the optimized plan and its estimates without executing anything. `explain(format="json")` returns the same tree as a document, with measured columns filled in after an `analyze=True` run:

```python
import batcher as bt
import json

ds = bt.from_pydict({"g": ["a", "b", "a", "c"], "x": [1, 2, 3, 4]})
q = ds.filter(bt.col("x") > 1).group_by("g").agg(s=bt.col("x").sum())
print(q.explain())
print("optimized_ir" in json.loads(q.explain(format="json")))  # True
```

:::{dropdown} The plan side of that output
```text
query plan (planned)                          3 operators
─────────────────────────────────────────────────────────
OPERATOR                 ESTIMATE  NOTES
aggregate  [by g · sum]     est≈1  (default)
└─ filter  [x > 1]          est≈3  (default)
   └─ scan  [source 0]      est≈4  (exact)  pushed[x > 1]
```

`est≈4 (exact)` on the scan is the metadata layer: an in-memory relation's row count is known. The nodes above it carry `default` provenance until a run has measured them. `pushed[x > 1]` records that the predicate already sits inside the scan.
:::

## Fixed costs

A small query's fixed cost is where this design is easiest to get wrong, so each cost is paid once rather than per batch:

| Cost | Paid | Why |
|---|---|---|
| Cranelift compilation | once per distinct `(expr, schema, simd)`, process-wide | Memoized in [`crates/bc-codegen/src/cache.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-codegen/src/cache.rs). |
| Thread pool construction | once per width, cached | `par.rs::pool_for`. The worker count is capped by the morsels the inputs can produce. |
| Plan JSON parse | once per `execute_plan` call | Not per batch. |
| Arrow handoff | once per input relation | Zero-copy through the C Data Interface. |
| The metrics walk | once per metered run | Only on the metered entry point. |

On the operator benchmarks, a global sum over TPC-H `lineitem` at scale factor 1, 6M rows on 16 cores, completes in 0.5 ms end to end against DuckDB's 2.7 ms. See {doc}`the analytics benchmarks </benchmarks/results/analytics>` for the full table and the hardware.

## Adaptive stages

A query can run steps 2 through 5 more than once. With `adaptive="auto"`, a single-node query stages only when it has a join whose inputs are still sized by a guess and it clears a floor of 5 million rows or about 320 MB per pipeline breaker the loop would cut at. On a cluster, a plan the one-shot dispatcher can't run correctly stages at any size. Inside a staged run, each breaker's output is measured, and if an estimate missed by more than `optimizer.reoptimize_error` (default 2.0) the rest of the plan is re-optimized on the measured numbers. See {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`.

:::{dropdown} Where the code lives
| Step | Code |
|---|---|
| Dataset / terminal ops | [`python/batcher/api/dataset/frame.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/api/dataset/frame.py), [`python/batcher/api/terminal/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/api/terminal) |
| The contract loop | [`python/batcher/api/orchestration/run.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/api/orchestration/run.py) |
| Metadata shortcut | [`python/batcher/api/terminal/metadata_answer/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/api/terminal/metadata_answer) |
| Logical plan + `to_ir()` | [`python/batcher/plan/logical/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/plan/logical), [`python/batcher/plan/ir_tags.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/plan/ir_tags.py) |
| Physical plan + `PhysicalPlan.to_json()` | [`python/batcher/plan/physical.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/plan/physical.py) |
| Core's call into the engine | [`python/batcher/core/executor.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/core/executor.py) |
| The FFI boundary | [`crates/bc-py/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/lib.rs) |
| The executor | [`crates/bc-interp/src/lib.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-interp/src/lib.rs) (sequential), `par.rs` and `stream/` (multi-core) |
:::

## See also

- {doc}`Architecture </architecture/index>`: the three-subsystem shape this sequence wires together.
- {doc}`Execution engine </architecture/internals/execution>`: the architecture-level view of steps 4 through 6.
- {doc}`Kyber </architecture/internals/kyber>`: what step 2 does to the plan.
- {doc}`Carbonite </architecture/internals/carbonite>`: what step 3 admits against.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: how to read the `explain()` output above.
- {doc}`Performance </user-guide/operate/tuning/performance>`: applying all of this to a slow query.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: where the 0.5 ms fixed-overhead figure comes from.
- {doc}`Plan IR </architecture/deep-dives/query/plan-ir>`: the JSON wire contract the boundary speaks.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: how a plan becomes work for N cores.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: why steps 2 through 5 can run more than once.
