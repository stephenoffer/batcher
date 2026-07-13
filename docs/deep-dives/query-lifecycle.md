# Query lifecycle

A `collect()` has to cross a language boundary. On one side is a Python object graph the
user built by chaining method calls; on the other is native code that must not call back
into Python for a single row. The lifecycle is the sequence that gets from one to the
other, exactly once per execution, and brings measurements back.

Nothing happens until a terminal call. `filter`, `select`, `join`, `group_by` each return a
new `Dataset` wrapping a new `LogicalPlan`. No data is read; no expression is evaluated.
That deferral is what makes whole-query optimization possible: by the time the engine runs
anything, it has seen the entire computation.

:::{important}
The control plane never touches a row. Python builds and optimizes a plan, ships it as a JSON
document, and receives Arrow buffers back. Every per-row and per-batch operation happens in
Rust. Iterating rows in Python, even once, is the one thing this lifecycle exists to prevent.
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

Drawn with the boundary in it, and with the loop that closes back on the optimizer:

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
        │           │ Carbonite admit│  fits the envelope? narrow it if not │
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

Steps 2–6 are sequenced in exactly one place: `python/batcher/api/orchestration.py`
(`run_relational`). Every relational terminal (single-node, distributed, and each adaptive
stage) routes through it, so the three subsystems are wired together once rather than at
each call site. `api` is the only layer permitted to import all of Kyber, Carbonite, and
Core; they cannot import each other.

### 1. The metadata shortcut

The cheapest execution is none. Before the engine is handed anything, the conductor asks
whether the answer is derivable from statistics the source already declares: a Parquet
footer, an ORC row count, a lakehouse manifest, all of which the IO layer opened anyway for schema
and split planning. `count()`, a global `min`/`max`, `is_empty()`, and a null-count filter
can all come back without a scan.

:::{important}
A shortcut fires only when every statistic on the path is `Provenance.EXACT`. A truncated
string bound, a sketch-derived distinct count, or a learned prior never answers an exact
terminal: the shortcut returns `None` and the query executes normally. Weaken that rule and
`count()` starts returning an estimate that looks like a fact.
:::

See `python/batcher/api/terminal/metadata_answer/` and
[the execution engine page](../internals/execution.md).

### 2–3. Optimize, then admit

Kyber rewrites the logical plan (predicate and projection pushdown, fusion, join order) and
lowers it to a `PhysicalPlan` carrying per-operator `ResourceBounds` and cardinality
estimates tagged with provenance. Carbonite reads those bounds and decides whether the plan
fits the memory envelope; if it does not, it narrows the envelope the executor is handed
(spill enabled, lower parallelism, a smaller credit window) rather than letting the process
walk into an OOM.

Neither subsystem touches data. Kyber decides, Carbonite protects, Core measures. The verbs
stay in their lanes because the subsystems cannot import one another.

### 4–5. Lower and execute

`PhysicalPlan.to_json()` serializes the relational IR. Core calls the one FFI entry point:

```text
out, metrics = _native.execute_plan_metered(plan.to_json(), sources, engine_config_json)
```

`sources[i]` is the relation bound to `Scan { source_id: i }`, a list of pyarrow
`RecordBatch`es. Two very different things cross here, and it is worth separating them:

::::{tab-set}
:::{tab-item} The plan
```text
plan.to_json()   a JSON document, a few kilobytes
                 parsed once per execute_plan call, never per batch
                 deserialized into a bc_ir::RelOp tree inside bc-py
```
Serialization format is irrelevant at this size, and a document you can print, diff, and
paste into a bug report is worth more than a few microseconds.
:::

:::{tab-item} The data
```text
sources[i]       pyarrow RecordBatches, bound to Scan { source_id: i }
                 crossed through the Arrow C Data Interface
                 no serialization, no copy, no Python object per row
```
The result morsels come back the same way. The bytes a pyarrow `RecordBatch` points at and
the bytes a Rust `arrow::RecordBatch` points at are the same bytes.
:::
::::

Everything else stays on its own side of the boundary.

### 6. Feed back

`execute_plan_metered` returns a metrics side-channel alongside the data: per-operator row
counts in and out, elapsed milliseconds, result bytes, whether the operator spilled and by
how much. Core records those into the `MetadataHub`, keyed by a *structural plan signature*
(stable across executions, unlike an operator's position in one plan walk). Kyber reads them
on the next run, and at a pipeline breaker during this one. That loop is what makes the
optimizer improve the more a query shape is run.

## What the user can see

```python
import batcher as bt

ds = bt.from_pydict({"g": ["a", "b", "a", "c"], "x": [1, 2, 3, 4]})
q = ds.filter(bt.col("x") > 1).group_by("g").agg(s=bt.col("x").sum())

# Still nothing has executed. `explain()` shows the optimized plan and its estimates.
print(q.explain())

# The terminal call is where the six steps happen.
print(q.sort("g").to_pydict())
```

:::{dropdown} The plan side of that output
```text
aggregate                       est≈1 (default)
  filter                        est≈1 (default)
    scan                        est≈4 (exact)
```

`est≈4 (exact)` on the scan is the metadata layer: the row count of an in-memory relation is
known exactly. The two nodes above it carry `default` provenance: nobody has measured a
selectivity for this predicate yet. Run the query a few times and Kyber will have.
:::

`explain(format="json")` returns the same tree as a document, with the measured columns
filled in after an `analyze=True` run.

## What it costs

The fixed cost of a small query is the thing this design most easily gets wrong, so it is
worth being exact about which side of the line each cost falls on.

| Cost | Paid | Why |
|---|---|---|
| Cranelift compilation | once per distinct `(expr, schema, simd)`, process-wide | `crates/bc-codegen/src/cache.rs`. Before the memo, a 64-row query with one filter and two projections paid 16.6 ms of it. |
| Thread pool construction | once per width, cached | `par.rs::pool_for`. A one-row query does not spin up 96 threads; the worker count is capped by the morsels the inputs can produce. |
| Plan JSON parse | once per `execute_plan` call | Not per batch. |
| Arrow handoff | once per input relation | Zero-copy through the C Data Interface. |
| The metrics walk | once per metered run | Only on the metered entry point. |

On the operator microbenchmarks a global sum over 6M rows completes in 0.5 ms end to end,
which is the honest measure of the fixed overhead (DuckDB: 2.7 ms; see
[the analytics benchmarks](../benchmarks/analytics.md)).

## Where the code lives

| Step | Code |
|---|---|
| Dataset / terminal ops | `python/batcher/api/dataset/frame.py`, `python/batcher/api/terminal/` |
| The contract loop | `python/batcher/api/orchestration.py` |
| Metadata shortcut | `python/batcher/api/terminal/metadata_answer/` |
| Logical plan + `to_ir()` | `python/batcher/plan/logical/`, `python/batcher/plan/ir_tags.py` |
| Physical plan + `to_json()` | `python/batcher/plan/physical.py` |
| Core's call into the engine | `python/batcher/core/executor.py` |
| The FFI boundary | `crates/bc-py/src/lib.rs` |
| The executor | `crates/bc-interp/src/lib.rs` (sequential), `par.rs` (multi-core) |

## Adaptive stages

One thing the diagram above flattens: a query with a pipeline breaker may run steps 2–5 more
than once. At a breaker the engine has *measured* the data it just processed. If an estimate
was off by more than `optimizer.reoptimize_error` (default 2x), the remainder of the plan is
re-optimized on the measured numbers and executed as a new stage. The stateful operator's
state lives in `bc-runtime`, not in generated code, so a compiled pipeline can be thrown away
and rebuilt at a breaker without losing progress.

## See also

:::{seealso}
- [Architecture](../architecture/index.md): the three-subsystem shape this sequence wires together
- [Execution engine](../internals/execution.md): the architecture-level view of steps 4–6
- [Kyber](../internals/kyber.md): what step 2 actually does to the plan
- [Carbonite](../internals/carbonite.md): what step 3 admits against
- [Reading a plan](../user-guide/explain-plans.md): how to read the `explain()` output above
- [Performance](../user-guide/performance.md): applying all of this to a slow query
- [Analytics benchmarks](../benchmarks/analytics.md): where the 0.5 ms fixed-overhead figure comes from
- [Plan IR](plan-ir.md): the JSON wire contract the boundary speaks
- [Morsel parallelism](morsel-parallelism.md): how a plan becomes work for N cores
- [Adaptive re-optimization](adaptive-reoptimization.md): why steps 2–5 can run more than once
:::
