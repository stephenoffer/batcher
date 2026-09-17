# Query lifecycle

A {py:meth}`collect() <batcher.Dataset.collect>` has to cross a language boundary. On one side is a Python object graph you built by chaining method calls. On the other is native code that must not call back into Python for a single row. The lifecycle is the sequence that gets from one to the other, exactly once per execution, and brings measurements back.

Nothing happens until a terminal call. `filter`, `select`, `join`, `group_by` each return a
new {py:class}`Dataset <batcher.Dataset>` wrapping a new `LogicalPlan`. No data is read; no expression is evaluated.
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

Steps 2 through 6 are sequenced in exactly one place: `run_relational` in `python/batcher/api/orchestration/run.py`. Every relational terminal routes through it, whether single-node, distributed, or an adaptive stage, so the three subsystems are wired together once rather than at each call site. `api` is the only layer permitted to import all of Kyber, Carbonite, and Core. They cannot import each other.

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
{doc}`the execution engine page </architecture/internals/execution>`.

### 2 and 3. Optimize, then admit

Kyber rewrites the logical plan by pushing down predicates and projections, fusing operators, and choosing a join order. It lowers the result to a `PhysicalPlan` carrying per-operator `ResourceBounds` and cardinality estimates tagged with provenance. Carbonite reads those bounds and decides whether the plan's dominant materializing operator fits the memory envelope. If memory is what doesn't fit, the verdict is a spill-friendly counter-offer: the query is routed out of core instead of walking into an OOM. Any other binding constraint has no spill remedy, so `run.py` raises a `PlanError` before anything executes.

Neither subsystem touches data. Kyber decides, Carbonite protects, Core measures. The verbs
stay in their lanes because the subsystems cannot import one another.

Drawn as the ring it is, with the outcome of admission the list above flattens:

![The contract loop a terminal op drives, as a clockwise ring of four stations, each carrying the verb that keeps it in its lane. Kyber DECIDES, in kyber.optimize_full, and hands Carbonite a PhysicalPlan with a resource bound per operator. Carbonite PROTECTS, in carbonite.validate, and when the plan fits it reserves and runs it. Core MEASURES, in core.execute, and returns per-operator metrics: actual rows, time, peak bytes. The MetadataHub REMEMBERS, in collect_source_metadata, and the dashed edge from it back to Kyber is read on the next run, not this one. Admission has a third outcome besides pass and fail: a plan that will not fit in memory gets a counter-offer and is routed out-of-core, to spill and then run. Any other binding constraint raises instead, because spilling only ever answers a memory constraint.](/_static/diagrams/query_lifecycle.svg)

### 4 and 5. Lower and execute

`PhysicalPlan.to_json()` serializes the relational IR. Core calls the one FFI entry point:

```text
out, metrics_json = _native.execute_plan_metered(plan.to_json(), sources, engine_cfg, query_id)
```

`sources[i]` is the relation bound to `Scan { source_id: i }`, a list of pyarrow
`RecordBatch`es. `query_id` makes the execution cancellable. Two very different things cross here:

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
counts in and out, elapsed and CPU nanoseconds, peak and result bytes, and whether the operator spilled and by how much. Core records those into the `MetadataHub`, keyed by a *structural plan signature*. That signature is stable across executions, unlike an operator's position in one plan walk. Kyber reads the measurements on the next run, and, when the query runs in adaptive stages, at the next pipeline breaker during this one. That loop is what makes the optimizer improve the more a query shape is run.

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
query plan (planned)                          3 operators
─────────────────────────────────────────────────────────
OPERATOR                 ESTIMATE  NOTES
aggregate  [by g · sum]     est≈1  (default)
└─ filter  [x > 1]          est≈3  (default)
   └─ scan  [source 0]      est≈4  (exact)  pushed[x > 1]
```

`est≈4 (exact)` on the scan is the metadata layer: the row count of an in-memory relation is
known exactly. The two nodes above it carry `default` provenance, because nobody has measured a
selectivity for this predicate yet. Run the query a few times and Kyber will have. The predicate
itself is already inside the scan, which is what `pushed[x > 1]` records.
:::

`explain(format="json")` returns the same tree as a document, with the measured columns
filled in after an `analyze=True` run.

## What it costs

A small query's fixed cost is where this design most easily goes wrong. The table says when each cost is paid.

| Cost | Paid | Why |
|---|---|---|
| Cranelift compilation | once per distinct `(expr, schema, simd)`, process-wide | Memoized in `crates/bc-codegen/src/cache.rs`, so a small query doesn't pay it repeatedly. |
| Thread pool construction | once per width, cached | `par.rs::pool_for`. A one-row query does not spin up 96 threads; the worker count is capped by the morsels the inputs can produce. |
| Plan JSON parse | once per `execute_plan` call | Not per batch. |
| Arrow handoff | once per input relation | Zero-copy through the C Data Interface. |
| The metrics walk | once per metered run | Only on the metered entry point. |

On the operator benchmarks a global sum over TPC-H `lineitem` at scale factor 1, 6M rows on 16 cores, completes in 0.5 ms end to end against DuckDB's 2.7 ms. A query that does almost no work is the cleanest measure of the fixed overhead. See {doc}`the analytics benchmarks </benchmarks/results/analytics>` for the full table and the hardware.

## Where the code lives

The steps below are in the order a query passes through them, so the table doubles as a
reading path through the control plane:

| Step | Code |
|---|---|
| Dataset / terminal ops | `python/batcher/api/dataset/frame.py`, `python/batcher/api/terminal/` |
| The contract loop | `python/batcher/api/orchestration/run.py` |
| Metadata shortcut | `python/batcher/api/terminal/metadata_answer/` |
| Logical plan + `to_ir()` | `python/batcher/plan/logical/`, `python/batcher/plan/ir_tags.py` |
| Physical plan + `PhysicalPlan.to_json()` | `python/batcher/plan/physical.py` |
| Core's call into the engine | `python/batcher/core/executor.py` |
| The FFI boundary | `crates/bc-py/src/lib.rs` |
| The executor | `crates/bc-interp/src/lib.rs` (sequential), `par.rs` and `stream/` (multi-core) |

## Adaptive stages

One thing the diagram above flattens: a query can run steps 2 through 5 more than once. With `adaptive="auto"`, the default, a single-node query runs in stages only when it contains a join whose inputs are still sized by a guess and it clears a floor of 5 million rows or about 320 MB per pipeline breaker the loop would cut at. On a cluster, a plan the one-shot dispatcher can't run correctly stages at any size. Everything else runs the steps once.

Inside a staged run, each breaker's output is materialized and *measured*. If an estimate was off by more than `optimizer.reoptimize_error`, which defaults to 2.0, the remainder of the plan is re-optimized on the measured numbers and executed as the next stage. Relational state lives in `bc-runtime` and the JIT compiles only scalar expressions, from a process-wide cache, so re-planning loses no finished work.

## See also

- {doc}`Architecture </architecture/index>`: the three-subsystem shape this sequence wires together.
- {doc}`Execution engine </architecture/internals/execution>`: the architecture-level view of steps 4 through 6.
- {doc}`Kyber </architecture/internals/kyber>`: what step 2 actually does to the plan.
- {doc}`Carbonite </architecture/internals/carbonite>`: what step 3 admits against.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: how to read the `explain()` output above.
- {doc}`Performance </user-guide/operate/tuning/performance>`: applying all of this to a slow query.
- {doc}`Analytics benchmarks </benchmarks/results/analytics>`: where the 0.5 ms fixed-overhead figure comes from.
- {doc}`Plan IR </architecture/deep-dives/query/plan-ir>`: the JSON wire contract the boundary speaks.
- {doc}`Morsel parallelism </architecture/deep-dives/operators/morsel-parallelism>`: how a plan becomes work for N cores.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: why steps 2 through 5 can run more than once.
