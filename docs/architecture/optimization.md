# Query optimization

This page describes what Batcher's optimizer, Kyber, does to a query: the phases it
runs, the rewrites each phase applies, and how it re-plans on measured numbers.

Kyber rewrites a logical plan into a better one and then lowers it to a physical plan.
It's an ordered set of passes rather than an unstructured catalog. A rule ships only
when it makes a query measurably better, and every rule is proven semantics-preserving.
The optimizer runs automatically on every terminal operation, so the plan you describe
and the plan that runs differ, but the result doesn't.

The authoritative model, covering the rule families, cost coefficients, and
configuration knobs, lives in {doc}`the Kyber reference <../internals/kyber>`.

## The phased pipeline

Rules run phase by phase, in a fixed order. The early rewrite phases iterate to a
fixpoint: their rules are confluent, so applying them in any order converges to the
same plan. The cost-based and physical phases run once, because they make a decision
rather than converge to one.

| Phase | Runs | What it does |
|-------|------|--------------|
| `NORMALIZE` | to fixpoint | constant folding, expression simplification, canonicalization |
| `REWRITE` | to fixpoint | algebraic rewrites (e.g. redundant-distinct removal) |
| `PUSHDOWN` | to fixpoint | predicate, projection, and limit pushdown; column pruning |
| `JOIN_REORDER` | once | cost-based multi-table join ordering |
| `FUSION` | once | operator and top-N fusion |
| `SELECTION` | once | physical algorithm choice (join build side) |
| `ENFORCE` | once | distribution/exchange enforcement and validation |

## What the passes do

The sections below walk the phases in the order they run, from the cheap syntactic
rewrites through the cost-based decisions that need an estimate to make.

### Constant folding and simplification

The `NORMALIZE` phase evaluates constant expressions at plan time and drops algebraic
identities such as `x + 0`, `x * 1`, an always-true filter, and an identity projection.
This shrinks the plan before any later pass reasons about it, and it collapses
expressions you wrote for clarity rather than for the engine's benefit.

### Predicate pushdown

Filters move toward the data source. A predicate that can run earlier reads less,
because the source skips data that would be discarded anyway, which cuts I/O and the
memory the rest of the pipeline carries.

```python
ds = bt.read("data.parquet").filter(bt.col("year") == 2024)
```

Kyber pushes the filter through projections, aggregates, sorts, and unions, splits
conjunctions so each part lands as early as it legally can, and merges adjacent
filters into one. For Parquet, a pushed predicate lets the reader skip row groups
whose statistics rule them out, and skip partitions entirely when the column is a
partition key. On a selective scan that can cut the work by orders of magnitude.

### Projection and column pruning

Only the columns a query actually uses are read and carried. Kyber tracks column
dependencies through the whole pipeline, including columns referenced only inside
expressions, and prunes the rest. On a wide table read column-by-column from
Parquet, selecting two of fifty columns reads two.

```python
ds = bt.read("wide_table.parquet").select("id", "name")
```

Pruning works through intermediates, not just at the scan: a column computed and then
never read in the final result is dropped, and the inputs that fed only that column
are dropped with it.

```python
ds = (
    bt.read("data.parquet")
    .with_columns(total=bt.col("price") * bt.col("quantity"))
    .select("id", "total")  # only id, price, quantity are ever read
)
```

### Limit pushdown

A limit pushes as early as the pipeline's semantics allow, so the engine can stop
once it has enough rows instead of producing the full intermediate. Kyber pushes
limits through projections and into the branches of a union.

### Top-N fusion

A `Limit` over a `Sort` is the special case worth its own operator. Sorting the whole
input only to take the first N rows is wasted work, so Kyber fuses the pair into a
single top-N operator that keeps only N rows in flight.

```python
ds = ds.sort("score", descending=True).limit(100)  # fused into top-N
```

### Join reordering

Join order dominates the cost of a multi-table query, because the wrong order
materializes a large intermediate that a better order never builds. Kyber reorders
joins cost-based, minimizing the estimated intermediate sizes. The search is exact
dynamic programming at or below `optimizer.join_dp_max_tables` tables, 12 by default,
and a greedy heuristic up to `optimizer.greedy_max_tables`, 25 by default. Beyond that
count, exhaustive search stops paying for itself and Kyber leaves the order alone.

```python
result = table_a.join(table_b, on="key").join(table_c, on="key")
```

### Join build-side selection

The hash join builds a table on one input and probes it with the other. Building the
smaller side keeps that table in memory and the larger side streaming, so the
`SELECTION` phase compares estimated input sizes and picks the build side, swapping
the inputs when that helps. When one side is small enough, it is broadcast rather
than shuffled.

## Adaptive re-optimization

Every estimate above is a guess until the query runs. At a stage boundary, which is a
pipeline breaker such as a sort, an aggregate, or a join build, the engine has
*measured* the real size of what it just processed. Core records that measurement, and
when an estimate was off by more than `optimizer.reoptimize_error`, 2.0 by default,
Kyber re-plans the rest of the query on the measured numbers before continuing. The
same mechanism runs single-node and distributed.

This is stage-boundary re-optimization, the same granularity Spark AQE adapts at, and
the difference is that Batcher runs it on one machine too. DuckDB optimizes once,
before execution, and never revises. The loop is gated by `adaptive="auto"`, the
default, which engages it only when measuring could flip a downstream decision such as
a build side or a join order. Kyber also carries a cross-query loop that neither DuckDB
nor Spark has: sketch-backed statistics and a bandit over join strategies, both
persisted between runs.

This split is the reason the architecture keeps Core, which measures, and Kyber, which
decides, as separate subsystems with a feedback loop between them.

## Cost and cardinality

The cost-based phases compare candidate plans against one scalar cost. Kyber's model
collapses three axes, CPU, I/O, and network, into that single number. It weights
network shuffle bytes more heavily than local bytes, because moving data between
workers costs more than touching it locally, and `optimizer.cost_weights.net` defaults
to 2.0 against 1.0 for the other two. Per-operator coefficients come from
`optimizer.cost_coeffs` and are recalibrated from measured operator times once enough
samples accumulate, clamped so timing noise can't skew the model.

Those costs ride on cardinality estimates. With nothing learned yet, Kyber uses
Selinger-style selectivities: `col = literal` passes 10% of rows, a range predicate a
third, and `IS NULL` 5%. Sketches built during execution, HyperLogLog for distinct
counts and KLL for quantiles, together with learned per-query statistics in the
MetadataHub, supersede those defaults and sharpen the estimates each time a query runs.

## Viewing the optimized plan

`explain` runs the optimizer and returns the resulting plan with per-node cardinality
estimates and the join build-side decisions Kyber made, without executing anything:

```python
ds = bt.read("data.parquet").filter(bt.col("status") == "active").select("id", "total")
print(ds.explain())
```

The output is the optimized plan tree annotated with estimated row counts and the
provenance of each estimate, which is a default, a sketch, or a learned statistic,
followed by any build-side swaps. This is how you confirm a predicate landed at the
scan, or that a join was reordered the way you expected.

## See also

- {doc}`Kyber reference <../internals/kyber>`: the rule families, cost coefficients, and knobs.
- {doc}`Architecture overview <overview>`: the control-plane and data-plane split.
- {doc}`Execution model <execution>`: the breakers the adaptive loop measures at.
- {doc}`Configuration options <../configuration/options>`: the cost-model and cardinality settings.
