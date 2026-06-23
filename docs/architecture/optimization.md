# Query optimization

Batcher's optimizer, Kyber, rewrites a logical plan into a better one and then
lowers it to a physical plan. It is a small, ordered set of passes, not a catalog
of hundreds of rules. The rewrite that produced v2 deleted exactly that kind of
sprawl; a rule ships only when it makes a query measurably better, and every rule is
proven semantics-preserving. The optimizer runs automatically on every terminal
operation, so the plan you describe and the plan that runs differ, but the result
does not.

This page is the architectural tour. The authoritative model (the rule families,
cost coefficients, and configuration knobs) lives in
[the Kyber reference](../internals/kyber.md).

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

### Constant folding and simplification

The `NORMALIZE` phase evaluates constant expressions at plan time and drops
algebraic identities — `x + 0`, `x * 1`, an always-true filter, an identity
projection. This shrinks the plan before any later pass reasons about it, and it
collapses expressions the user wrote for clarity rather than the engine's benefit.

### Predicate pushdown

Filters move toward the data source. A predicate that can run earlier reads less:
the source skips data that would be discarded anyway, which cuts I/O and the memory
the rest of the pipeline carries.

```python
ds = bt.read("data.parquet").filter(bt.col("year") == 2024)
```

Kyber pushes the filter through projections, aggregates, sorts, and unions, splits
conjunctions so each part lands as early as it legally can, and merges adjacent
filters into one. For Parquet, a pushed predicate lets the reader skip row groups
whose statistics rule them out, and skip partitions entirely when the column is a
partition key — which can cut a selective scan by orders of magnitude.

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
dynamic programming at or below `optimizer.join_dp_max_tables` tables (12 by
default), a greedy heuristic up to 25, and no reordering beyond that, the point
where exhaustive search stops paying for itself.

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

This is the part a static optimizer cannot do. Every estimate above is a guess until
the query runs. At a pipeline breaker — a sort, an aggregate, a join build — the
engine has *measured* the real size of what it just processed. Core records that
measurement, and when an estimate was off by more than `optimizer.reoptimize_error`
(2× by default), Kyber re-plans the rest of the query on the measured numbers before
continuing. The same mechanism runs single-node and distributed.

DuckDB optimizes once, before execution. Spark AQE adapts only at stage boundaries.
Kyber adapts at every breaker. That is the moat — and the reason the architecture
keeps Core (which measures) and Kyber (which decides) as separate subsystems with a
feedback loop between them.

## Cost and cardinality

The cost-based phases compare candidate plans against one scalar cost. Kyber's model
collapses three axes — CPU, I/O, and network — into that single number, weighting
network shuffle bytes more heavily than local bytes (`optimizer.cost_weights.net`
defaults to 2× the others), since moving data between workers costs more than
touching it locally. Per-operator coefficients come from `optimizer.cost_coeffs` and are
recalibrated from measured operator times once enough samples accumulate, clamped so
timing noise cannot skew the model.

Those costs ride on cardinality estimates. With nothing learned yet, Kyber uses
Selinger-style selectivities: `col = literal` passes 10% of rows, a range predicate a
third, `IS NULL` 5%. Sketches built during execution — HyperLogLog for distinct
counts, KLL for quantiles — and learned per-query statistics in the MetadataHub
supersede those defaults and sharpen the estimates each time a query runs.

## Viewing the optimized plan

`explain` runs the optimizer and returns the resulting plan with per-node cardinality
estimates and the join build-side decisions Kyber made, without executing anything:

```python
ds = bt.read("data.parquet").filter(bt.col("status") == "active").select("id", "total")
print(ds.explain())
```

The output is the optimized plan tree annotated with estimated row counts and the
provenance of each estimate (a default, a sketch, or a learned statistic), followed
by any build-side swaps. This is how you confirm a predicate landed at the scan or a
join was reordered the way you expected.

## See also

- [Kyber reference](../internals/kyber.md) — the rule families, cost coefficients, and knobs
- [Architecture overview](overview.md) — the control-plane / data-plane split
- [Configuration options](../configuration/options.md) — the cost-model and cardinality settings
