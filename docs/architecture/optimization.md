# Query optimization

This page describes what Batcher's optimizer, Kyber, does to a query: the phases it
runs, the rewrites each phase applies, and how it re-plans on measured numbers.

Kyber rewrites a logical plan into a better one and then lowers it to a physical plan.
It's an ordered set of phases rather than an unstructured catalog of rules. Property
tests run the full rule set and check that it changes the plan and never the answer, and
that it converges to a deterministic fixpoint. The optimizer runs automatically on every
terminal operation, so the plan you describe and the plan that runs differ, but the
result doesn't.

The authoritative model, covering the rule families, cost coefficients, and
configuration knobs, lives in {doc}`the Kyber reference </architecture/internals/kyber>`.

## The phased pipeline

Rules run phase by phase, in a fixed order. The early rewrite phases iterate to a
fixpoint: their rules are confluent, so applying them in any order converges to the
same plan. The cost-based and physical phases run once, because they make a decision
rather than converge to one.

| Phase | Runs | What it does |
|-------|------|--------------|
| `NORMALIZE` | to fixpoint | constant folding, expression simplification, canonicalization, common subexpressions |
| `REWRITE` | to fixpoint | subquery decorrelation, set-operation rewrites, CTE handling |
| `PUSHDOWN` | to fixpoint | predicate, projection, and limit pushdown; partition pruning |
| `JOIN_REORDER` | once | cost-based multi-table join ordering |
| `FUSION` | to fixpoint | operator fusion, top-N fusion, late materialization |
| *canonicalization round* | to fixpoint | the contracting rewrites, re-run after fusion |
| `SELECTION` | once | physical algorithm choice, such as the join build side and aggregate strategy |
| `ENFORCE` | once | distribution and exchange enforcement, validation |

The canonicalization round exists because the pipeline is a single forward pass. A rule that
collapses a shape, such as folding two adjacent filters into one conjunction, runs early and
then never sees the plan again. Pushdown, join reordering, and fusion all re-create that
shape after it has run. The round re-applies exactly those contracting rewrites once the
plan's structure has settled, so the engine is not handed an operator the optimizer already
knew how to remove. See {doc}`internals/kyber` for what qualifies a rule to take part and why
the round must sit before `SELECTION`.

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
partition key. On a selective scan, most of the file is never decoded.

A source pushes what its backend can express, and no more. Every backend has terms it
cannot spell. A database has no portable literal for `NaN`, and a Parquet reader will not
prune on a temporal literal whose physical unit it cannot verify. When one term of an
`AND` is untranslatable, the rest are still pushed, because dropping a conjunct only
widens what the source returns and the engine keeps its own `Filter` to re-check every
row. An `OR` is the opposite case and pushes all or nothing: dropping a disjunct would
narrow the filter and lose rows that never crossed the wire.

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

A limit pushes as early as the pipeline's semantics allow. The engine can then stop once
it has enough rows instead of producing the full intermediate, and Kyber pushes limits
through projections and into the branches of a union.

### Top-N fusion

A `Limit` over a `Sort` is the special case worth its own operator. Sorting the whole
input to take the first N rows is wasted work. Kyber fuses the pair into a single top-N
operator that keeps only N rows in flight.

```python
ds = ds.sort("score", descending=True).limit(100)  # fused into top-N
```

### Join reordering

Join order dominates the cost of a multi-table query. The wrong order materializes a
large intermediate that a better order never builds. Kyber reorders
joins cost-based, minimizing the estimated intermediate sizes, using dynamic
programming over connected subsets of the join graph and falling back to a greedy
builder when the graph is too large or too dense to search.

How hard Kyber searches is decided per query rather than by a fixed table count. The
optimizer prices the join region, grants a share of that estimated cost back as search
time, and spends it in units of candidate join pairs. A query too cheap to repay the
search stops early and takes the greedy order, and a query large enough to repay far
more searching gets it. See
{doc}`Cost model </architecture/deep-dives/adaptive/cost-model>` for the measurements
behind the budget.

```python
result = table_a.join(table_b, on="key").join(table_c, on="key")
```

### Join build-side selection

The hash join builds a table on one input and probes it with the other. Building the
smaller side keeps that table in memory and the larger side streaming, so the
`SELECTION` phase compares estimated input sizes and picks the build side, swapping
the inputs when that helps. When one side fits under `optimizer.broadcast_max_bytes`,
it is broadcast rather than shuffled. The default of `0` sizes that threshold from the
machine's last-level cache.

## Adaptive re-optimization

Every estimate above is a guess until the query runs. At a stage boundary, which is a
pipeline breaker such as a sort, an aggregate, or a join build, the engine has
*measured* the real size of what it just processed. Core records that measurement, and
when an estimate was off by more than `optimizer.reoptimize_error`, 2.0 by default,
Kyber re-plans the rest of the query on the measured numbers before continuing. The
same mechanism runs single-node and distributed.

This is stage-boundary re-optimization, the same granularity Spark AQE adapts at, and
Batcher runs it on one machine too. DuckDB optimizes once, before execution, and never
revises. Check the gate before assuming your query is in it. On a single node,
`adaptive="auto"`, the default, engages the loop only on a query that contains a join,
where measuring could still flip a downstream decision such as a build side or a join
order, and that clears a floor of 5 million rows or about 320 MB for each pipeline breaker
the loop would cut at. A query with no join is out at any size. On a cluster, a plan the
one-shot dispatcher can't run correctly takes the staged path regardless.
Kyber also carries a cross-query loop that neither DuckDB nor Spark has: sketch-backed
statistics, calibrated cost coefficients, and a bandit over join strategies, all persisted
between runs.

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
provenance of each estimate: `exact`, `histogram`, `sketch`, `learned`, or `default`.
Predicates pushed into a scan show as `pushed[...]` on the scan line. This is how you confirm a predicate landed at the
scan, or that a join was reordered the way you expected.

## See also

- {doc}`Kyber reference </architecture/internals/kyber>`: the rule families, cost coefficients, and knobs.
- {doc}`Architecture overview <overview>`: the control-plane and data-plane split.
- {doc}`Execution model <execution>`: the breakers the adaptive loop measures at.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: where each estimate comes from.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: the stage loop and its gate.
- {doc}`Configuration options <../configuration/options>`: the cost-model and cardinality settings.
