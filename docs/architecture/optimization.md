# Query optimization

This page describes what Batcher's optimizer, Kyber, does to a query: the phases it
runs, the rewrites each phase applies, and how it re-plans on measured numbers.

Kyber rewrites a logical plan into a better one and lowers it to a physical plan, automatically, on every terminal operation. The plan you describe and the plan that runs differ. The result doesn't: property tests run the full rule set and check that it changes the plan and never the answer, and that it converges to a deterministic fixpoint.

The examples on this page share one small dataset:

```python
# docs: run
import batcher as bt

ds = bt.from_pydict({
    "id": [1, 2, 3, 4],
    "year": [2023, 2024, 2024, 2025],
    "score": [7, 9, 3, 5],
    "price": [2.0, 3.0, 4.0, 1.0],
    "quantity": [5, 1, 2, 3],
})
```

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

:::{dropdown} Why a canonicalization round runs after fusion
The pipeline is a single forward pass. A rule that collapses a shape, such as folding two adjacent filters into one conjunction, runs early and then never sees the plan again, while pushdown, join reordering and fusion can re-create that shape. The round re-applies exactly those contracting rewrites once the plan's structure has settled. See {doc}`internals/kyber` for what qualifies a rule to take part and why the round must sit before `SELECTION`.
:::

## What the passes do

The sections below walk the phases in the order they run, from the cheap syntactic
rewrites through the cost-based decisions that need an estimate to make.

### Constant folding and simplification

The `NORMALIZE` phase evaluates constant expressions at plan time and drops algebraic identities such as `x + 0`, `x * 1`, an always-true filter, and an identity projection. The plan shrinks before any later pass reasons about it.

### Predicate pushdown

Filters move toward the data source, so the source skips data that would be discarded anyway. The pushed predicate shows on the scan line of `explain()`:

```python
# docs: run
plan = ds.filter(bt.col("year") == 2024).explain()
assert "pushed[year = 2024]" in plan
```

Kyber pushes a filter through projections, aggregates, sorts, and unions, splits conjunctions so each part lands as early as it legally can, and merges adjacent filters. For Parquet, a pushed predicate skips row groups whose statistics rule them out, and whole partitions when the column is a partition key.

:::{dropdown} What a source can and can't push
A source pushes what its backend can express, and no more. Every backend has terms it
cannot spell. A database has no portable literal for `NaN`, and a Parquet reader will not
prune on a temporal literal whose physical unit it cannot verify. When one term of an
`AND` is untranslatable, the rest are still pushed, because dropping a conjunct only
widens what the source returns and the engine keeps its own `Filter` to re-check every
row. An `OR` is the opposite case and pushes all or nothing: dropping a disjunct would
narrow the filter and lose rows that never crossed the wire.
:::

### Projection and column pruning

Only the columns a query uses are read and carried. Kyber tracks column dependencies through the whole pipeline, including columns referenced only inside expressions, so selecting two of fifty Parquet columns reads two. Pruning works through intermediates too: a computed column that the result never reads is dropped, along with the inputs that fed only it.

```python
# docs: run
q = ds.with_columns(total=bt.col("price") * bt.col("quantity")).select("id", "total")
print(q.to_pydict())  # only id, price and quantity are read
# {'id': [1, 2, 3, 4], 'total': [10.0, 3.0, 8.0, 3.0]}
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
# docs: run
top = ds.sort("score", descending=True).limit(2)
assert "top 2 by score" in top.explain()
print(top.select("id", "score").to_pydict())
# {'id': [2, 1], 'score': [9, 7]}
```

### Join reordering

Join order dominates the cost of a multi-table query. The wrong order materializes a
large intermediate that a better order never builds. Kyber reorders
joins cost-based, minimizing the estimated intermediate sizes, using dynamic
programming over connected subsets of the join graph and falling back to a greedy
builder when the graph is too large or too dense to search.

How hard Kyber searches is decided per query: it prices the join region and grants a share of that estimated cost back as search time. {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>` has the measurements behind the budget.

```python
# docs: run
a = bt.from_pydict({"key": [1, 2, 3], "a": [1, 2, 3]})
b = bt.from_pydict({"key": [1, 2], "b": [5, 6]})
c = bt.from_pydict({"key": [2, 3], "c": [7, 8]})
abc = a.join(b, on="key").join(c, on="key")
assert "join build side" in abc.explain()
print(abc.to_pydict())
# {'key': [2], 'a': [2], 'b': [6], 'c': [7]}
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

This is stage-boundary re-optimization, the same granularity Spark AQE adapts at, run inside the Python process. On a single node, `adaptive="auto"`, the default, engages the loop on a joined query, where measuring could still flip a build side or a join order, once it clears 5 million rows or about 320 MB per pipeline breaker the loop would cut at. Kyber also carries a cross-query loop that neither DuckDB nor Spark has: sketch-backed statistics, calibrated cost coefficients, and a bandit over join strategies, all persisted between runs. That loop is why Core, which measures, and Kyber, which decides, are separate subsystems.

## Cost and cardinality

The cost-based phases compare candidate plans against one scalar cost that collapses CPU, I/O, and network. Network shuffle bytes weigh double, because moving data between workers costs more than touching it locally:

```python
# docs: run
print(bt.active_config().optimizer.cost_weights)
# CostWeights(cpu=1.0, io=1.0, net=2.0)
```

Per-operator coefficients come from `optimizer.cost_coeffs` and are recalibrated from measured operator times once enough samples accumulate, clamped so timing noise can't skew the model.

Those costs ride on cardinality estimates. With nothing learned yet, Kyber uses
Selinger-style selectivities: `col = literal` passes 10% of rows, a range predicate a
third, and `IS NULL` 5%. Sketches built during execution, HyperLogLog for distinct
counts and KLL for quantiles, together with learned per-query statistics in the
MetadataHub, supersede those defaults and sharpen the estimates each time a query runs.

## Viewing the optimized plan

`explain` runs the optimizer and returns the plan with per-node cardinality estimates and the decisions Kyber made, without executing anything:

```python
# docs: run
print(ds.filter(bt.col("year") == 2024).select("id").explain())
```

Each estimate carries its provenance: `exact`, `histogram`, `sketch`, `learned`, or `default`. Predicates pushed into a scan show as `pushed[...]` on the scan line.

## See also

- {doc}`Kyber reference </architecture/internals/kyber>`: the rule families, cost coefficients, and knobs.
- {doc}`Architecture overview <overview>`: the control-plane and data-plane split.
- {doc}`Execution model <execution>`: the breakers the adaptive loop measures at.
- {doc}`Cardinality estimation </architecture/deep-dives/adaptive/cardinality-estimation>`: where each estimate comes from.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: the stage loop and its gate.
- {doc}`Configuration options <../configuration/options>`: the cost-model and cardinality settings.
