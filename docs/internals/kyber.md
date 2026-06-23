# Kyber query optimizer

Kyber turns a logical plan into a better logical plan, then into a physical one. It
is an ordered list of rule- and cost-based passes (plan in, plan out), grouped by
family in `kyber/rules/` and registered through a `@rule` decorator into a
`RuleRegistry`. The architecture is built to grow to many rules, but the shipped set
is deliberately small, and every rule is proven semantics-preserving. Kyber decides;
it never executes and never collects runtime metadata.

This is a direct reaction to what the rewrite replaced: a "127 passes, 519 rules"
optimizer whose size was the problem, not the achievement. A rule earns its place by
making a query measurably better, not by padding a catalog.

## The phased pipeline

Rules run phase by phase, in a fixed order. The early rewrite phases iterate to a
fixpoint because their rules are confluent: applying them in any order converges to
the same plan. The cost-based and physical phases run once, since they make a
decision rather than converge to one.

| Phase | Runs | What it does |
|-------|------|--------------|
| `NORMALIZE` | to fixpoint | constant folding, expression simplification, canonicalization |
| `REWRITE` | to fixpoint | algebraic rewrites (e.g. redundant-distinct removal) |
| `PUSHDOWN` | to fixpoint | predicate, projection, and limit pushdown; column pruning |
| `JOIN_REORDER` | once | cost-based multi-table join ordering |
| `FUSION` | once | operator and top-N fusion, late materialization |
| `SELECTION` | once | physical algorithm choice (join build side, aggregate strategy) |
| `ENFORCE` | once | distribution/exchange enforcement and validation |

Each rule also carries a category (`REWRITE`, `SELECTION`, `ESTIMATION`,
`VALIDATION`, or `ENFORCE`) that drives `explain` output and telemetry, not control
flow.

## Shipped rules

The rules registered today are node-local transformations: a rule matches an
operator type and returns a rewritten subtree (or the input unchanged). Grouped by
the family module they live in:

- `normalize`: `constant_folding`, `expr_simplification`, `eliminate_identity_project`,
  `merge_projections`, `prune_true_filter`, `eliminate_sort_before_aggregate`.
- `pushdown`: `predicate_pushdown` and `projection_rewrite`, plus the structural
  pushdowns `merge_adjacent_filters`, `push_filter_through_project`,
  `push_filter_through_aggregate`, `push_filter_through_sort`, `push_filter_into_union`,
  `push_limit_through_project`, `push_limit_into_union`.
- `algebraic`: `remove_redundant_distinct`.
- `join_order`: cost-based multi-table ordering, using exact DP at or below
  `optimizer.join_dp_max_tables` tables (default 12), a greedy heuristic up to
  `greedy_max_tables` (25), and no reordering above that.
- `fusion`: `topn_fusion`, where a `Limit` over a `Sort` becomes a single top-N operator.
- `selection`: `adaptive_build_side`, the cost-based choice of which join input
  builds the hash table.

Adding a rule means dropping a function into the right family module and decorating
it with `@rule(name=..., phase=..., matches=...)`; the registry discovers it. See
the `add-kyber-optimizer-pass` recipe.

## Cost and cardinality

Cost-based phases need to compare plans. Kyber's cost model collapses three axes —
CPU, I/O, and network — into one scalar, weighting network shuffle bytes more
heavily than local bytes (`optimizer.cost_weights.net` defaults to 2× the others).
Per-operator costs come from `optimizer.cost_coeffs` (for example, inserting a row
into a hash table costs more than probing one), and they are recalibrated from
measured operator times once enough samples accumulate, clamped so timing noise
cannot produce a degenerate model.

Cardinality estimation drives those costs. Before anything is learned, Kyber falls
back to Selinger-style selectivities (`col = literal` passes 10% of rows, a range
predicate a third, `IS NULL` 5%). Sketches built during execution (HyperLogLog for
distinct counts, KLL for quantiles) and learned per-query statistics in the
MetadataHub supersede those defaults and sharpen across runs.

## Adaptive re-optimization

This is the part static optimizers cannot do. An estimate is only a guess until the
query runs; at a pipeline breaker, the engine has *measured* the real size of what it
just processed. Core records those measurements, and when an estimate was off by more
than `optimizer.reoptimize_error` (default 2×), Kyber re-plans the rest of the query
on the measured numbers. The same mechanism works single-node and distributed.

DuckDB optimizes once, before execution. Spark AQE adapts only at stage boundaries.
Kyber adapts at every breaker, which is the moat.

## Using it

You rarely call Kyber directly — it runs automatically on every terminal operation:

```python
import batcher as bt

ds = bt.read("data.parquet").filter(bt.col("value") > 100).select("id", "name", "value")
result = ds.collect()  # Kyber optimizes here, then the engine runs the plan
```

To see the optimized plan without running it, use `explain`:

```python
print(ds.explain())
```

The public optimizer surface (`batcher.kyber`) is small: `optimize` and
`optimize_traced` run the pipeline (the latter returns the per-rule decision log),
and the learning entry points (`record_execution`, `record_column_stats`,
`record_selectivity`, `load_learned_stats`) feed the MetadataHub that later plans
read from.

## See also

- [Carbonite](carbonite.md) — checks the feasibility of the plan Kyber produces
- [Execution engine](execution.md) — runs it and measures what Kyber re-plans on
- [Configuration options](../configuration/options.md) — the cost-model and cardinality knobs
