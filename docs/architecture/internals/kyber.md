# Kyber query optimizer

Kyber turns a logical plan into a better logical plan, then into a physical one. It
is an ordered list of rule- and cost-based passes (plan in, plan out), grouped by
family in `kyber/rules/` (and its `kyber/rules/extra/` subpackage) and registered
through a `@rule` decorator into a `RuleRegistry`. Every rule is proven
semantics-preserving in isolation, and the *combination* is property-tested for
result-invariance and confluence. Kyber decides; it never executes and never
collects runtime metadata.

This is a deliberate reaction to what the rewrite replaced: a "127 passes, 519
rules" optimizer whose sprawl was the problem rather than the achievement, with a
god-pass in one place and a one-file-per-rule catalog in another. The roughly 390
rules Kyber ships are *not* a return to that bloat. They are the sanctioned "many
small things" pattern: small, node-local rewrites grouped by family, one module per
family a user names in one breath, discovered by a registry, each earning its place
by making a query measurably better. Unlike v1, each is individually proven correct
and the whole set is proven not to interfere. *Correctness of the rule set*, below,
is where that proof lives.

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

Most rules are node-local transformations. A rule matches an operator type and
returns a rewritten subtree, or the input unchanged, and the driver supplies the
bottom-up traversal and fixpoint iteration. The few holistic and cost-based rewrites
(column pruning, join reordering, build-side selection) reason over the whole tree
at once. The core families live in `kyber/rules/`:

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

### The `extra/` families

The bulk of the rule count lives in `kyber/rules/extra/`, a subpackage of
grouped-by-family modules. It exists because the parent `rules/` directory hit the
12-files-per-directory structure cap. The subpackage keeps each family a small,
self-contained module, allowlisted in `tools/lint_structure.py` as the sanctioned
large-rule-set pattern, rather than flattening names into a god file. Each rule is
node-local and individually differential-tested against DuckDB.

Each row below is one module, named for the shape it rewrites:

| Family module | What it rewrites |
|---|---|
| `boolean_algebra` | boolean / CASE / COALESCE / NULL algebra: annihilators (`x AND FALSE`), idempotence, absorption, complementation on total predicates, `NOT` through a comparison, `x = TRUE → x`, single/duplicate `IN` lists |
| `sargable` | strips arithmetic wrappers so `col + k = lit` becomes a bare `col = lit - k` that zone-map pruning and source pushdown can use |
| `arith_algebra` | integer constant reassociation and factoring inside a `Filter`/`Project` |
| `temporal_sargable` | rewrites temporal-extraction predicates such as `year(ts) = 2020` into sargable `[lo, hi)` ranges on the raw column |
| `predicate_infer` | inference over a `Filter`'s top-level conjunction: contradictions, redundant conjuncts, implied bounds |
| `join_extra` | structural join rewrites that fire on join *shape* rather than statistics, such as a join whose result is provably fixed |
| `setops` | `UNION` / `DISTINCT` structural simplifications |
| `topn_limit` | `LIMIT` / `OFFSET` shapes the base limit rules don't already cover |
| `agg_extra` | local `Aggregate` / `GROUP BY` simplifications |
| `projection_scan` | projection, ordering, and scan/schema cleanups |
| `window_rules` | canonicalize a `Window`'s partition/order keys and prune dead window output |
| `empty_relation` | fold a provably-empty subtree (constant-`FALSE` filter, empty input to `Project`/grouped `Aggregate`/`Window`) to the canonical `Limit(x, 0)` marker |
| `adaptive_meta` | simplifications a provably-EXACT *cardinality* unlocks (dead limits, empty inputs) |
| `metadata_adaptive` | simplifications EXACT per-column metadata unlocks: skip a `Sort` over ≤ 1 row, prune constant sort keys, drop a `Distinct` over a proven-unique column, decide a `col OP col` filter from both columns' bounds |

The two metadata families are gated on `Provenance.EXACT` proof only. A learned or
sketch estimate can never drive them, because dropping a `Distinct` or `Sort` on a
wrong guess is silent data corruption.

### The `exprs/` and `relational/` families

Two later subpackages sit beside `extra/`. `kyber/rules/exprs/` holds expression
algebra grouped by the value domain it rewrites, and `kyber/rules/relational/` holds
rewrites that move a plan node rather than change an expression. The split keeps each
directory inside the file-count cap and makes a rule findable from the shape it acts
on. The modules and what each one rewrites:

| Family module | What it rewrites |
|---|---|
| `exprs/numeric` | identities whose soundness depends on the operand's type and nullability: `x // 1`, `x * 0`, `x % 1`, `pow(x, 0)`, `hypot(x, 0)`, and the `gcd`/`lcm`/shift folds |
| `exprs/cast_unwrap` | Spark's `UnwrapCastInBinaryComparison` -- lifts a widening `int -> float` cast out of a comparison against a literal, which is what lets zone-map pruning and source pushdown see the predicate at all |
| `exprs/boolean_normalize` | drives `NOT` down to the leaves: both De Morgan laws (verified over the full Kleene cross-product) and double negation, so the comparisons underneath reach `fold_not_comparison` |
| `exprs/comparisons` | the reflexive six, `x = x` through `x >= x`, folded to a constant on a non-nullable non-float operand |
| `exprs/conditionals` | moving work across a `CASE`: pushing a foldable comparison into literal branches, flattening a nested `ELSE` ladder, unwrapping a boolean-branch `CASE` inside a filter, pruning a dominated `GREATEST`/`LEAST` literal |
| `exprs/complex_types` | extract-over-construct (`make_struct(...).a`, `[x, y][0]`, `len([x, y])`) and list-function algebra (idempotence, involution, slice composition) |
| `exprs/text_folds` | constant folding for the string functions -- lengths, the digests (`md5`/`sha1`/`sha256`/`crc32`), `hex`, the pads, `repeat`, `initcap`, and the trims, each verified against the engine and ASCII-guarded where Unicode case or whitespace handling could differ |
| `exprs/text_algebra` | the remaining regex de-specializations (`regexp_replace_all` to `replace`, `regexp_split` to `split`) and composing stacked `substr` calls |
| `exprs/text` | regex de-specialization, where a metacharacter-free `regexp_matches` becomes `contains`, `starts_with`, `ends_with`, or `=`, plus the string identities `reverse(reverse(x))`, `repeat(x, 1)`, and a full-range `substr` |
| `exprs/temporal` | reading a date part through a finer truncation (`year(date_trunc('day', t))`), `last_day` idempotence, and day/microsecond offset fusion |
| `streaming/windows` | collapses nested event-time window alignment (`window(window(t, 5m), 15m)`) when the outer width is a whole multiple of the inner -- per-row work removed from a pipeline that never ends |
| `relational/windows` | transposing two independent `Window` nodes into a canonical spec order so the collapse rule can find them, and pushing a top-N below an unpartitioned ranking window |

Two constraints shape what these families can contain, and both are worth knowing
before adding a rule here.

There is no null *literal* in the IR: `Lit` rejects `None`, so a rule that matches
`NULLIF(x, NULL)` would be unreachable and one that produces a null literal would emit
a plan that fails to serialize. That rules out most "fold this call to a constant"
rewrites over a nullable input, because the constant would have to be null on the null
rows.

Anything that trades `NULL` for `FALSE` is sound only at the top level of a `Filter`,
which keeps a row just when the predicate is true. Those rules apply their rewrite to
`split_conjuncts` of the predicate rather than walking the expression tree, since one
`NOT` or `OR` deeper the two values are distinguishable.

Some identities that look safe are refused, and the unit suite pins each refusal.
`pow(x, 1)` is not folded, because libm's `pow` is not correctly rounded and the
identity is therefore not provable, while `pow(x, 0)` is folded because IEEE 754
specifies it as exactly one for every base. No timezone-conversion rewrite exists at
all: a same-zone conversion is not the identity, since the engine nulls DST-ambiguous
and nonexistent local times, and dropping the call would resurrect those rows' values.

Adding a rule means dropping a function into the right family module and decorating
it with `@rule(name=..., phase=..., matches=...)`, and the registry discovers it. See
the `add-kyber-optimizer-pass` recipe and {doc}`/architecture/internals/extending`.

### Correctness of the rule set

A large rule set is only safe if two things hold. Each rule must preserve results,
and no two rules may interfere in combination. Kyber proves both mechanically.

- **Each rule, individually.** Every rewrite carries a `tests/unit/` plan-shape test
  (the *plan* changes) and a `tests/differential/` test (the *answer* still matches
  DuckDB), across nulls, empties, and type edges.
- **The whole set, in combination.** `tests/property/test_prop_optimizer_result_invariance.py`
  (Hypothesis) generates a random table and a random-but-valid pipeline and asserts
  `result(full rule set) == result(no rules) == ds.collect()` under an
  order-independent multiset compare, so the subtle rule interaction an example misses
  falls out as a counterexample. The same suite proves **confluence and termination**:
  the combined set converges to a deterministic plan fixpoint within the *production*
  `optimizer.fixpoint_iterations` budget, so the rules provably do not oscillate or
  fight each other.

Two correctness constraints bound what the algebraic families may safely do, and the
rules are written to respect them:

- **Wrapping i64 arithmetic.** The engine's integer `+`, `-`, and `*` wrap on overflow,
  bit-for-bit with the Cranelift JIT. So `sargable` and `arith_algebra` reduce only
  `=` and `<>`, which are wrap-invariant bijections of the 64-bit integers, and never an
  *ordered* comparison, because wrapping breaks monotonicity. Every folded literal is
  guarded to stay in i64.
- **Three-valued (Kleene) logic.** `NULL AND FALSE = FALSE`, `NULL OR TRUE = TRUE`,
  and a null comparison is null. `boolean_algebra` proves each law under all three
  values (`T`, `F`, and `N`) and guards anything that would hold only for non-null
  operands. `x AND NOT x → FALSE`, for instance, fires only on never-null predicates
  such as `is_null`.

## Cost and cardinality

Cost-based phases need to compare plans. Kyber's cost model collapses three axes,
CPU, I/O, and network, into one scalar, weighting network shuffle bytes more
heavily than local bytes. `optimizer.cost_weights.net` defaults to twice the others.
Per-operator costs come from `optimizer.cost_coeffs`, where inserting a row into a
hash table costs more than probing one. They are recalibrated from measured operator
times once enough samples accumulate, clamped so timing noise cannot produce a
degenerate model.

### What a shuffled byte costs on this cluster

The network axis is not one price. A hash exchange sends each row to one of `W`
buckets, and on a fleet of dense nodes many of those buckets are on the producer's own
host, where a byte moves at host-memory or NVLink rate rather than at the NIC's.
Thirty-two devices on four eight-GPU hosts and thirty-two devices on thirty-two
one-GPU hosts report the same device count and the same worker count, and their
exchanges differ by a factor of eight.

`HardwareProfile.cluster` carries the fleet's shape into the optimizer: one record per
node with its cores, devices, coherent-fabric width, rack, and egress rate. From it
Kyber derives what share of an exchange stays inside a worker, a NVLink domain, a
host, and a rack, prices each tier against the node's measured fabric rate, and
charges the network axis the weighted result. A shuffle that never leaves a host stops
being priced as though it crossed a network.

The same shape decides whether an operator's workers prefer to be packed onto few
nodes or spread across many. Packing is preferred when the gang fits a handful of
nodes and doing so moves a material fraction of the exchange onto a faster tier.

Two things are deliberately left alone. Every tier whose bandwidth cannot be read is
charged the network rate, so an unlabelled fleet, a container that cannot see the
host's `/sys`, and a single-node run are all ranked exactly as they were before any of
this existed. And skew is charged only to a partitioned window, which can neither
pre-aggregate its hot partition away the way an aggregate does nor spread it across
reducers the way a salted join does.

Cardinality estimation drives those costs. Before anything is learned, Kyber falls
back to Selinger-style selectivities, where `col = literal` passes 10% of rows, a
range predicate passes a third, and `IS NULL` passes 5%. Sketches built during execution (HyperLogLog for
distinct counts, KLL for quantiles) and learned per-query statistics in the
MetadataHub supersede those defaults and sharpen across runs.

## Adaptive re-optimization

This is the part static optimizers cannot do. An estimate is only a guess until the
query runs. At a pipeline breaker, the engine has *measured* the real size of what it
just processed. Core records those measurements, and when an estimate was off by more
than `optimizer.reoptimize_error` (default 2×), Kyber re-plans the rest of the query
on the measured numbers. The same mechanism works single-node and distributed.

DuckDB optimizes once, before execution, and cannot revise the plan afterwards.
Spark AQE re-plans at stage boundaries, and so does Kyber: it is the same mechanism at
the same granularity, not a finer one. It also engages only on a joined query whose scan
input clears 20M rows or roughly 1.3 GB, so small queries never reach it.

What has no equivalent in either engine is the *cross-query* half. Core records each
run's measured cardinalities, operator times, and peak memory into the MetadataHub, and
the next run of that shape plans against them: sketch-backed cardinality, cost
coefficients calibrated from measured operator times, and a UCB1 bandit over join
strategies. A query therefore gets a better plan the more often it runs.

Two of those loops do more than tune a number, because what they remember lets a later run
skip work outright rather than choose between two ways of doing it.

A **top-N remembers its k-th best value** (`kyber/learned_tuning/topn_bound.py`). Before the
scan nothing knows a value that separates the top ten from the rest, so the query has to read
everything and then select; afterwards the engine does know one, and the next run of the shape
starts from it as a predicate. Row-group zone maps and late materialization then answer the
query without decoding what it excludes. The rewrite is safe under any staleness: the filter
removes only rows strictly worse than the bound, so if `k` rows survive they are the true
top-k regardless of where the bound came from. Too few survivors is the one failure mode, it
is visible in the row count, and the conductor re-runs the plan as written. That verification
is why the seeding is a Kyber decision but the fallback is `api`'s: a plan → plan pass cannot
express "run it, look at the answer, run something else".

A **distributed sort remembers its range boundaries** (`dist/sort_boundaries.py`). The sample
barrier executes the whole mapped prefix over every split to produce a few dozen floats per
worker, and the map barrier then executes that same prefix again to do the partitioning. The
remembered grids remove the first of those two passes. Boundaries decide only which reducer a
row lands on, and the ordered concatenation is correct for any monotone boundary list, so a
stale grid costs balance and never a row.

## Using it

You rarely call Kyber directly, because it runs automatically on every terminal
operation:

```python
import batcher as bt

ds = bt.read("data.parquet").filter(bt.col("value") > 100).select("id", "name", "value")
result = ds.collect()  # Kyber optimizes here, then the engine runs the plan
```

To see the optimized plan without running it, use `explain`:

```python
print(ds.explain())
```

The public optimizer surface (`batcher.kyber`) is small. `optimize` and
`optimize_traced` run the pipeline, and the second one also returns the per-rule
decision log. The learning entry points `record_execution`, `record_column_stats`,
`record_selectivity`, and `load_learned_stats` feed the MetadataHub that later plans
read from.

## See also

- {doc}`/architecture/internals/carbonite`: checks the feasibility of the plan Kyber produces.
- {doc}`/architecture/internals/execution`: runs the plan and measures what Kyber re-plans on.
- {doc}`/configuration/options`: the cost-model and cardinality knobs.
- {doc}`/architecture/deep-dives/adaptive/cardinality-estimation`: how the estimates themselves are built.
- {doc}`/architecture/deep-dives/adaptive/adaptive-reoptimization`: the re-planning loop in full detail.
