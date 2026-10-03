# Cardinality estimation

This page describes how Kyber estimates the rows a plan will produce, how it tracks how far to trust each estimate, and how measured runs correct it.

Join order, build side, broadcast eligibility, memory admission, and worker fan-out all rest on one number: how many rows will this subtree produce? The number is usually a guess. The discipline is in knowing how good the guess is, and in never letting an inexact number answer a question that demands an exact one.

## Provenance

Every estimate carries a tag saying where it came from, ordered strongest-first so trust composes with `max`:

```python
# docs: skip
# python/batcher/plan/stats.py
class Provenance(IntEnum):
    EXACT = 0  # provably correct without execution (a footer, a manifest)
    HISTOGRAM = 1  # KLL / t-digest / DDSketch quantile sketch measured from data
    SKETCH = 2  # HLL / theta-sketch distinct count (approximate by construction)
    LEARNED = 3  # a prior from a past run, keyed by plan signature
    DEFAULT = 4  # Selinger heuristic / an unconstrained guess
```

:::{important}
No call site may hand-set `EXACT` on a derived facet, so a statistic can only be *weakened* as it propagates up a plan. That is what lets `count()` from a Parquet footer or `min()` from a zone map short-circuit a query safely. An inexact statistic may inform cost. It may never answer an exact terminal.
:::

You can read the tags in `explain()`:

```python
import batcher as bt

ds = bt.from_pydict({"g": [i % 8 for i in range(2000)], "x": [float(i) for i in range(2000)]})
print(ds.filter(bt.col("x") > 100).group_by("g").agg(n=bt.count()).explain())
```

```text
query plan (planned)                                    3 operators
───────────────────────────────────────────────────────────────────
OPERATOR                         ESTIMATE  NOTES
aggregate  [by g · count_star]      est≈8  (default)
└─ filter  [x > 100]              est≈667  (default)
   └─ scan  [source 0]          est≈2,000  (exact)  pushed[x > 100]
```

The scan is `exact` because an in-memory source knows its row count. The filter above it is derived, so it can't inherit `exact`, and the aggregate takes the weakest tag below it.

## Cold start: Selinger constants

With nothing measured, the estimator falls back to the System R constants in {py:class}`CardinalityConfig <batcher.config.config.CardinalityConfig>`:

```python
import batcher as bt

c = bt.CardinalityConfig()
print(c.eq_selectivity, c.range_selectivity)          # 0.1 0.3333333333333333
print(c.substring_selectivity, c.prefix_selectivity)  # 0.05 0.1
```

A `LIKE '%x%'` is unknowable without a string histogram, so 0.05 is a prior about workloads: analytic substring filters are usually selective. After one run, the measured selectivity replaces it:

```python
import batcher as bt

names = bt.from_pydict({"name": [f"n{i}" for i in range(2000)]})
q = names.filter(bt.col("name").str.contains("7"))
print(q.collect().num_rows)  # 542
print(q.explain())           # the filter now reads est≈542 (learned), not est≈100 (default)
```

`unknown_rows = 1e12` is a sentinel meaning "unbudgeted", not an estimate. Memory admission refuses to budget a plan at or above it, and the aggregate estimators never shrink it into something that looks admissible.

## Composing predicates

Two conjuncts are rarely independent: `country = 'US' AND state = 'CA'` multiplied naively gives 0.01 where the truth is nearer 0.1. `kyber/stats/selectivity/combine.py` uses exponential backoff over the ascending-sorted selectivities instead:

```text
s₁ · s₂^(1/2) · s₃^(1/4) · …
```

The most selective conjunct counts fully and each further one is damped by another square root, so the result sits between the independence product and the most selective conjunct alone. Two range conjuncts on one column are combined as a single interval first. `OR` uses inclusion-exclusion, and `NOT` subtracts the null mass first, because SQL keeps only TRUE.

## Joins

`_inner_join_rows` in `kyber/stats/estimator.py` is Selinger containment:

```text
|L| · |R| / max(d_L, d_R)     capped at the cartesian bound |L| · |R|
```

where `d` is the key's distinct count. A composite key whose combined NDV saturates its row count (ratio ≥ 0.95) short-circuits to `max(|L|, |R|)`, the PK-FK answer. With no NDV at all it also returns `max(|L|, |R|)`. A semi join keeps `min(1, d_R/d_L)` of the left rows and an anti join the complement, and outer joins take the appropriate floor.

Resident in-memory sources get source-side HLL NDV on their join keys before the optimizer runs, at no extra I/O (`api/terminal/_metadata.py::seed_column_ndv`). A file-backed source learns its NDV from the post-run pass instead. On TPC-H q5, the learned NDV takes the query from 7,115 ms cold to 300 ms warm.

:::{dropdown} `combine_ndv`, shared by joins, group-by and `DISTINCT`
```python
# docs: skip
ordered = sorted((d for d in per_column if d > 0), reverse=True)
combined, exponent = 1.0, 1.0
for d in ordered:
    combined *= d**exponent
    exponent /= 2.0
return max(1.0, min(combined, cap))
```

Bounded below by `max_i d_i` and above by `∏ d_i`, capped at the relation's row count. One definition serves join keys, group-by keys, and `DISTINCT` column sets, so they cannot disagree.
:::

## Aggregates

A grouped aggregate's row count is the distinct combinations of its keys, through the same `combine_ndv`. Its *column* statistics are derived in `kyber/stats/aggregate_columns.py`, and the rule that governs them is which outputs grouping leaves alone.

:::{dropdown} What carries through a group-by, and what doesn't
- A bare-column group key keeps the child's `min` and `max` at the child's provenance. Its distinct count does not stay exact, because the group count is an estimate.
- Grouping collapses every null key into one group, so the null count is pinned in two cases only: zero nulls in means zero out, and with a single key the nulls become exactly one group. With several keys nothing is claimed.
- `min`, `max`, `avg` and `median` outputs stay inside their column's range. A per-group count lies between one and the child's row count, and that upper bound is published only when the child's count is exact, because a pruning rule may fold a `HAVING count(*) > n` on it.
- A global aggregate emits one row, so `count(*)`, `min`, `max`, `sum` and `count_distinct` become constants whenever the child's exact statistics determine them.

The pinned null counts matter beyond estimation: `constant_value` and `_predicate_status` need a known-zero null count before they will call a key constant or a predicate provably true.
:::

## Sketches

Once a query has run, sketches from `bc-sketches` supersede the constants. All of them are `Mergeable` and hash with the same fixed, portable seed, so a sketch built on one worker merges with one built anywhere else:

```rust
// crates/bc-sketches/src/lib.rs
pub(crate) const SEED: bc_arrow::PortableBuildHasher =
    bc_arrow::PortableBuildHasher::with_seed(0x534B_4554_4348_4553);
```

| Sketch | Answers | Default | Error |
|---|---|---|---|
| `HyperLogLog` | distinct count (NDV) | precision 14 → 16 KB | ~1.04/√m ≈ 0.8% |
| `KllSketch` | quantiles / range selectivity | k = 200 | ~1% rank error |
| `FrequentItems` | *find* the hot keys (Misra-Gries) | capacity | ≥ N/(cap+1) guaranteed found |
| `BloomFilter` | membership (data skipping) | `fp_rate` | one-sided |

HyperLogLog folds by register-wise max and Bloom by bitwise OR, so any merge order reaches a bit-identical state. KLL compacts and TDigest re-clusters, so two merge orders agree only to within the sketch's own rank error, which [`crates/bc-sketches/tests/merge_order.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-sketches/tests/merge_order.rs) pins. `FrequentItems` isn't covered by that test either way. The HLL uses Ertl's improved maximum-likelihood estimator, which needs neither a linear-counting handover nor HyperLogLog++'s bias tables.

![The sketches behind an estimate, split by how they merge. Two of them reach the same state in any merge order: HyperLogLog, for distinct counts, folds by register-wise max; and Bloom, for membership and data skipping, folds by bitwise OR. ColumnStats' min, max, count and ndv fold the same way, and exact counts from that side feed the cardinality estimate of row counts and per-column stats. The quantile sketches only agree within their own rank error: KLL's merge compacts and TDigest re-clusters its centroids, so two merge orders give two answers. The worst gap measured in rank was 0.0097 for KLL at k equals 200 and 0.0050 for TDigest at compression 100, which is the error those sketches already promise rather than a defect. They feed quantiles and range selectivity. Never assert that a quantile sketch merges to an identical state, and never set out to fix the fact that it does not.](/_static/diagrams/cardinality_sketches.svg)

:::{important}
A distinct count carries its own provenance. `ColumnStat.ndv_provenance` tags a sketched NDV separately from the column's exact bounds ([`python/batcher/plan/stats.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/plan/stats.py)), and `ndv_is_exact` is the gate every exact-answer path reads. A sketch informs cost but can never answer a {py:meth}`count_distinct <batcher.plan.expr_ir.core.Expr.count_distinct>`.
:::

## The correction loop

The last layer is empirical. Core reports, per operator, the rows it produced against the rows Kyber estimated *before* correction, and the geometric mean of that q-error, per operator signature, multiplies the next estimate. A join Kyber under-estimated 8x is next planned for at 8x.

The knobs live on `OptimizerConfig`:

```python
import batcher as bt

opt = bt.active_config().optimizer
print(opt.cardinality_correction_min_samples)  # 2
print(opt.cardinality_correction_max_factor)   # 32.0
print(opt.cardinality_correction_window)       # 8
```

A factor needs two samples before it's trusted, is clamped to 32x either way, and averages only the last eight runs, because the structural estimator sharpens as NDVs accumulate. `_CORRECTABLE` is `(Aggregate, Distinct, Join, MapBatches, Unnest)`. `Filter` is left out because its selectivity is already learned per signature. `MapBatches` qualifies because its signature carries the UDF's qualified name, so one UDF's fan-out can't answer for another's.

## Cold and warm, side by side

::::{tab-set}
:::{tab-item} Cold: nothing measured
```text
scan     row count from the source (often EXACT)
filter   Selinger constants: eq 0.1, range 1/3, null 0.05,
         substring 0.05, prefix 0.10, otherwise 0.5
join     Selinger containment, and max(|L|, |R|) when there is no NDV at all
aggregate / distinct   combine_ndv over the key columns

provenance: DEFAULT above the leaves. TPC-H q5 takes 7,115 ms here.
```
:::

:::{tab-item} Warm: the loop has run
```text
scan     source-side HLL NDV, KLL quantiles, most-common-values, avg bytes
filter   a learned per-signature selectivity
join     Selinger containment against a MEASURED NDV
aggregate / distinct   the same, plus a q-error correction factor

provenance: SKETCH / LEARNED. The same q5 takes 300 ms.
```
:::
::::

## Practical limits

- There are no multi-column histograms and no correlation model. Exponential backoff stands in for both until a run measures the truth.
- A UDF's or an `Unnest`'s fan-out is structurally unknowable, so the first run assumes rows pass through one for one and the correction loop learns the rest.
- On a query large enough to stage, the engine also re-plans at a pipeline breaker once it has counted. See {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`.

:::{dropdown} Code map
| Concern | File |
|---|---|
| The estimator | [`python/batcher/kyber/stats/estimator.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/kyber/stats/estimator.py) |
| Predicate selectivity | [`python/batcher/kyber/stats/selectivity/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/kyber/stats/selectivity) |
| Merging learned column stats into a scan | [`python/batcher/kyber/stats/columns.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/kyber/stats/columns.py) |
| Aggregate output column stats | [`python/batcher/kyber/stats/aggregate_columns.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/kyber/stats/aggregate_columns.py) |
| `Provenance`, `RelStats`, `ColumnStat` | [`python/batcher/plan/stats.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/plan/stats.py) |
| The sketches | [`crates/bc-sketches/src/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-sketches/src) |
| Cold-start constants | `python/batcher/config/config.py::CardinalityConfig` |
:::

## See also

- {doc}`Architecture </architecture/index>`: Kyber's lane, where it decides and never executes or measures.
- {doc}`Kyber optimizer </architecture/internals/kyber>`: the passes these estimates feed.
- {doc}`Reading a plan </user-guide/operate/tuning/explain-plans>`: the `est≈` and provenance tags in the tree.
- {doc}`Optimizing a slow query </getting-started/tutorials/foundations/optimizing-a-slow-query>`: what to do when an estimate is badly off.
- {doc}`TPC-H benchmarks </benchmarks/results/tpch>`: the join shapes this page names.
- {doc}`Cost model </architecture/deep-dives/adaptive/cost-model>`: what consumes these row counts.
- {doc}`Adaptive re-optimization </architecture/deep-dives/adaptive/adaptive-reoptimization>`: measuring the truth at a breaker.
- {doc}`Learned metadata </architecture/deep-dives/adaptive/learned-metadata>`: where the NDVs and corrections are stored.
