# Cardinality estimation

Every cost-based decision the optimizer makes rests on one number: how many rows will this
subtree produce? Join order, build-side choice, broadcast eligibility, memory admission,
worker fan-out. All of them are downstream of a row count nobody has counted yet.

The number is usually wrong. The discipline is in knowing *how* wrong, and in never letting
an inexact number answer a question that demands an exact one.

## Provenance

Every estimate carries a tag saying where it came from. This is the single most important
type in the estimator.

```python
# docs: skip
# python/batcher/plan/stats.py
class Provenance(IntEnum):
    EXACT = 0      # provably correct without execution (a footer, a manifest)
    HISTOGRAM = 1  # KLL / t-digest / DDSketch quantile sketch measured from data
    SKETCH = 2     # HLL distinct / Count-Min frequency (approximate by construction)
    LEARNED = 3    # a prior from a past run, keyed by plan signature
    DEFAULT = 4    # Selinger heuristic / an unconstrained guess
```

Ordered strongest-first, so trust composes with `max`. There is exactly one combiner:

```python
# docs: skip
def weakest(*provenances) -> Provenance:  # == max(provenances)
```

:::{important}
No call site may hand-set `EXACT` on a derived facet. That one rule is the firewall: a statistic
can only ever be *weakened* as it propagates up a plan. It is what lets the metadata-answer path
(`count()` from a Parquet footer, `min()` from a zone map) short-circuit a query without ever
risking a wrong answer. An inexact statistic may inform cost. It may never answer an exact
terminal.
:::

```text
   strongest ──────────────────────────────────────────────────────► weakest
   EXACT        HISTOGRAM        SKETCH        LEARNED        DEFAULT
   a footer,    a KLL / t-digest HLL / Count-  a prior from   a Selinger
   a manifest   quantile sketch  Min, approx   a past run     heuristic


        aggregate     est≈200    (learned)   ◄── the weakest input wins
            │
        filter        est≈667    (default)   ◄── nobody has measured this predicate
            │
        scan          est≈2,000  (exact)     ◄── an in-memory source: the count is known
```

You can see it in `explain()`:

```python
import batcher as bt

ds = bt.from_pydict({"g": [i % 8 for i in range(2000)], "x": [float(i) for i in range(2000)]})
print(ds.filter(bt.col("x") > 100).group_by("g").agg(n=bt.count()).explain())
```

```text
aggregate                       est≈200 (learned)
  filter                        est≈667 (default)
    scan                        est≈2,000 (exact)
```

The scan is `exact`: an in-memory source with a known row count. The filter is
`default`, because nobody has measured this predicate. The aggregate is `learned`; the group
cardinality came from a prior run.

## Cold start: Selinger

With nothing measured, the estimator falls back to constants that have been the industry's
answer since System R. They live in `CardinalityConfig`:

```python
# docs: skip
eq_selectivity: float = 0.1           # col = literal
range_selectivity: float = 1.0 / 3.0  # col < | <= | > | >= literal
null_selectivity: float = 0.05        # col IS NULL
substring_selectivity: float = 0.05   # LIKE '%x%' / contains / regex
prefix_selectivity: float = 0.10      # LIKE 'x%' / starts_with / ends_with
default_filter_selectivity: float = 0.5
```

The string-pattern ones earn their place. Without a string histogram a `LIKE '%green%'` is
genuinely unknowable, but it is near-universally *selective*. Nobody writes a substring
search that matches half the table. Falling back to 0.5 made Kyber believe TPC-H Q9's
`p_name LIKE '%green%'` kept 100k of 200k parts (it keeps 10.7k), which hid the most
selective join in the query and steered the order into gigabyte intermediates.

`unknown_rows = 1e12` is not an estimate. It is a sentinel meaning "unbudgeted", and the
downstream consumers know it: `annotate.py` refuses to budget memory for a plan whose rows
are at or above it, and the aggregate/distinct estimators deliberately do not *shrink* it,
because shrinking a placeholder would make an unbudgeted guess look like a real, admissible
estimate.

## Composing predicates

Two conjuncts are almost never independent. `country = 'US' AND state = 'CA'` multiplied
naively gives 0.01. The real figure is nearer 0.1, because the second predicate implies the first.

`kyber/stats/selectivity/combine.py` uses exponential backoff over the ascending-sorted
selectivities rather than a product:

```text
s₁ · s₂^(1/2) · s₃^(1/4) · …
```

The most selective conjunct counts fully, and each subsequent one is damped by a further
square root. The result lands between the pure independence product, which is a lower bound
exact only when the conjuncts really are independent, and the most selective conjunct alone,
which is the upper bound of the perfectly-correlated case. That makes it dramatically less
wrong than independence on correlated columns, which is most real schemas. Two range
conjuncts on the same column are recognized first and combined as a single interval, since
two bounds on one column carve one range rather than two independent predicates. `OR` uses
honest inclusion-exclusion. `NOT` subtracts the null mass first, because SQL keeps only TRUE.

## Joins

`_inner_join_rows` in `kyber/stats/estimator.py` is Selinger containment:

```text
|L| · |R| / max(d_L, d_R)     capped at the cartesian bound |L| · |R|
```

where `d` is the key's distinct count. With a composite key whose combined NDV saturates
its row count (ratio ≥ 0.95, i.e. it is effectively a primary key), it short-circuits to
`max(|L|, |R|)`, because a PK-FK join produces one row per FK row.

With no NDV at all it also returns `max(|L|, |R|)`, which assumes many-to-one.

:::{warning}
That assumption is the known cold-start failure. A genuinely many-to-many low-NDV join gets
estimated **64 to 80 times low**, and the join order that follows drives into intermediates of
12M to 18M rows. TPC-H q5 cold takes 7,115 ms. Warm, with the NDV learned, it takes 300 ms.
:::

Seeding source-side HLL NDV on base join keys closes that gap before the optimizer runs, but
only for *resident* sources that are already in memory, where sketching costs no extra I/O
(`api/terminal/_metadata.py::seed_column_ndv`). A file-backed source is skipped, because
re-reading it purely to sketch would double the query's I/O, so it still plans blind on its
first run and learns its NDV from the post-run pass.

A semi join keeps the matched fraction `min(1, d_R/d_L)` of the left rows, and an anti join
takes the complement, so the two partition `|L|` exactly. Outer joins take the appropriate
floor, so `left` becomes `max(inner, |L|)`.

Multi-column key sets combine through one shared function, `combine_ndv`, and it uses the
same exponential-backoff shape.

:::{dropdown} `combine_ndv`, and the bounds it respects
```python
# docs: skip
ordered = sorted((d for d in per_column if d > 0), reverse=True)
combined, exponent = 1.0, 1.0
for d in ordered:
    combined *= d ** exponent
    exponent /= 2.0
return max(1.0, min(combined, cap))
```

Bounded below by `max_i d_i` and above by `∏ d_i` (the Fréchet bounds), capped at the
relation's row count. One definition serves join keys, group-by keys, and `DISTINCT` column
sets, so they cannot disagree.
:::

## Sketches

Once a query has run, sketches from `bc-sketches` supersede the constants. They are all
`Mergeable` with a fixed seed, so a sketch built on partition 3 of worker 7 merges
identically with one built anywhere else:

```rust
// crates/bc-sketches/src/lib.rs
pub(crate) const SEED: ahash::RandomState =
    ahash::RandomState::with_seeds(0xC0FF_EE01, 0xDEAD_BEEF, 0x1234_5678, 0xABCD_EF01);
```

| Sketch | Answers | Default | Error |
|---|---|---|---|
| `HyperLogLog` | distinct count (NDV) | precision 14 → 16 KB | ~1.04/√m ≈ 0.8% |
| `KllSketch` | quantiles / range selectivity | k = 200 | ~1% rank error |
| `CountMinSketch` | frequency of a known key | `width = ⌈e/ε⌉`, `depth = ⌈ln(1/δ)⌉` | ≤ εN, never under |
| `FrequentItems` | *find* the hot keys (Misra-Gries) | capacity | ≥ N/(cap+1) guaranteed found |
| `BloomFilter` | membership (data skipping) | `fp_rate` | one-sided |

Count-Min and Misra-Gries are used together on purpose. Count-Min never under-counts;
Misra-Gries never over-counts and is guaranteed to *contain* every key above `N/(capacity+1)`.
One sizes a hot key you already know about; the other finds the ones you do not.

One detail in the HLL worth knowing, because it is a deliberate deviation from the paper.
The handover from linear counting to the HLL estimator sits at load factor **3.5**, not
Flajolet's 2.5. At 2.5 the discontinuity produced a +2.4% systematic overestimate at 26 to 42
standard errors. Bias, not noise. Sweeping the threshold at p=14: 2.5 gives 0.915% RMSE and
2.56% worst bias; **3.5 gives 0.746% and 0.38%**. HLL++'s alternative is roughly 3,000
empirical bias-correction constants; moving the handover was cheaper and better.

### The rule that keeps sketches honest

A distinct count is the one statistic that has to be tracked apart from the rest of its bundle.
A Parquet footer gives exact min, max, and null counts but never a trustworthy distinct count,
so the only NDV such a column can carry is an HLL estimate. When a `ColumnStat` carried a single
provenance for the whole bundle, attaching that estimate to an otherwise `EXACT` column tagged
the NDV `EXACT` too, which would let an approximate count answer a `count_distinct`.

:::{important}
`ColumnStat.ndv_provenance` carries the distinct count's *own* tag, separately from the bundle's
(`python/batcher/plan/stats.py`). A sketched NDV rides alongside exact bounds, and
`ndv_is_exact` is the gate every exact-answer path reads, so the sketch informs cost while it
still refuses to answer a terminal. `kyber/stats/columns.py::scan_columns` merges a measured NDV
onto a column that has none and tags it `SKETCH` whatever the bounds are worth.
:::

The purely descriptive stats, meaning quantiles, most-common-values, and average bytes, attach
to any column without disturbing provenance.

## The correction loop

Structural estimation only gets you so far. The last layer is empirical: Core reports, per
operator, the rows it actually produced against the rows Kyber estimated *before* correction.
The geometric mean of that q-error, per operator signature, multiplies the next estimate.

A join Kyber has consistently under-estimated 8× is next planned for at 8×.

Guardrails: `cardinality_correction_min_samples` (2) before a factor is trusted;
`cardinality_correction_max_factor` (32) clamps it both ways; `cardinality_correction_window`
(8) averages only the recent past, because the structural estimator itself sharpens as NDVs
accumulate and an all-history mean would keep applying a correction it has outgrown.

`_CORRECTABLE` is `(Aggregate, Distinct, Join, MapBatches, Unnest)`. `Filter` is excluded on
purpose, because its selectivity is already learned per-signature and correcting it again
would count the same error twice. `Scan` is excluded because `plan_signature` structures every
scan as the bare token `["scan"]`, so all scans in a process would collide on one entry.

`MapBatches` belongs for the same reason as `Unnest`. A UDF may filter, explode, or pass rows
through one for one, and which one it does is a property of the code rather than of the plan,
so the structural estimator can only assume 1:1. That is safe to correct only because a
`map_batches` signature carries the UDF's identity by qualified name
(`kyber.signature._udf_identity`), so one UDF's learned fan-out cannot answer for another's.
An anonymous lambda still collides, and that is the floor.

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

## Limits

The estimator has no multi-column histograms and no correlation model. Exponential backoff
is a stand-in for both, and on a query where a filter's columns are strongly correlated it
will still be off by an order of magnitude on the first run.

A UDF's fan-out and an `Unnest`'s fan-out are both structurally unknowable, so the estimator
assumes rows pass through unchanged and leaves the learned loop to correct them. The first run
of an exploding `flat_map` is therefore planned as if it explodes not at all.

And the thing every estimator shares: it is a prediction. What makes it survivable is that
the engine measures the truth at every pipeline breaker and re-plans on it. See
{doc}`Adaptive re-optimization <adaptive-reoptimization>`.

## Code map

Each estimate described above has one owning file. Start here when you want to see how
a number is actually derived:

| Concern | File |
|---|---|
| The estimator | `python/batcher/kyber/stats/estimator.py` |
| Predicate selectivity | `python/batcher/kyber/stats/selectivity/` |
| Merging learned column stats into a scan | `python/batcher/kyber/stats/columns.py` |
| `Provenance`, `RelStats`, `ColumnStat` | `python/batcher/plan/stats.py` |
| The sketches | `crates/bc-sketches/src/` |
| Cold-start constants | `python/batcher/config/config.py::CardinalityConfig` |

## See also

:::{seealso}
- {doc}`Architecture <../architecture/index>`: Kyber's lane, where it decides and never executes or measures
- {doc}`Kyber optimizer <../internals/kyber>`: the passes these estimates feed
- `docs/internals/mathematical_foundations.md` (in the repo, not a site page): the sketch error bounds, derived
- {doc}`Reading a plan <../user-guide/explain-plans>`: the `est≈` and provenance tags in the tree
- {doc}`Optimizing a slow query <../tutorials/optimizing-a-slow-query>`: what to do when an estimate is badly wrong
- {doc}`TPC-H benchmarks <../benchmarks/tpch>`: q5 and q9, the two queries this page keeps naming
- {doc}`Cost model <cost-model>`: what consumes these row counts
- {doc}`Adaptive re-optimization <adaptive-reoptimization>`: measuring the truth at a breaker
- {doc}`Learned metadata <learned-metadata>`: where the NDVs and corrections are stored
:::
