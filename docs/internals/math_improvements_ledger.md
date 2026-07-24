# Mathematical improvements ledger

A running record of improvements to the *mathematics* of Batcher's internal algorithms:
cardinality and selectivity estimation, sketch estimators, the cost model, the learning
loop, resource control, and the distributed scheduling heuristics.

Each entry names the quantity being estimated, what the previous formula got wrong, and the
closed form that replaced it. Entries are grouped by subsystem and numbered continuously.
This page is not published (`exclude_patterns` in `docs/conf.py`); it is a working index for
anyone touching one of these estimators.

## Cardinality and column statistics (`kyber/stats/`)

A new module, `kyber/stats/distribution.py`, holds the distributional primitives the row
estimator, the column-stat propagator, and the selectivity model all need, so the three
cannot drift into different approximations of the same quantity.

| # | Improvement |
|---|---|
| 1 | `residual_mass` — the probability mass a column's most-common-value table does not cover. |
| 2 | `residual_eq_frequency` — a non-most-common value occurs `(1 - Σf)/(d - k)` times, not `1/d`. Pricing it at `1/d` double-counts the skew the MCV table already records, which is the largest systematic over-estimate in equality selectivity on Zipfian data. |
| 3 | Degenerate case: when the MCV table enumerates the whole column, an unlisted value is bounded by the rarest *listed* frequency rather than the uniform average. |
| 4 | `distinct_after_selection` — Yao/Cardenas: a selection keeping `k` of `n` rows leaves `d·(1 - (1-k/n)^(n/d))` distinct values, not `min(d, k)`. |
| 5 | Evaluated as `-expm1(x·log1p(-s))` so a highly selective predicate does not lose the answer to cancellation. |
| 6 | Unique-column shortcut: when values do not repeat, the surviving distinct count is exactly the surviving row count. |
| 7 | `union_ndv` — a union's distinct count from the Fréchet bounds `max d_i ≤ d ≤ Σ d_i`, interpolated with an independent-membership model instead of being dropped. |
| 8 | `overlap_fraction` — the fraction of one value range lying inside another. |
| 9 | `join_match_fraction` — one definition of the containment fraction `min(1, d_R/d_L)`, so semi/anti/outer estimation cannot disagree. |
| 10 | `mcv_join_rows` — the four-term skew/residual join decomposition (both-listed, left-only, right-only, neither), replacing "uniform, floored by the shared-hot-value term". |
| 11 | Cross terms included: a hot left value joins against its counterpart's *residual* frequency, which is what an anti-correlated skew looks like. |
| 12 | `merge_quantile_grids` — a union's CDF is the exact row-weighted mixture of its branches' CDFs; re-gridded by bisection instead of dropping quantiles at a union. |
| 13 | Monotonicity of the merged grid restored by a running max, since bisection on a flat region can emit a non-monotone pair. |
| 14 | `geometric_mean` — log-space averaging for ratios, where an arithmetic mean is asymmetric (4x over and 4x under average to 2.125, not 1). |
| 15 | `_point_mass` (range-boundary mass) reads the residual frequency. |
| 16 | Wildcard-free `LIKE` (an equality in disguise) reads the residual frequency. |
| 17 | `IN` list: unlisted literals read the residual frequency. |
| 18 | `col = literal`: unlisted literals read the residual frequency. |
| 19 | `col_a = col_b`: MCV-aware, using the same decomposition a join uses, instead of the plain `1/max(d_a, d_b)`. |
| 20 | `filter_columns` shrinks each column's distinct count by Yao rather than capping it at the row count. |
| 21 | `limit_columns` deliberately does *not*: a prefix is not a uniform random subset, so only `min(d, rows)` is sound there. |
| 22 | Union column stats: `total_sum` is additive over branches (exact). |
| 23 | Union column stats: `mean` is the row-weighted mean, not the unweighted one. |
| 24 | Union column stats: `ndv` propagates via `union_ndv` instead of being dropped. |
| 25 | Union column stats: `avg_bytes` is row-weighted, not the widest branch's. |
| 26 | Union column stats: quantile grids merge as a weighted CDF mixture. |
| 27 | `_row_weighted` requires *every* branch to have the statistic, so a partial average cannot masquerade as the union's. |
| 28 | Equi-join key bounds are **intersected** across the two sides — only values in both survive. |
| 29 | Equi-join key `ndv` is bounded by `min(d_L, d_R)` (containment), much sharper than `min(d_side, out_rows)` on a fan-out join. |
| 30 | An empty intersection never emits inverted bounds. |
| 31 | Bound comparison gated on ordinal, non-ambiguous (no NaN/signed-zero) values. |
| 32 | `LEFT JOIN` = inner + unmatched-left, not `max(inner, |L|)`, which is only a lower bound. |
| 33 | `RIGHT JOIN` likewise. |
| 34 | `FULL JOIN` = inner + both unmatched terms. |
| 35 | The unmatched term defaults to zero when the distinct counts are unmeasured, so an unmeasured outer join keeps its previous estimate rather than inventing null-extended rows. |
| 36 | Semi and anti derive from **one** estimate (`anti = |L| - semi`), so they partition `|L|` by construction. |
| 37 | With no usable distinct counts both fall back to `|L|` without claiming the complement is empty. |
| 38 | The inner join reads the MCV decomposition when both key frequency tables exist. |
| 39 | `_range_scaled` — a join estimate scales by how much the two key `[min, max]` ranges actually overlap, not just the all-or-nothing disjointness test. |
| 40 | A window's partition count uses the damped `combine_ndv`, not the independence product of the partition keys. |
| 41 | `combine_ndv` returns the exact answer when one column is already a key of the relation (it functionally determines the rest). |
| 42 | `UNION` (distinct) estimates the deduplicated count instead of reporting the concatenated total. |

## Sketches (`crates/bc-sketches`)

| # | Improvement |
|---|---|
| 43 | HyperLogLog: Ertl's improved (maximum-likelihood) estimator replaces the raw estimator plus linear-counting handover — one continuous function, essentially unbiased across the whole range, no threshold and no per-precision bias tables. |
| 44 | The `sigma` zero-register correction, in closed form. |
| 45 | The `tau` saturated-register correction, covering the large-range end. |
| 46 | The exact limiting constant `α_∞ = 1/(2 ln 2)` replaces the per-`m` raw-estimator corrections. |
| 47 | The estimate is computed from an integer register-multiplicity histogram, which is also a cheaper pass than the float harmonic sum. |
| 48 | An empty sketch estimates exactly zero. |
| 49 | A saturated sketch reports a finite bound rather than an infinity. |
| 50 | Bloom filter: the integer `k` is chosen first and `m` solved for it (`m = -kn/ln(1 - p^(1/k))`), so the filter actually meets its target false-positive rate — rounding `k` after fixing `m` missed it. |
| 51 | Both integers bracketing the real optimum are evaluated; the one needing fewer bits wins. |
| 52 | Probe positions use the full 64-bit hash, so a filter above 512 MB can address all of its bits. |
| 53 | Count-Min: a debiased count-mean-min estimator subtracts each cell's expected collision noise `(N - c)/(w - 1)`. |
| 54 | The median across rows makes it robust to a row that collided with something enormous. |
| 55 | Clamped into `[0, min]`, so the guaranteed no-under-count upper bound still holds. |
| 56 | `is_heavy` (the shuffle-salting decision) reads the debiased estimate, since the min's bias is systematic and one-directional. |

## Aggregate numerics (`crates/bc-runtime/src/agg`)

| # | Improvement |
|---|---|
| 57 | `NeumaierSum` — Kahan-Babuska-Neumaier compensated summation, `O(eps.Sum|x|)` error independent of `n` instead of `O(n.eps.Sum|x|)`. The Neumaier form (not plain Kahan) so an addend larger than the accumulator is also handled. |
| 58 | `covar_state`'s pass-1 means are compensated — the two-pass co-moment conditions its whole result on that mean. |
| 59 | `moment_state`'s pass-1 mean likewise. |
| 60 | `corr` computes `sqrt(M2x)·sqrt(M2y)`, never `sqrt(M2x·M2y)`, which overflows to infinity past ~1e154 and reported a perfect correlation as 0. |
| 61 | `corr` is clamped to `[-1, 1]`; Cauchy-Schwarz bounds it, and rounding across three moments must not report 1.0000000000000002. |
| 62 | A non-finite denominator yields NULL rather than a nonsense value. |
| 63 | Skewness uses `m2·sqrt(m2)` rather than `m2.powf(1.5)` — one correctly-rounded root instead of two transcendentals. |
| 64 | `merge_covar` copies the first partial for a group exactly instead of folding it through the correction formula. |
| 65 | `merge_welford` likewise, so a group whose rows all landed in one partition finalizes to exactly that partition's value. |
| 66 | `merge_welford` combines balanced partials as the weighted average `(n_a·m_a + n_b·m_b)/n` and unbalanced ones incrementally — Chan's own distinction. Morsel-parallel aggregation merges balanced partials by construction, which is the case the incremental form serves worst. |

## The learned cardinality correction (`kyber/correction.py`)

| # | Improvement |
|---|---|
| 67 | The correction is a **recency-weighted** geometric mean, so it tracks a shift in the data rather than averaging it against a stale regime. |
| 68 | Kish's effective sample size, so the decayed tail of the window is not counted as full evidence. |
| 69 | **Shrinkage** toward no correction: the posterior mean under a normal prior, `mu.tau²/(tau² + s²/n_eff)`. A consistent 8x bias passes through; a mean assembled from scattered runs is mostly noise and is pulled back to 1.0. |
| 70 | Samples are clipped **before** averaging, so one absurd run cannot drag the mean to the clamp and hold it there for a whole window. |
| 71 | The half-life is chosen so symmetric noise still nearly cancels — a faster decay invented an 8% correction out of balanced samples. |
| 72 | Bessel correction through the effective sample size. |

## Cost model (`kyber/cost.py`)

| # | Improvement |
|---|---|
| 73 | A **cache-residency factor** on random hash-table access: flat while the table fits the last-level cache, then growing per octave of overflow and flattening once every access already misses. Without it a 1,000-row and a 100-million-row build side cost the same to probe. |
| 74 | Applied to the aggregate's group table. |
| 75 | Applied to `DISTINCT`. |
| 76 | Applied to a `UNION`'s dedup table. |
| 77 | **Spill IO** charged for a join build side past the memory budget (the grace-hash fallback). |
| 78 | Spill IO for an over-budget aggregate. |
| 79 | Spill IO for an over-budget distinct. |
| 80 | An out-of-core sort is charged `ceil(log_F(state/budget))` merge passes, so its IO grows logarithmically in the overflow rather than being free. |
| 81 | Top-N is costed as `n` root comparisons plus the **expected** `k(1 + ln(n/k))` heap insertions, not `n·log2(k)` — a 3.3x over-charge for `LIMIT 10` over a large scan, which made the optimizer nearly indifferent to fusing the limit. |
| 82 | A limit at or above the input degenerates to exactly the full-sort cost. |
| 83 | A window sorts *within* partitions: `n·log2(n/p)`, not `n·log2(n)`. |
| 84 | Its partition count comes from the same damped `combine_ndv` the cardinality estimator uses, so cost and cardinality cannot disagree. |
| 85 | `UNION` (distinct) pays a dedup build; it was priced as a streaming concatenation. |
| 86 | Subtree costs are memoized by node identity, so repeated cost-based rules do not make planning quadratic in plan size. |
| 87 | The planner reads the same spill budget the engine is given, so "this will spill" means one thing. |

## The join-strategy bandit (`kyber/learned_tuning/bandit.py`)

| # | Improvement |
|---|---|
| 88 | **Discounted UCB** (Garivier-Moulines): evidence decays geometrically, so an arm measured badly long ago can be re-examined after the hardware or data changes. Undiscounted UCB is not converged there, it is stuck. |
| 89 | The discount applies to *every* arm per observation, so a rarely-chosen arm's ancient evidence does not keep full weight. |
| 90 | Arm statistics are a Welford `(n, mean, m2)` state, not `(n, sum, sumsq)` — the latter recovers the variance by subtracting two nearly-equal large numbers, the same cancellation the variance aggregate was rewritten to avoid. |
| 91 | The pooled reward spread combines the per-arm states with Chan's parallel formula. |
| 92 | **UCB-V**: the confidence radius uses each arm's *own* spread, so an arm that wins every single time is not explored as hard as an erratic one. |
| 93 | The radius floor is a fraction of the pooled spread rather than of the arm's mean, which is what leaves room for that distinction. |

## Shuffle flow control (`carbonite/policies/flow_control.py`)

| # | Improvement |
|---|---|
| 94 | Congestion avoidance follows a **CUBIC** curve back toward the window congestion was last observed at, instead of `+alpha` per round. After a backoff from 64 credits, additive increase needs 32 network round trips to return to a window already measured as safe. |
| 95 | Growth is the maximum of the cubic and the additive law (CUBIC's TCP-friendly region), so recovery can never be *slower* than before. |
| 96 | The window at backoff is remembered as the channel's measured capacity. |
| 97 | The byte-bounded memory-safe ceiling still clamps every growth path. |

## Distributed skew (`dist/skew.py`)

| # | Improvement |
|---|---|
| 98 | The salt fan-out is sized from the overload it must repair, `s >= f·P`, rather than a constant 4. A 10% key across 200 reducers overloads by 20x and a fan-out of 4 leaves it 5x over average — and the wide shuffle is the case that only appears at scale. |
| 99 | Capped, since each sub-partition replicates the matching build-side rows. |
| 100 | Floored at 2 — salting at all means splitting in two. |

## Quantile and sampling sketches (`crates/bc-sketches`)

| # | Improvement |
|---|---|
| 101 | Reservoir sampling moves from Vitter's Algorithm R to Li's **Algorithm L**: the same uniform distribution (`k/n` per item, exactly) in `O(k(1 + log(n/k)))` random draws instead of one per item. Sampling 1,000 rows from 10 million goes from 10 million draws to ~30 thousand, on the scan path. |
| 102 | The skip length is drawn from the geometric distribution the per-item dice would have produced, so the two algorithms sample the same distribution rather than merely similar ones. |
| 103 | A `unit()` generator over the **open** interval `(0, 1)`, built from 53 significant bits — every caller feeds it to a logarithm, and 0 or 1 there is an infinity or a degenerate jump. |
| 104 | The skip saturates rather than overflowing once `w` underflows on an enormous stream. |
| 105 | KLL `rank` interpolates a **continuous** CDF instead of summing weights at or below `x`. After compaction each retained item stands for `2^h` rows, so the step function moved a whole step for a literal a hair from a boundary — and that rank is a selectivity estimate. |
| 106 | Each item's anchor is its **rank centre** (mid-weight), the same convention t-digest's quantile interpolation uses, so the two are mutual inverses rather than off by half a bucket. |
| 107 | Below the first and above the last retained item the CDF runs to the exactly-tracked min and max. |
| 108 | A zero-width span (repeated values) takes the upper anchor instead of dividing by zero. |
| 109 | KLL quantile lookup is a binary search over prefix sums; the histogram grid is `buckets + 1` lookups off one sketch, so the previous per-quantile scan made building it quadratic in the retained item count. |
| 110 | The prefix sums are built once per batch of lookups. |
| 111 | t-digest centroid absorption is an **incremental** mean correction, not `(m_a·w_a + m_b·w_b)/w`. A centroid near the median absorbs thousands of values, and the product form rebuilds the mean from large numbers each time — visible as drifting quantiles on timestamps or prices. |
| 112 | t-digest `rank` interpolates the tail between the last centroid's mean and the exactly-tracked maximum instead of saturating at 1.0, so it agrees with `quantile` there and stops reporting `col <= x` as keeping everything for an `x` inside the data. |

## Learned-model fitting (`kyber/ols.py`, `kyber/calibration.py`, `metadata/smoothed.py`)

| # | Improvement |
|---|---|
| 113 | The crossover regressions accumulate **centered** moments (West's weighted update), not the power sums `(Sx, Sy, Sxx, Sxy)`. Recovering a slope from power sums forms `n·Sxx - Sx²`, whose operands agree to fourteen digits on clustered input — at a 1e12 offset with a 1e6 spread it retains no significant digits at all. |
| 114 | The spread guard is now about **extrapolation** (a slope fitted over a 1e-6 relative spread should not be projected to a distant crossover) rather than doubling as a workaround for float noise. |
| 115 | A fit must explain the data: `R² = C_xy²/(M2x·M2y)` below 0.5 is refused. Spread in `x` makes a slope identifiable; it does not make it real, and that intercept is the whole numerator of a threshold. |
| 116 | `M2y` is accumulated, which is what makes R² available at no extra cost. |
| 117 | A legacy power-sum state is discarded rather than misread, and the bucket restarts. |
| 118 | Cost-coefficient shrinkage is **geometric**, `prior·(measured/prior)^w`. A coefficient is a positive scale, and the arithmetic blend is not symmetric under inversion (10x blends to 5.5x but 0.1x to 0.55x), so a family whose measurements straddle the prior drifts upward with every fit. It also matches `_clamp`, which was already multiplicative. |
| 119 | A non-positive coefficient (no logarithm) falls back to the arithmetic blend rather than being dropped. |
| 120 | The neutral learned-scalar smoother takes a `max(floor, 1/(n+1))` step: a running mean while evidence is thin, settling into an exponential average. A static weight left the *first* observation an eighth of the estimate after four runs, forever. |
| 121 | The observation count is stored alongside the value, and a legacy bare float is still read and folded into. |
| 122 | The floor is `learned_scalar_alpha_floor`, not `learning_smoothing_alpha` — at the latter's 0.5 the `1/(n+1)` term would never bind and the running-mean phase would not exist. This is the neutral-layer twin of the same distinction `kyber.learning._smooth` makes. |

## Adaptive re-optimization (`api/adaptive/gating.py`)

| # | Improvement |
|---|---|
| 123 | A predicted-empty intermediate that came back empty is a *perfect* estimate, not a miss. The symmetric q-error cannot express `0/0`, and calling it inaccurate forced an extra re-optimization pass — and another pipeline break — on precisely the query whose estimates were right. |

## Join-key range overlap (`kyber/stats/estimator.py`)

| # | Improvement |
|---|---|
| 124 | A key's overlap with the other side is measured as **mass** (`F(hi) - F(lo)` from its quantile grid) rather than as a share of its `[min, max]` width. A fact table spanning three years whose rows concentrate in the most recent one overlaps a one-year dimension in a third of its width but most of its mass, and the width-based factor would cut the join estimate to a third of the truth. |
| 125 | Falls back to the width share when no grid has been measured, so the cold path is unchanged. |

## Derived-column statistics (`kyber/stats/derived.py`)

| # | Improvement |
|---|---|
| 126 | A monotonic projection carries its **quantile grid** through, exactly: `F_y(g(x)) = F_x(x)` for increasing `g`, so the boundary values map and the probabilities do not. `SELECT x*100 AS cents ... WHERE cents BETWEEN ...` fell from histogram interpolation to a flat constant on a column whose distribution was fully known. |
| 127 | A decreasing map reverses the values and complements the probabilities, so the grid stays ascending in both — the shape the interpolator requires. |
| 128 | The **most-common values** carry to their images. The map is injective, so no two values collide and every measured frequency transfers, which keeps a `WHERE derived = k` on the measured skew instead of a uniform `1/ndv`. |
| 129 | `mean(a·x + b) = a·mean(x) + b`, exactly. |
| 130 | The measured byte width carries through — an affine map of a number does not change how wide it is. |
| 131 | `least`/`greatest` bounds are **tight**, not a union box: `least` is at most the smallest of the arguments' maxima and `greatest` at least the largest of their minima. `least(price, 100)` over `[0, 10^6]` is bounded above by 100, so a downstream `> 500` is provably empty instead of estimated at a third of the table. |

## Which join keys count as skewed (`kyber/stats/skew.py`)

| # | Improvement |
|---|---|
| 132 | Hotness is measured against the shuffle's **width**: a reducer's fair share is `1/P`, so a value at 5% is harmless across 4 reducers and a 10x straggler across 200. A fixed fraction cannot be right for both. |
| 133 | A value's share of the join **output** is `f_L·f_R/S`, where `S = Σ f_L(u)f_R(u)` is the total match probability — a number far below 1, so the raw product is amplified by roughly the key's distinct count. Two frequencies that look unremarkable on their own inputs can hand one reducer most of the join, and no test on either side alone can see it. |
| 134 | `S` is the same skew-plus-residual total the cardinality estimator uses for the join, so the two cannot disagree about how large it is. |
| 135 | Without a frequency table on both sides there is no product to evaluate and only the input test applies. |

## Grouped-aggregate output bounds (`kyber/stats/aggregate_columns.py`)

| # | Improvement |
|---|---|
| 136 | A grouped `min`/`max`/`avg`/`median` output inherits its input column's `[min, max]` — every group's value lies inside the column's own range. A `HAVING` clause is a predicate on exactly these columns, and with no statistics at all it fell to a flat constant; `HAVING max(v) > 10^6` over a column whose maximum is 1,000 is now provably empty. |
| 137 | A grouped counting aggregate is bounded by `[1, |input|]` (`[0, |input|]` for `count(col)`, which is 0 over an all-null group). |
| 138 | Always `DEFAULT` provenance: these bound a *set* of per-group values, so nothing here may answer an exact terminal. |

## A soundness correction to the union merge

| # | Improvement |
|---|---|
| 139 | The union's `ndv` carries its **own** `DEFAULT` provenance. It is an estimate however exact the branches are — the branches' value-set overlap is unmeasured — and without its own tag it inherited an EXACT bundle provenance and would have let `count_distinct` answer a `UNION ALL` from a model. |

## Two more quantile/frequency sketch corrections

| # | Improvement |
|---|---|
| 140 | DDSketch `rank` interpolates the **straddling bucket** log-uniformly (`log_γ(x/lower)`), matching the sketch's own geometric spacing, instead of counting 0 or all of it. The CDF is continuous, so the selectivity of `col <= x` and `col <= x + ε` differ by ε rather than jumping a whole bucket — consistent with the KLL and t-digest rank fixes. |
| 141 | Handled across zero: the negative mirror map is ordered the opposite way, so the interpolation direction is mirrored to keep `rank` monotone through 0. |
| 142 | Misra-Gries `heavy_hitters` tests each key's **upper** bound (`counter + total/(capacity+1)`), not the raw counter. The counters underestimate, so filtering on the raw value drops exactly the borderline-heavy keys whose counters were decremented most — a false negative that leaves a shuffle straggler. Reporting on the upper bound cannot miss a hot key, the safe direction for salting (matching Count-Min `is_heavy` and the join bloom). |
