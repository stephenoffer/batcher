# Contract-loop & optimizer-math audit ledger

A systematic audit of the Kyber → Carbonite → Core contract loop and of Kyber's internal
mathematics. Four independent passes produced ~118 concrete findings. This ledger records
what was **fixed** (with the test that pins it) and what remains **open**, so the
remainder is recoverable work rather than lost analysis.

Scope reminder: Batcher spans sub-second to PB, single-node to distributed, batch and
streaming. A fix that helps one regime and harms another is not a fix. Each entry notes
the regime it affects.

---

## Fixed

### The contract loop

| # | Defect | Where | Regime | Test |
|---|--------|-------|--------|------|
| L1 | `_estimate_accurate` used a relative error normalized by the estimate. That is bounded by 1 for *any* over-estimate, so with `reoptimize_error = 2.0` every over-estimate compared "accurate" and adaptive re-optimization stopped — disabling the loop for exactly the case it exists to catch (a selective filter feeding a join). Now a symmetric q-error. | `api/adaptive.py` | large, distributed | `test_contract_loop.py` |
| L2 | Carbonite never read `Provenance`, though `plan/physical.py` promises it does. A `Provenance.DEFAULT` guess could raise `PlanError` on a plan that would have run. Infeasibility from a guess is now *advisory*: it still routes out-of-core, it cannot fail the query. | `carbonite/policies.py`, `plan/resource.py`, `api/orchestration.py` | all | `test_contract_loop.py` |
| L3 | The per-task Ray `memory=` reservation divided one **machine's** budget by the **cluster-wide** task count and took the `min`. A 100-task job reserved 1/100th of a node per task, so Ray over-packed and OOMed. The plan peak is already divided by the task count; the budget must not be divided again. | `carbonite/policies.py` | distributed | `test_contract_loop.py` |
| L4 | `aimd_beta` was unvalidated. `beta >= 1` turns multiplicative *decrease* into growth — the window grows on congestion, an unstable control law. Now clamped to `(0, 1)`. | `carbonite/policies.py` | streaming, distributed | `test_contract_loop.py` |
| L5 | `ShuffleRecovery.run` recomputed lost partitions on its **final** round, then exited into `ResourceError` and discarded the work — the most expensive step in recovery, paid for nothing. | `carbonite/resilience/recovery.py` | distributed | `test_contract_loop.py` |
| L6 | The reserve-fail spill fallback spilled the **raw** plan, not the optimized one — running a comma join as a cartesian product out-of-core. The sibling spill site warns about exactly this. | `api/orchestration.py` | large, spill | — (covered by differential) |
| L7 | `record_broadcast_timing` had **no callers**, so `learned_broadcast_max_bytes` was permanently `None` and the broadcast threshold never left its static 10 MiB default. Loop closed (distributed path only — a single-node hash join is not a "shuffle" and would poison the fit), carrying the build **bytes** Kyber computes. | `api/tuning/decisions.py`, `kyber/rules/selection.py` | distributed | `test_contract_loop.py` |
| L8 | The build side after a swap is the **original left**, but two sites used `min(left_rows, right_rows)`. The cost-based swap compares byte-aware `op_cost`, so it can build a row-heavier but narrower side — mislabeling the sort-merge gate and the crossover's x-axis. | `kyber/rules/selection.py`, `api/tuning/decisions.py` | large joins | — |
| L9 | `eliminate_sort_before_aggregate` dropped a `Sort` below **any** `GROUP BY`. `list_agg`/`array_agg` collect in arrival order and `arg_min`/`arg_max`/`mode` break ties by it: a live **wrong-result** bug (`sort(x).group_by(g).agg(array_agg(x))` returned unsorted lists). Now gated on an explicit order-insensitivity whitelist, so a new aggregate must opt in. | `kyber/rules/algebraic.py` | all | `test_diff_order_sensitive_aggregates.py` |

### Kyber's mathematics

| # | Defect | Where | Test |
|---|--------|-------|------|
| M1 | `semi` and `anti` joins both returned `left.rows`. They **partition** `|L|`, so both can equal it only if one is empty; a near-empty anti-join was costed at full width. Now `|L| · min(1, d_R/d_L)` and its complement, falling back to the `|L|` upper bound when a distinct count is missing. | `kyber/stats/estimator.py` | `test_join_cardinality.py` |
| M2 | Outer joins had no preserved-side floor. A `LEFT JOIN` must emit `>= |L|` rows; the inner containment estimate put it *below* that — a count no execution can produce. | same | same |
| M3 | A cartesian (comma) join estimated `max(|L|, |R|)` instead of `|L| · |R|` — short by a factor of `min(|L|, |R|)` on the operator whose size most needs believing. | same | same |
| M4 | The inner-join estimate could exceed the cartesian bound `|L| · |R|`; now capped. | same | same |
| M5 | Group-by and `DISTINCT` multiplied their key NDVs (independence), while joins **damped** theirs for the same quantity. Correlated keys saturated the row cap, so the optimizer concluded grouping reduced nothing. One shared `combine_ndv` now serves join keys, group keys, and distinct column sets. | same | `test_join_cardinality.py` |
| M6 | `col < v` was estimated as `col <= v` (and `>=` as `>`): the boundary's point mass landed on neither side, so `P(x<v) + P(x>=v) != 1`. Now `lt = F - eq`, `ge = 1 - F + eq`. | `kyber/stats/selectivity.py` | `test_selectivity_math.py` |
| M7 | A quantile grid's endpoints clamped to 0/1 *at* the boundary, discarding `probs[0] + (1 - probs[-1])` of the distribution for a grid inset from the true extremes. Now inclusive. | same | same |
| M8 | `NOT p` and `col != v` assumed 2-valued logic. SQL keeps only TRUE rows, so a NULL operand is dropped by `p` **and** by `NOT p`: `sel(p) + sel(NOT p) = 1 - f_null(p)`. The null mass is now subtracted, using the column's *measured* null fraction where a footer supplied one. `IS NULL`/`IS NOT NULL` stay two-valued (their negation is exact). | same, plumbed from `estimator.py` | same |
| M9 | `IS NULL` ignored measured null counts and always returned the `null_selectivity` prior. | same | same |
| M10 | `IN` lists did not deduplicate literals (`x IN (1,1,2)` counted three values) and ignored measured MCV frequencies. Note: the union of same-column equalities is their **sum** — they are mutually exclusive — so `k/ndv` was already the right form. | same | same |
| M11 | MCV lookups keyed on `str(value)`, so a `5.0` literal missed an int column's `"5"` entry — exactly the skewed values the table exists to sharpen. Now type-tolerant (and `bool` still cannot collide with `1`). | same | same |
| M12 | Top-N sort cost used `log2(limit)`, so a `LIMIT` larger than the input was costed **above** a full sort. Now `log2(min(limit, n))`. | `kyber/cost.py` | — |
| M13 | Calibration blended a fixed `alpha = 0.5` of the shipped **default** into every fit, so its fixed point was `0.5·measured + 0.5·default`: a coefficient whose true value is 10× the default converged to 5.5× and stayed there however much data arrived. Replaced with evidence-weighted shrinkage (`n / (n + prior_strength)`), which is stable cold and converges to the measurement. | `kyber/calibration.py` | `test_cost_calibration.py` |

---

## Open — ranked

These are audited, located, and reproducible; they are not yet fixed.

### Tier 1 — measurement corruption (needs Rust changes)

1. **`peak_bytes` is the operator's OUTPUT array size, not its peak working set**
   (`crates/bc-interp/src/lib.rs:100,367`). Carbonite's `LearnedMemoryModel` fits
   `bytes_per_row = median(peak / rows_in)` on it, so admission, spill routing, buffer
   reservation and per-worker sizing are all fit on output size. A 60M-row → 4-group
   aggregate records ~0 peak. The profile layer already relabels it honestly
   (`result_bytes`), so the two consumers of one wire field disagree. **Regime: large,
   distributed, spill — systematic under-provisioning of exactly the cardinality-reducing
   breakers that spill.** Fix: instrument true per-op peak (materialized input + hash
   table) and emit it as a separate field; until then, do not fit the memory model on it.

2. **Join `rows_in` is the SUM of both sides** (`bc-interp/src/lib.rs:281`), so
   `selectivity = rows_out / (left + right)` is neither probe selectivity nor fan-out, and
   `hash_join` is calibrated against a figure that conflates the asymmetric build and probe
   costs. Compounds (1). Fix: record build and probe rows separately.

3. **`threads` is hardcoded to 1** (`bc-interp/src/lib.rs:99,366`), so `cpu_utilization`
   divides by 1 and cannot distinguish CPU-bound from IO-bound families — the CPU-share
   learner it feeds is inert.

### Tier 2 — loops that never close

4. Only the **disk aggregate** distributed path is metered. Distributed joins, sorts,
   windows, distinct, write, and every Flight path use unmetered `execute_plan`, so they
   contribute nothing to calibration or the learned memory model.
5. The distributed branch records strictly less than single-node (no `record_execution`,
   no `learn_column_stats`, no `record_selectivity`, no `record_run_feedback`), so the
   "single-node == distributed" invariant holds for *results* but not for *learned state*.
6. The spill path and the streaming path emit no operator feedback at all. The largest
   queries — the ones that spill — contribute least to the memory model that decides
   spilling.
7. The join-strategy bandit's reward is **whole-query wall time**, recorded only for
   single-join plans. UCB1 assumes a stationary per-arm reward; mixing input sizes makes
   an arm sampled early on large inputs look permanently worse. Multi-join plans never
   tune.
8. Distributed adaptive never records a flip (the accuracy check is gated on
   `isinstance(result, pa.Table)`, but distributed stages return a `MaterializedSource`),
   so `learned_adaptive_helps` can never turn on for distributed shapes.

### Tier 3 — optimizer soundness claims that are not enforced

9. `_apply_node_rules` fuses node-local rules into one `transform_up` and claims
   observational equivalence. That is false when a rule rewrites a node into a **new
   subtree**: the new children are never visited by the later rules in that pass. Harmless
   in a fixpoint phase (costs an iteration); in the once-run SELECTION/ENFORCE phases a
   legitimate rewrite is silently dropped.
10. `Rule.idempotent` is threaded through the registry and **never read**. Fixpoint phases
    assume confluence by convention; a non-confluent rule can oscillate.
11. `_run_phase` truncates at `fixpoint_iterations` with no non-convergence signal, so the
    plan a query gets depends on the iteration cap, silently.
12. `plan_signature` erases literal values, so `WHERE x < 0` and `WHERE x < 1e9` share a
    signature and therefore a learned row count / selectivity. Contained today only because
    every result-affecting rule gates on `EXACT`; it is a live landmine the moment a rewrite
    consumes `LEARNED`.
13. **Bloom `eq` pruning is an unguarded cross-language trust boundary**
    (`zonemap_pruning.py:179`). `contains → False` *drops the row group*. Soundness rests
    on Python's `canonical_bytes` being byte-identical to how Rust inserted the values; a
    divergence (temporal normalization, dictionary encoding, oversized ints) is a
    **wrong-result** bug, not a slow plan. Needs an explicit type-domain guard and a
    differential test.
14. Greedy and DP join enumerators optimize different cost recurrences (`cost.cost` full
    subtree vs additive `op_cost`), so the two paths can pick different plans at the
    threshold.
15. `op_cost(Join)` hardcodes the build side as the right input, but SELECTION flips it
    *after* JOIN_REORDER — the reorder is costed against an orientation the physical plan
    will not use.

### Tier 4 — architecture boundary

16. `dist/executors/join.py` re-materializes the build side at execution time and
    **reverses** the planner's broadcast decision against `broadcast_max_bytes`, and
    `_bloom_beneficial` instantiates Kyber's estimator inside the executor. Planning logic
    in the execution plane, which the layering contract reserves for Kyber/Carbonite.

### Tier 5 — smaller, still real

17. `CacheStore` and `BufferPool` account against **disjoint** budgets, so
    "storage yields to execution" (`evict_to_free` before `reserve`) cannot change whether
    a reservation fits — and the two limits *stack* past the physical envelope.
18. `should_spill` returns `False` for an un-estimable plan ("never spill on a guess") while
    the scheduler treats the same plan as the one most needing a spill budget. The two
    halves of Carbonite disagree.
19. `TieredSpillStore._local_used` is updated only at bucket **close**, so concurrent
    streaming buckets all read a stale value and all choose the local tier; the counter is
    also unsynchronized.
20. `PressureMonitor.level()` advances its de-escalation EWMA on **every call**, so
    hysteresis depends on call frequency rather than time.
21. `n_max_parallelism == 0` / `c_max_credits == 0` are overloaded: the admission
    counter-offer emits 0 meaning "unconstrained", the scheduler reads 0 as "unsized → use
    defaults".
22. `merge_metric_ops` sums `elapsed_ns` across workers that ran in **parallel** and
    presents the sum as the operator's wall time.
23. Sketches: HLL omits the HLL++ bias-correction tables in the mid-cardinality zone;
    Count-Min applies neither conservative-update nor the mean/noise-floor correction, so
    heavy-hitter frequencies (which drive skew salting) are biased high.
24. `_smooth`'s `alpha` floor of 0.5 gives every learned per-signature scalar a ~2-sample
    memory: one anomalous run swings it by half.
25. Missing high-impact rewrites vs DuckDB/Spark: no general CSE, no TopN-through-join, no
    distinct/window pushdown through join, no true semi-join reduction (sideways
    information passing is min/max-`BETWEEN` only and runs in ENFORCE, *after* join
    ordering, so the reduced cardinality cannot inform it).
26. Zonemap-produced emptiness (`Limit(x, 0)`, emitted in SELECTION) cannot propagate: the
    empty-join folds live in PUSHDOWN, which already ran, and `propagate_empty_relation`
    does not fold through `Join`/`Project`/`Aggregate`.

---

## Notes on findings that were *rejected* on inspection

Recording these so they are not "re-discovered":

* **`IN`-list selectivity is not a Bonferroni over-estimate.** `col = v_i` and `col = v_j`
  are mutually exclusive, so the union of `k` distinct equalities is their **sum**, `k/d`,
  exactly under uniformity. `1 - (1 - 1/d)^k` would model `k` independent Bernoulli draws,
  which this is not. Only the missing dedup and the unused MCV were real.
* **`eliminate_sort_before_sample` is sound.** `Sample` keeps a row by a stable hash of its
  *values*, so the sampled multiset does not depend on input order.
* **`Cost.total()` excluding `mem` is a design decision, not a bug.** `mem` is a peak that
  gates feasibility, not throughput. Folding it into the scalar would change every join
  order; the right fix is to charge the *IO* of a spill, not to weight a peak.
* **`outer_to_inner_join`'s null-rejection analysis is correct** under 3-valued logic, as
  are `join_to_semijoin`, `push_semijoin_through_join`, `infer_join_predicates`,
  `constant_propagation`, `factor_common_conjuncts`, `combine_limits`, and the
  `UNION ALL`-only gating of the limit pushdowns.
