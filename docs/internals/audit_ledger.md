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
| L13 | Every learned per-signature scalar (join sides, partition rows, group reduction, output rows, filter selectivity) smoothed with `alpha = max(0.5, 1/(n+1))`, so from the second observation onward the newest run always carried half the weight: a ~2-sample memory that never converged, and one anomalous run swung the prior by 50% (100 -> 550 in a test). The knob was *overloaded* — `learning_smoothing_alpha` is also a static blend weight for `blend_peak`, the pressure EWMA and autobatch — so the EWMA now has its own `learned_scalar_alpha_floor` (0.1), giving a running mean while evidence is thin and a ~10-observation memory after. | `config`, `kyber/learning.py`, `kyber/learned_tuning.py` | all | — |
| L14 | `PressureMonitor.level()` folded a reading into its de-escalation EWMA on **every call**, and three components called it per round at unrelated cadences — so the average advanced several steps, collapsed toward the raw reading, and defeated the anti-flap smoothing that stops SPILL<->NORMAL oscillation from thrashing the AIMD credit window. `level()` remains the single sampler; readers use the new pure `classify()`. | `carbonite/memory/pressure.py`, `carbonite/manager.py`, `carbonite/transfer/session.py` | streaming, distributed | `test_contract_loop.py` |
| L15 | `merge_metric_ops` **summed** `elapsed_ns` across workers that ran concurrently and presented it as the operator's wall time — reporting a W-worker stage as W times slower. Wall time now takes the max (slowest worker); rows/CPU/threads sum, which also makes the merged `cpu_util` a coherent cluster-wide per-core figure. | `plan/profile/collect.py` | distributed | — |
| L23 | `HyperLogLog::estimate` handed over from linear counting to the raw estimator at Flajolet's `2.5m`. The raw HLL estimator is biased high for `m < n < 5m`, so `2.5m` steps straight onto the worst of it. **Measured** (20-40 independent trials per point, so bias separates from the `1.04/sqrt(m)` sampling variance): a **+2.4% systematic overestimate at 26-42 standard errors**, at both p=14 and p=16, decaying to ~0 by `n = 4m`. HLL++ removes this with ~15x200 empirical per-precision bias tables; rather than carry (or invent) those constants, the handover moves to where the two estimators' *total* error balances. Linear counting's relative standard error is closed-form (`sqrt(e^t - t - 1)/(t*sqrt(m))`), so sweeping the threshold and scoring RMSE over `n/m` in [0.5, 8] gives a minimum near 3.2-3.5 at **every** precision tested (12/14/16) — as theory predicts, since both errors depend on `t = n/m` alone. `3.5` is within 1% of the RMSE minimum everywhere and cuts worst-case bias ~6x (2.56% -> 0.38%). NDV feeds join cardinality, so the bias fed straight into Kyber's join estimates. | `crates/bc-sketches/src/hll.rs` | all (NDV/join estimation) | `hll.rs::no_bias_spike_at_the_linear_counting_handover` (fails at 2.5 with "2.247% exceeds 1%") |
| L22 | `ResourceManager.default_bounds` — a permissive `m_max_bytes = 1 << 62` envelope — had **no caller anywhere** (not in a Protocol, not in a doc, not in a test). Had anything ever annotated an operator with it, `peak_operator_bytes` would take the max and `should_spill` would compare `1<<62 > budget` and route *every* such query out-of-core. Deleted; `ruff` then flagged the `ResourceBounds` import as unused, confirming it was the only user. | `carbonite/manager.py` | — | `test_carbonite_*` (unchanged, still green) |
| L21 | `CacheStore.on_pressure` — the pressure ladder the module docstring advertises as "the storage-vs-execution split Spark's `UnifiedMemoryManager` makes" — had **no production caller**, only a unit test. Cache bytes are not accounted against the buffer pool, so `result_cache_max_bytes` (256 MiB default) sat entirely *outside* the envelope every other Carbonite decision sizes against: the two budgets were disjoint and could stack past physical RAM, and the cache never yielded RAM until a reservation happened to hit an exact deficit. `ResourceManager.reserve` now applies the ladder (3/4 at ELEVATED, 1/2 at SPILL, all at CRITICAL) before the deficit check, reading `classify()` so it does not consume the AIMD round's sample. `NORMAL` leaves the cache untouched, so the common path costs nothing. Evicting a cache only forces a recompute — result-invariant. | `carbonite/manager.py` | all, memory-pressured | `test_carbonite_cache.py` |
| L20 | The join-strategy UCB1 bandit was greedy in all but name, on two counts. (a) Its reward was **raw wall time**, but UCB1 assumes a stationary per-arm reward: the same signature runs over 1M rows today and 50M tomorrow, so an arm sampled once on a large input carried a permanently inflated mean and was never chosen again. Simulated (hash truly 2x faster, first run on a 50x input): cumulative regret **312 ms, hash pulled once in 80 rounds**. `record_join_strategy` now divides by the join's input rows -> regret **4 ms**, hash pulled 80/81. (b) The confidence radius `c*sqrt(2 ln N / n)` was **dimensionless while the mean was in ms** — against a 500 ms mean it is a 0.19% nudge. Scaling it by the pooled reward spread (recovered from the `sumsq` `record_arm` was already storing and **nothing ever read**) makes `c` dimensionless: verified that regret and arm pulls are now *identical* whether the reward is recorded in ms/Mrow, ms/row or us/Mrow. Namespace bumped to `join_arm_v2` so ms-scale history cannot mix with the new per-row scale. Bounded honestly: the scaled radius does **not** resurrect a genuinely worse arm (pooled spread decays as `1/sqrt(N)`) — that is what the normalization is for, and a test pins the boundary. | `kyber/learned_tuning.py`, `api/tuning/decisions.py` | joins, all scales | `test_join_strategy_bandit.py` |
| L19 | `execute_spilling_aggregate` answered an empty input to a **global** aggregate with `combine_finalize(..., [])`, which has no partial state to type the result from and raises `aggregation over empty input is not yet supported`. So `group_by().agg(median(v))` over an empty relation *crashed* on the spill path while the single-node engine and DuckDB both return one NULL row. A schema-carrying empty batch is now routed through the same map -> partial -> finalize pipeline, so the aggregate's identity element falls out of the mergeable algebra instead of being special-cased. (Pre-existing at HEAD; confirmed by running the test in a HEAD worktree.) | `dist/spill.py` | distributed + spill | `test_adaptive_stress.py::test_tiny_budget_spill_matches_baseline[global_median-empty]` |
| L18 | The adaptive loop gauged estimate accuracy with `isinstance(result, pa.Table)`, but a distributed stage parks a `MaterializedSource`/`FlightMaterializedSource` — never a table. So `flipped` stayed `False` on every distributed run and the learned adaptive verdict could **never** turn on for a distributed shape: the adaptive gate could not learn from precisely the queries that most need it. Both source types already carry an exact `row_count` from their reduce tasks; `_stage_row_count` now reads it, returning `None` only for a genuinely unbounded (streaming) source. The single-node-only early-exit is left as documented. | `api/adaptive.py` | distributed + adaptive | `test_distributed_feedback.py` |
| L17 | Only the **disk-aggregate map task** was metered. Distributed sorts, windows, joins, distinct and the broadcast probe all called the unmetered `execute_plan`, so a distributed run contributed **nothing** to the cost calibration or the learned memory model — while being the path that runs the largest inputs and the one that spills. `Core measures` held single-node only. A worker-side `execute_metered` + driver-side `record_worker_metrics` seam now closes it for the sort map+reduce, the window map+reduce, the shuffle-join reduce (the hash-join breaker, carrying `rows_build`/`peak_bytes`), the broadcast probe (one document per chunk), and distinct (which rides the aggregate shuffle). Degrades to the unmetered call on an engine without the metered entry point — a worker never fails a query to collect a statistic. `metrics_out` (the profile's map-sub-plan display channel) is deliberately **not** fed from the new sites: `merge_metric_ops` merges by `op_id`, and reduce ops share ids with map ops of a different sub-plan. | `dist/executors/ray_runtime/metering.py`, `dist/executor.py`, `dist/executors/{sort,window,join,distinct,aggregate}.py` | distributed | `test_distributed_feedback.py` |
| L16 | `op_cost(Join)` prices the build side as the right input, but the build-side rule runs in SELECTION — *after* JOIN_REORDER. With `hash_build_row` twice `hash_probe_row`, the reorder penalized every order that put the large table on the right, even though SELECTION would have swapped it. Ordering now uses `join_op_cost`, which prices the cheaper (commutative) orientation; `op_cost` stays orientation-specific because the build-side rule compares the two against each other. Non-inner joins are priced as written. | `kyber/cost.py`, `kyber/rules/join_order.py` | large joins | `test_contract_loop.py` |
| L11 | `_apply_node_rules` fused a phase's node rules into one bottom-up walk and claimed observational equivalence. It is not: `transform_up` has already visited a node's children when a rule fires on it, so a rule that rewrites a node into a **new subtree** leaves those children unvisited by the later rules of that run. A fixpoint phase recovers them next iteration; the once-run SELECTION/ENFORCE phases lost the rewrite outright. Fusing is now confined to the phases that iterate. | `kyber/optimizer.py` | all | `test_contract_loop.py` |
| L12 | `_run_phase` truncated at `fixpoint_iterations` with **no signal**, so a non-confluent rule made the plan a query gets depend silently on the iteration cap. It now warns. (Results were always correct — every rule is semantics-preserving — but plan quality was non-reproducible.) | `kyber/optimizer.py` | all | — |
| L10 | The bloom `eq` data-skip treated `contains() -> False` as a proof that no row matches, replacing the whole relation with an empty one — so a single false negative **deletes rows**, and the proof rested on an *unverified* cross-language agreement between the Rust builder and a pure-Python reader. The agreement is now verified by property tests over 1000+ values (including `i64::MIN`/`MAX`, unicode, embedded NULs). A cross-domain probe (an int literal against a string index) *does* report a definitive false absence; today a `Cast` implicitly prevents it from reaching the rule, and an explicit type-domain guard now makes that safety a property of the rule rather than an accident of the rewriter. | `kyber/rules/zonemap_pruning.py` | all | `test_bloom_no_false_negatives.py` |
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

### The engine's measurement contract (Rust)

| # | Defect | Where | Test |
|---|--------|-------|------|
| E1 | `peak_bytes` recorded the operator's **output** array size for every operator. Carbonite fits its per-family memory model on it and drives admission, spill routing, buffer reservation and per-worker sizing from that fit — so a 2M-row aggregate over 4 groups reported a 64-byte peak and the model learned ~0 bytes/row. `peak_bytes` is now the true peak working set (a breaker's materialized input **plus** its result; a streaming operator's result alone), and the old value is preserved as a separate `result_bytes` the profiler reports. Measured: the aggregate's learned footprint went from ~0 to **16.05 bytes/row**, its true row width. | `bc-interp/src/{metrics,lib,par}.rs`, `carbonite/memory/learned.py`, `plan/{feedback,profile}` | `crates/bc-interp/tests/metrics_contract.rs` |
| E2 | `batch_bytes` used `get_array_memory_size()`, which reports the **whole parent buffer** for a sliced array. Morselizing one 32 MB table into 122 morsels measured 3.9 GB — a 123x over-count that would have made the corrected `peak_bytes` useless. Now sums each column's `get_slice_memory_size()`. | `bc-interp/src/lib.rs` | same |
| E3 | A join's `rows_in` was the **sum** of both sides, so `rows_out / rows_in` was neither the probe rate nor the fan-out, and one calibrated coefficient conflated the asymmetric build and probe costs. `rows_in` is now the probe side and `rows_build` reports the hashed side. | `bc-interp/src/{metrics,lib,par}.rs`, `plan/feedback.py`, `core/executor.py` | same |

---

### Enterprise security floor (2026-07-26)

Four fixes, each verified by running the failure rather than reasoning about it. Full
detail in `databricks_parity.md`; the short form:

1. **UDF workers inherited the engine's credentials.** A `forkserver` child got the whole
   parent environment, including `env:` secret material and `BATCHER_SECRET_COMMAND` —
   which names a program that hands out arbitrary secrets on request. Closed by
   `core/udf/isolation.py` (`execution.udf_isolation`, default `"env"`). Demonstrated:
   under `"none"` a worker reads `AWS_SECRET_ACCESS_KEY`, under `"env"` it reads nothing.
   **Covers the process path only** — a UDF on the thread path *is* the engine process, and
   that limit is pinned as a test rather than left to be discovered.

2. **Every artifact was world-readable.** Measured, not assumed: spill files 0644 with the
   directory protected only by a `chmod` that ignores its own failure; shuffle scratch root
   0755 on a shared cluster mount; `/dev/shm` UDF shards (the query's batches) 0644 in a
   world-writable directory; event-log documents (which carry literal predicate constants)
   0644; the learned-stats database 0644. Now 0700/0600 by construction, via
   `_internal/paths.py` and `bc-runtime/agg/spill.rs::create_private`.

3. **A governed column's bloom filter was an exact membership oracle.** Persisted on every
   write into a `MetadataHub` whose backends include Redis and object storage. Probing the
   *actually persisted* artifact through its *actual* consumer answered PRESENT for both
   real SSNs and absent for three fabricated ones — a read of a governed column that never
   touches the table, and (no false negatives) an `absent` is proof of absence. Now
   redacted for masked/invisible columns; cardinalities survive so the optimizer still
   orders joins.

4. **Governance failed open.** Enforcement applied only inside a `security()` block, so
   forgetting the block silently unmasked a column. `GovernanceConfig.mode`
   (`off`/`advisory`/`strict`) makes it mandatory; `strict` also refuses sources that
   cannot be governed at all rather than exempting them. `advisory` is what makes `strict`
   adoptable and is not optional polish.

**Still open, and unchanged by any of the above:** Batcher does not authenticate. A
`Principal` is caller-asserted, so in-process code can claim any identity. The trust
boundary is the process. `docs/user-guide/hardening.md` says so in the user's language.

## Open — ranked

These are audited, located, and reproducible; they are not yet fixed.

### Tier 1 — measurement corruption

*All three original Tier-1 items are now resolved (E1–E3 above), except the residual note
below.* The hash table's own overhead is still not measured: `peak_bytes` accounts for the
materialized input and the result, not the hash table an aggregate/join builds on top. That
is an under-count of a bounded factor rather than the ~1000x one that existed, and closing
it needs the engine to instrument its memory pool per operator.

### Tier 2 — loops that never close

4. ~~Only the **disk aggregate** distributed path is metered.~~ *Fixed in L17 for the disk
   sort/window/join/distinct paths and the broadcast probe.* **Still open:** the Flight
   transport paths (`flight_{aggregate,join,sort,window}.py`), `dist/executors/write.py`,
   and the out-of-core `reduce_join_paths_spilling` branch remain unmetered.
5. The distributed branch still records strictly less than single-node at the *query* level
   (no `record_execution`, no `learn_column_stats`, no `record_selectivity`, no
   `record_run_feedback`). Per-operator feedback now flows (L17), so the gap is narrowed
   to the plan-level learned state, but "single-node == distributed" still holds for
   *results* and not yet for all *learned state*.
6. The spill path and the streaming path emit no operator feedback at all. The largest
   queries — the ones that spill — contribute least to the memory model that decides
   spilling.
7. ~~The join-strategy bandit's reward is whole-query wall time... mixing input sizes makes an
   arm sampled early on large inputs look permanently worse.~~ *Fixed in L20 (size-normalized
   reward + scale-aware radius).* **Still open:** the reward is still whole-query wall time
   attributed to the single join, and **multi-join plans never tune** (`len(joins) != 1`
   returns early). Now that every join reports its own `t_op_ms` (L17), the reward could be
   the operator's own measured time, which would also unlock multi-join plans.
8. ~~Distributed adaptive never records a flip.~~ *Fixed in L18.*

### Tier 3 — optimizer soundness claims that are not enforced

9. `Rule.idempotent` is threaded through the registry and **never read**. Fixpoint phases
   assume confluence by convention; nothing asserts it. (Non-convergence is now at least
   *reported* — see L12 — but not attributed to a rule.)
10. `plan_signature` erases literal values, so `WHERE x < 0` and `WHERE x < 1e9` share a
    signature and therefore a learned row count / selectivity. Contained today only because
    every result-affecting rule gates on `EXACT`; it is a live landmine the moment a rewrite
    consumes `LEARNED`.
14. Greedy is myopic where DP is exhaustive (inherent), but both now rank by the same
    incremental term at the same orientation. *(The audit's "different recurrences" claim
    was overstated: greedy's accumulated prefix is common to every candidate in a step, so
    it cancels from the argmin. Only the redundant subtree re-walk was real, and is gone.)*

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
21. `n_max_parallelism == 0` / `c_max_credits == 0` are overloaded: the admission
    counter-offer emits 0 meaning "unconstrained", the scheduler reads 0 as "unsized → use
    defaults".
23. Sketches: HLL omits the HLL++ bias-correction tables in the mid-cardinality zone;
    Count-Min applies neither conservative-update nor the mean/noise-floor correction, so
    heavy-hitter frequencies (which drive skew salting) are biased high.
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
* **`threads` is not hardcoded to 1 on the path that matters.** `par.rs` reports
  `rayon::current_num_threads()`. Only the *sequential oracle* records `1`, where 1 is the
  correct answer. The CPU-share learner is fed by the parallel path.
* **`outer_to_inner_join`'s null-rejection analysis is correct** under 3-valued logic, as
  are `join_to_semijoin`, `push_semijoin_through_join`, `infer_join_predicates`,
  `constant_propagation`, `factor_common_conjuncts`, `combine_limits`, and the
  `UNION ALL`-only gating of the limit pushdowns.

## Rejected on inspection
* **Count-Min conservative update (Estan-Varghese) is missing.**
  Rejected, twice over. (1) `CountMinSketch` has **no consumer**: `heavy_hitters` — the only
  skew/frequency path — is backed by Misra-Gries (`FrequentItems`), and no crate outside
  `countmin.rs` references it. Tightening an unused estimator changes nothing observable.
  (2) Conservative update is **not mergeable**, which `rust-engine.md` requires ("partition-built
  sketches merge identically"). Measured over 3000 random instances (W=8, D=3): plain Count-Min's
  merged estimate equals the single-node estimate in **100%** of trials; conservative update
  diverges in **96%**. It stays *sound* (never undercounts), but single-node and distributed would
  disagree on the same data — exactly the invariant the mergeable algebra exists to protect.


* **`should_spill` returns `False` for un-estimable plans (claimed inconsistent with the scheduler).**
  Not a live bug. `peak_operator_bytes` takes the *max* over per-operator bounds, so the
  `estimated <= 0` branch needs a plan with **zero** sized operators. Measured through the real
  conductor: `scan+filter` -> 1 MiB, `aggregate` -> 1 MiB, `sort` -> 3.2 MB — never 0. And the
  semantics is right anyway: a plan with no sized breaker is a streaming pipeline, which never
  needs to spill. Chasing it did surface a real defect next door — see L22.
