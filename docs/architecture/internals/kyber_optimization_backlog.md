# Kyber optimization backlog: the measured metadata the optimizer does not yet spend

**Status:** audit and proposal catalog, 2026-08-04. Every claim about Batcher below was checked
against the code in this tree, not against its documentation, and the greps and scripts that
produced each number are recorded inline so they can be re-run. Two findings are demonstrated
with a runnable script rather than asserted.

This document answers two questions:

1. Are the hardware metrics Batcher collects actually observed and acted upon? (Part 1. The
   short answer is that most of them are collected, shipped, stored, and then read only by
   `EXPLAIN`.)
2. Given the metadata Batcher has that a static optimizer does not, what optimizations become
   possible? (Part 3 catalogues 124 proposals; Part 4 gives each a verdict, Part 5 ranks the
   survivors, and Part 6 records what shipped.)

`competitive_architecture.md` is the scorecard and outranks this file on any competitive claim.
Its verified lead #3, "a learned cross-query loop nobody else has", is the thing this catalog
tries to spend. Nothing here restores a claim that document retires: in particular the adaptive
loop is stage-boundary adaptation, the same granularity as Spark AQE, and several proposals
below exist precisely because it is not yet finer than that.

## Method and its limits

Read: all 200 files under `python/batcher/kyber/`, `python/batcher/metadata/`,
`python/batcher/_internal/hardware/`, `python/batcher/plan/feedback.py`,
`python/batcher/plan/profile/`, `python/batcher/plan/resource/`, and the distributed hardware
probe under `python/batcher/dist/executors/ray_runtime/`. Rule counts come from the live
registry, not from counting decorators.

Three limits bound how much weight this carries:

1. The proposals are design sketches, not measured wins. None has been benchmarked. Where a
   proposal's value depends on a number nobody has measured, the entry says so.
2. The audit covers the CPU and memory metric path in depth. The GPU path
   (`kyber/gpu/`, `carbonite/accel/`) was read for its metric sources but not audited to the
   same depth, and the device tier cannot be verified without hardware
   (`.claude/rules/device-tier.md`).
3. Findings H1 through H4 concern the distributed path, which CI never executes. They were
   reproduced in-process by constructing the states a cluster produces, which is evidence about
   the code's logic and not a substitute for a recorded cluster run.

## Part 1: the hardware-metric audit

### What Kyber is given, and what it reads

`OperatorFeedback` (`plan/feedback.py`) carries 26 fields per operator per execution: two of
identity (`op_id`, `kind`), two of provenance (`signature`, `hw_fingerprint`), and 22
measurements. Counting consumers in `kyber/` and `carbonite/`:

| Field | Read by Kyber | Read by Carbonite | Where it actually goes |
|---|---|---|---|
| `selectivity` | yes (21 sites) | 1 | the estimator, `measured_selectivity` |
| `signature` | yes (12) | 0 | the learning loop's key |
| `n_actual`, `n_estimated` | yes | 1 | q-error correction |
| `expr_factor` | yes (4) | 0 | divided back out in calibration |
| `batch_size` | yes (5) | 0 | UDF sizing |
| `t_op_ms` | yes (1) | 0 | cost calibration |
| `backend` | yes (2) | 0 | GPU/CPU crossover only |
| `cpu_utilization` | yes (1) | 0 | `cpu_shares` |
| `m_peak_bytes` | **no** | 2 | Carbonite's learned memory model |
| `spill_bytes`, `peak_rss_bytes`, `n_build` | **no** | 1 each | Carbonite |
| `threads`, `result_bytes`, `algorithm` | **no** | **no** | `EXPLAIN` text only |
| `minor_faults`, `major_faults` | **no** | **no** | `EXPLAIN` text, `observe/insights` |
| `vol_ctx_switches`, `invol_ctx_switches` | **no** | **no** | as above, plus one suppression gate |
| `io_read_bytes`, `io_write_bytes` | **no** | **no** | `EXPLAIN` text only |
| `hw_fingerprint` | **no** (see H4) | **no** | a filter that discards rows |

Reproduce with:

```bash
for f in minor_faults major_faults vol_ctx_switches invol_ctx_switches io_read_bytes \
         io_write_bytes threads result_bytes algorithm m_peak_bytes; do
  printf "%-20s kyber=%s carbonite=%s\n" "$f" \
    "$(grep -rn "\.$f\b\|\"$f\"" python/batcher/kyber --include='*.py' | wc -l)" \
    "$(grep -rn "\.$f\b\|\"$f\"" python/batcher/carbonite --include='*.py' | wc -l)"
done
```

Six per-operator hardware counters are measured in Rust, flattened into the metrics document,
shipped across FFI, stamped with a machine fingerprint, stored, bounded and pruned. Four of
them — `minor_faults`, `vol_ctx_switches`, `io_read_bytes`, `io_write_bytes` — reach no
decision at all, and are used only to print a line such as `disk-read=1.2 GiB` in
`plan/profile/types.py:407`. Part 3 section B is what to do with them.

**Correction, on review before implementing.** An earlier draft of this section said all six
were unused. That was wrong, and the error mattered enough to fix rather than quietly drop:
`plan/feedback.py::oversubscribed` already consumes `invol_ctx_switches`, `major_faults` and
`threads`, and `kyber/cpu_shares.py:152` acts on it — suppressing a CPU-share shrink when a
family's low utilization is explained by contention rather than idleness. That is a real,
carefully argued use of the counters, and the surviving claim is narrower: the counters inform
one *suppression gate* and no plan-shape decision.

### H1. The cost model's cache term reads the driver's cache, not the worker's

`kyber/cost/terms.py:112` calls the process-local `l3_cache_bytes()`:

```python
def cache_factor(state_bytes: float) -> float:
    cache = float(l3_cache_bytes())
```

`HardwareProfile.l3_cache_bytes` exists for exactly this, is populated for a cluster by
`cluster_l3_cache_bytes()` (the minimum across probed node shapes), and is threaded into
`CostModel.__init__` as `hardware`. The cost model spends it on `memory_budget` and the
locality factor, and never on `cache_factor`, which is called at five sites
(`cost/model.py:317, 355, 372, 407` plus the join probe) and prices the dominant term in join
ordering, distinct, aggregate, and distinct union. `terms.py`'s own comment calls the cache
knee "precisely what join ordering is choosing between".

Measured on this machine, planning for a worker with an 8 MiB L3:

```
driver L3 bytes: 25952256
profile.l3_cache_bytes (binding worker): 8388608
cache_factor(256MiB) as costed              : 2.1797
cache_factor(256MiB) if it used the worker's: 2.75
```

A 26% error here, and much wider on a real fat-driver shape (a `c6i.metal` driver publishes
over 100 MiB of L3 against a 4 MiB worker), in the term that decides build side and join order.
Single-node is unaffected, because there the two sources agree.

**Fixed.** `cache_factor` takes an optional `cache_bytes`, and `CostModel` supplies the target
node's cache. `None` keeps the local probe, so single-node ranking cannot move. Pinned by
`tests/unit/test_cost_worker_cache.py`.

### H2. Spill is priced against the driver's disk, and there is no channel to fix it

`terms.spill_io` scales by `spill_device_factor()` (`kyber/storage_cost.py`), which resolves
*this process's* spill directory and reads its device class. `storage_cost.py`'s own docstring
puts the spread across device classes at "roughly thirty-fold, and it runs in the direction that
changes the decision".

Unlike H1 this cannot be fixed by threading an existing field, because `plan.resource.
HardwareProfile` carries no storage class at all. The worker probe already collects it
(`_profile_on_this_worker` returns `hardware_profile().to_dict()`, which includes
`storage_class`), so the measurement exists and is discarded at the driver.

**Fixed.** `HardwareProfile.storage_class` carries the binding worker's measured device class
(`cluster_storage_class()` takes the **worst** across probed shapes, as the binding-node rule
requires), and `spill_io`/`merge_io` take it. `""` reads the local volume, so single-node is
unchanged. Pinned by `tests/unit/test_cost_storage_awareness.py`.

### H3. The worker probe collects fifteen fields and two are consumed

`dist/executors/ray_runtime/hardware_probe.py` runs one task per distinct node shape and
returns the whole `HardwareProfile.to_dict()`, deliberately:

> The whole profile rather than one number, because the probe's cost is the round trip and
> every additional field is free once the task has been scheduled.

The only readers of `cluster_hardware_profiles()` are `cluster_l3_cache_bytes()` (reads
`caches.l3`) and `cluster_is_heterogeneous()` (reads `fingerprint`). Discarded at the driver:
`storage_class`, `numa_nodes`, `physical_cores`, `simd_bits`, `page_bytes`,
`memory_per_core_bytes`, `vendor`, `model`, `fabric_class`, `platform_system`, `accelerators`,
`logical_cpus`, `memory_bytes`. The round trip is already paid.

### H4. Cost calibration and CPU-share learning discard every worker measurement on a mixed fleet

This is the sharpest finding, and it is demonstrable.

`MetadataHub.op_stats_by_kind()` **filtered** rows through `measured_here()`, which compared the
row's `hw_fingerprint` against *this process's* fingerprint. On a distributed run the driver's
hub receives worker rows correctly stamped with the worker's fingerprint (metering.py:96 stamps
them at the worker, and `core/executor.py:146` preserves the stamp). The driver then filters
them all out, because the driver is not a worker.

Both consumers of that view run on the driver and are about worker work:

* `kyber/calibration.py:180` fits the cost coefficients that rank every plan.
* `kyber/cpu_shares.py:145` sizes the `num_cpus` a *distributed task* requests. This module's
  entire purpose is the distributed path.

Reproduced with 500 worker-measured operators against a driver of a different class:

```
driver fingerprint: e714b847b624 | simulated worker: ffffffffffff

worker-measured (distributed run)
  rows stored: 500 | visible to calibration: {}  <-- ALL DROPPED
  coefficients moved: False
  cpu_shares learned utilization: {}  <-- nothing learned

driver-measured (single-node)
  rows stored: 500 | visible to calibration: {'hash_join': 500}
  cpu_shares learned utilization: {'hash_join': 0.95}
```

The machine-scoping in `metadata/hardware_scope.py` is right, and the module is careful and
well-argued: a coefficient in nanoseconds per row genuinely does not transfer between machine
classes. The defect is that the *consumer* scopes to the wrong machine. Kyber builds the plan
for the binding **worker** (`HardwareProfile.for_cluster` is explicit that every field is the
weakest worker) and then calibrates that plan's coefficients from rows measured on the
**driver**. The shape of the model and the constants in it are scoped to two different
machines.

Note the failure mode: no error, no warning, and a plan that is merely mis-ranked. On a
homogeneous fleet where the driver runs the same instance type, it works, which is why it
survives.

**Fixed.** The view is now *bucketed* per machine class rather than filtered to one, and
`op_stats_by_kind(hw_fingerprint)` names the class to read. `HardwareProfile.fingerprint` carries
it, populated by `cluster_worker_fingerprint()` — the key every probed worker shares, `""` on a
mixed fleet, following the same rule `accelerator_type` does. `calibrate` and
`load_cpu_utilization` take it and include it in their memo keys. `measured_here` became dead and
was deleted. Pinned by `tests/unit/test_learning_scoped_to_target_machine.py`, whose first
assertion is the one that used to fail.

### H5. Calibration can only learn relative family costs, never absolute machine speed

`calibration._calibrate` chooses a global anchor `k` so that "the default model's total work
over all usable samples equals their total measured ms". With samples from a single operator
family that constraint is a fixed point: the fitted coefficient is algebraically the default,
whatever the measurements say. Demonstrated:

```
one family only, 10x slower than the model expects:
   hash_probe_row 1.0 default 1.0 | moved: False
two families, join 10x slower than filter relative to defaults:
   hash_probe_row 1.449 default 1.0
   filter_row     0.05  default 0.5 | moved: True
```

This is deliberate and documented ("when reality matches the defaults, calibration is a
no-op"), and for pure plan ranking a relative model is sufficient. Two consequences are not
stated anywhere and are worth knowing:

* A workload dominated by one operator family (a scan-filter-write ETL is the common case)
  calibrates nothing, however long it runs.
* A uniform machine-speed error is exactly the error hardware differences produce, and it is
  the one error this method is structurally unable to see. The fingerprint scoping exists to
  separate machine classes, then the fit within a class throws the machine-speed dimension
  away.

**Fix:** proposal A9, an absolute anchor from a fixed micro-benchmark recorded per fingerprint.

### H6. Kyber reads no live signal at all

`_internal/hardware/cgroup.py` and `carbonite/memory/probe.py` collect PSI memory pressure,
`cgroup_throttled_ratio`, `cgroup_current_bytes`, `available_bytes`, and CFS quota, and
Carbonite acts on them. A grep for any of them under `kyber/` returns nothing.

Every hardware input to a plan is either a static capacity or a historical measurement. Two
nodes with identical nominal capacity, one idle and one with a co-tenant consuming 90% of
memory and being CFS-throttled, receive the same plan. Section J is the proposal set.

### H7. The local hardware profile is memoized for the life of the process

`_internal/hardware/profile.py:260` caches `_PROFILE` and `_FINGERPRINT` module-globally, with
`_reset_profile` documented as a test hook. The cluster profile is re-probed when the topology
signature changes; the local one never is. A long-lived session (the notebook and BI-server
cases the hub's own docstring calls out) plans against boot-time capacity forever, which is
wrong after a Kubernetes in-place resize or a cgroup limit change. Minor next to H4, but it is
also the reason a fingerprint can silently stop describing the machine.

### H8. The planner contract omits five probed fields

`plan/resource/HardwareProfile` carries 8 fields. `_internal.hardware.HardwareProfile` measures
13. Absent from the planner's view, all of them probed and all of them changing the right plan:
`physical_cores` (SMT ratio: hash builds scale with physical cores, IO-bound scans with
logical), `numa_nodes`, `simd_bits` (the JIT speedup prior is a flat 4.0 regardless of vector
width), `page_bytes`, `storage_class`, and the derived `memory_per_core_bytes`, whose own
docstring argues it "determines whether a plan should trade memory for parallelism" and which
nothing reads.

### H9. Two FFI surfaces have no consumer

`bc_py::reservoir_sample` and `bc_py::merge_column_stats` are exposed and called from nowhere
in `python/batcher/`. The reservoir is the more interesting of the two: a uniform column sample
is the one statistic that answers *joint* selectivity without an independence assumption, which
is the classic failure of every cost-based optimizer. Section F builds on it.

### H10. The same mis-scoping, generalized: 50 more learned values keyed by the driver

Found while implementing the H4 fix, and it is the same defect one level up. H4 was about a
*reader* filtering to the local machine class. This is about the **key itself**.

`metadata/hardware_scope.scoped(namespace)` qualifies a namespace with `fingerprint()` — always
*this process's*. Fifty call sites across eleven modules store machine-unit quantities under it:

| Module | Sites | What it keys |
|---|---|---|
| `kyber/gpu/adaptive.py` | 10 | the GPU/CPU crossover fit |
| `kyber/learned_tuning/priors.py` | 7 | build-side priors |
| `kyber/learned_tuning/bandit.py` | 6 | UCB1 join-strategy arm latencies |
| `kyber/learned_tuning/crossover.py` | 4 | broadcast and sort-merge crossovers |
| `kyber/gpu/energy.py`, `dist/adaptive_sizing/` | 6 | device power, task sizing |
| `metadata/io_stats.py` | 2 | measured per-source read throughput |
| `ml/gpu.py`, `ml/autobatch.py`, `core/udf/*` | 15 | device utilization, batch sizes |

They split cleanly in two, and only one half is wrong:

* **Written and read on the worker** (`core/udf/*`, `ml/autobatch`, `ml/gpu` — these run inside
  the UDF and inference paths, on whichever process executes them). The local key *is* the
  machine that measured it. Correct as written.
* **Written and read on the driver, describing work done on the workers** (the bandit, the
  crossovers, the priors, `dist/adaptive_sizing`). `api/tuning/decisions.py::record_join_outcomes`
  runs on the driver and records the query's wall time; Kyber reads the arm back on the driver.
  The key is *self-consistent*, so unlike H4 nothing is silently dropped — but it names the
  wrong machine.

The consequence is narrower than H4's and real: an autoscaling fleet whose driver stays put
while its worker instance type changes files both worker classes' measurements under one key,
and a broadcast crossover fitted for small workers is then applied to large ones. This is the
blend `hardware_scope` was written to prevent, arriving through the key rather than the filter.

**Fixed, and not the way this section first proposed.** Threading the class through fifty call
sites is what makes the failure below possible, so the fix is a **scope**, not a parameter:
`hardware_scope.planning_for(fingerprint)` sets an ambient class for its span, `scoped()` falls
back to it, and `api/orchestration/run.py` opens it around the span that both plans a run and
records its outcome. One resolution of the cluster profile, one key, both halves inside it.
`_optimize` now takes the already-resolved profile instead of asking for its own, which is the
duplicate lookup that made the two halves able to disagree.

`dist/executors/ray_runtime/scaling.py::_TOPOLOGY` is the same pattern for the same reason, so
this is the codebase's own idiom rather than a new one. Pinned by
`tests/unit/test_learned_state_keyed_to_target_machine.py`, which asserts the scope restores on
exit *and* on an exception, that an explicit argument still outranks it, that two worker classes
do not blend, and that a single-node run keys everything locally.

**The constraint that made a parameter the wrong fix.** The read and the write must
derive the key from **one** resolved `HardwareProfile`, spanning plan time to record time —
not from two independent calls to `distributed_hardware()`. The reader has `ctx.hardware` in
`rules/selection.py`; the writer is `api/tuning/decisions.py::record_join_outcomes`, reached from
`api/orchestration/run.py` where the profile is a local (`run.py:244`) in a *different* function
from the recorder. Resolving it twice looks equivalent and is not: a cluster that autoscales
mid-query returns a different profile at record time than at plan time, so the arm is written
under a key nothing will ever read. That failure is silent and total — every learned value stops
accruing, with no error and nothing in the result to show it — which is strictly worse than the
blend being fixed. Thread the profile; do not re-resolve it.

Recorded here so the next agent finds the whole shape of the defect rather than the third of it
that was cheap to fix.

### H11. Payload column widths are never learned, and that is the real multimodal gap

Found by building C2/C8, watching it work, and then discovering it was sound for the wrong
reason. The whole episode is recorded because the wrong turn is the instructive part.

**The defect.** `api/terminal/_metadata.py::learn_column_stats` restricts what it measures to
`learnable_columns(plan)` — "the union of the two things the estimator consults: `ndv_columns`
(join keys, group keys, equality predicates) and the columns any `Filter` mentions", with the
stated reason that "a column outside that union has no consumer; sketching it is pure loss".

That reasoning holds for distinct counts, quantile grids and most-common-values. It does **not**
hold for the *average byte width*, because `StatsEstimator.row_width` sums per-column widths over
every output column, not over the predicated ones. So the columns that dominate a row's width —
the payload, the embedding, the image, the document — are exactly the ones never measured.

Demonstrated on a two-column table (`k` int64, `doc` 4 KiB string), four executions of a filter
on `k`:

```
width cold (no runs)        : 44.0 B/row
width after 4 runs          : 44.0 B/row      <- unchanged
true                        : 4104.0 B/row
learned per-column widths   : {'k': 8.0}      <- 'doc' never measured
```

A **93x** under-estimate that no amount of running the query corrects. `annotate.py`'s own table
says what that costs: at the flat assumption a 768-dim embedding sizes 12 GB tasks and a 1080p
frame sizes 25 TB ones. This is that failure, at its root.

**The trap, for whoever fixes it.** The obvious fix — measure widths for every column — breaks a
deliberate invariant. `learn_column_stats` uses *the presence of an avg-byte width* as its
"already sketched" marker, and its docstring explains why that field and not `ndv` (the cheaper
`seed_column_ndv` also writes `ndv`, so gating on it would suppress this pass entirely). Writing
a width for an unsketched payload column therefore marks it as sketched, and its quantiles and
MCVs can never be learned afterwards — a silent regression traded for the fix.

`QUANTILES_KEY` cannot stand in as the marker either, and for a reason the same docstring gives:
`column_statistics` "records one [width] for every column it touches (numeric or not)", whereas a
string column yields no quantile, so a text column would go unmarked and be re-sketched on every
query.

**Fixed** with the design that survives both constraints: a **separate** cheap-width table
(`column_tables.ROW_BYTES_KEY`), written by `api/terminal/_metadata.py::_learn_row_bytes` for
every column of every keyable source, and merged *beneath* the sketched width in
`StatsEstimator.learned_columns`. Arrow already knows each array's buffer size, so the
measurement is `nbytes / num_rows` — O(columns), no sample, no sketch, no per-row work — and the
marker keeps meaning exactly what it meant.

Measured after the fix, on the same table:

```
width cold        : 44.0 B/row
width after 3 runs: 4108.0 B/row      (true: 4104.0)
sketched (marker) : {'k': 8.0}        <- 'doc' still unsketched, still sketchable
cheap widths      : {'k': 8.0, 'doc': 4100.0}
```

`tests/unit/test_payload_column_widths.py` pins both halves of the trap: that a cheap width does
*not* mark a column sketched, and that a later query filtering on that column still sketches it.
Two details a test found rather than review. Writing the widths is best-effort **per source**
rather than inside the caller's `try`, because it runs before the sketch pass and one unreadable
source must not cost every other source its quantiles. And the write is **gated on a material
change**: unlike the sketches it is not gated by the "already measured" marker, so it runs on
every execution, and `merge_column_table` is a whole-table read-modify-write — a served workload
would otherwise pay that write forever to re-record a number that cannot move.

**What was built and reverted.** A per-*signature* measured width (`result_bytes / n_actual`,
`kyber/measured_width.py`) fixed the same symptom and was reverted before it could ship, because
its key is wrong: `plan_signature` deliberately normalizes literals so learning generalizes across
shapes, and it carries no source identity. Measured:

```
wide filter sig  : 0de362859664fbbb
narrow filter sig: 0de362859664fbbb
same signature?  : True

narrow width after 4 runs of the *wide* query: 4108.0 B/row  (true: 16.0)
```

A 4 KiB table's width applied to an unrelated 16-byte table, a 257x over-estimate inherited from
a query over different data. `estimator.py` already documents this hazard for learned *row
counts* — "`plan_signature` structures every scan as the bare token `["scan"]`, carrying no source
identity — so *all* scans in a process share one learned entry" — and excludes scans from
correction because of it. The same hazard applies to width and the same exclusion was missing.
A width is a property of a **schema**, so it must be keyed by source and column, which is exactly
what `AVG_BYTES_KEY` already is and what H11 is about filling in.

### H12. The scan collision reaches a *shipped* loop: a 500x cardinality error

The bug class every other finding here shares, found once more — this time in live code, and
this time producing a plan that is measurably worse than having learned nothing.

`kyber/measured_selectivity.py` learns a filter's kept fraction under its structural plan
signature, and `StatsEstimator` prefers it over the structural guess. But `kyber/signature.py`
renders every scan as the bare token `["scan"]`, so **two filters of the same shape over
different relations share one entry**. Reproduced on the real engine:

```
selective sig : 0de362859664fbbb      (k < 40 over 0..19,999  -> keeps 40)
permissive sig: 0de362859664fbbb      (k < 40 over i % 39     -> keeps all)
SAME SIGNATURE: True

permissive query, rows estimated cold : 6,667
permissive query, rows estimated warm : 40        <- after 4 runs of the OTHER table
permissive query, rows ACTUALLY       : 20,000
```

A **500x** under-estimate on a table the learning never saw, and *worse than the cold estimate
it replaced*. An error that size flips a join order and a build side.

This is not an unknown hazard in the codebase — it is a **half-applied** one. `signature.py`
names it "the scan-collision defect" in its own `MapBatches` comment and fixes it there by
putting the UDF's identity in the token; `estimator.py` excludes `Scan` from `_CORRECTABLE` for
exactly this reason. Both are per-consumer patches for a defect in the key, and the consumers
that were not patched — measured selectivity, and the q-error correction beside it — still have
it.

**Fixed at the root**, in a later pass, after first shipping the mitigation below.

The fix is the one the `MapBatches` precedent argues for: the scan token carries the source's
identity. It did **not** need the parameter threading through 22 call sites that made this look
dangerous — the identity belongs on the `Scan` node, exactly as `identity_suffix()` already
carries the schema there for `content_key`'s benefit (the same collision, found and fixed for
the plan cache). `Scan` gains a `source_key` field, `api/session/_scan.py` stamps it, and every
one of the 22 call sites gets a consistent key for free, because a node cannot disagree with
itself about what it is.

Two constraints had to hold together, and they pull opposite ways:

* different relations must not share — the defect;
* the **same** relation must still share across runs *and across sizes*, because a selectivity
  is a ratio and its whole value is surviving the table growing.

So the token carries only a **data-stable** identity (`plan.source_stats.stable_source_key`). A
file path stays put as the file grows, satisfying both. An in-memory relation identifies by
schema plus row count, which satisfies neither — two unrelated frames of the same shape collide,
and one frame grown by a row stops matching itself — so it contributes nothing and keeps the old
shared token. Measured on two parquet files:

```
distinct signatures: True
perm.parquet estimated cold: 6,667   warm: 6,667   actual: 20,000
same file, new handle -> same signature: True
```

The 500x poisoning is gone, and cross-run learning for one table still accumulates. Four
existing tests failed on the first attempt at this — including one asserting that a selectivity
*ratio* generalizes over input size — and every one of them was pointing at a real flaw in the
first design rather than needing to be weakened; narrowing the token to data-stable identities
made all four correct again with no test changed.

**The mitigation shipped first and stays.** A confidence gate: a signature whose observed
selectivities are not concentrated is refused, and the estimator falls back to the structural
guess. It was the right thing to land before the key fix, and it is still the right thing to
keep after it — it now covers the deliberate residual (same-shaped in-memory relations) and the
genuinely multi-modal case, which no key can fix.

```
after A only  : {'0de3628596': 0.002}
after A and B : {}  <- REFUSED by the gate
true A: 0.002   true B: 0.5   -- a mean of these predicts neither
```

`cpu_shares` already applied this exact test to its utilization medians for the same reason, so
the predicate was lifted to `_internal.mathx.is_concentrated` and both now read it rather than
the codebase carrying it twice. Pinned by `tests/unit/test_selectivity_confidence_gate.py`.

The gate bounds the damage to "no worse than not learning"; the key fix above makes the key
correct. Together they close this finding.

### Findings summary

| # | Finding | Severity | Scope | Status |
|---|---|---|---|---|
| H4 | Worker measurements discarded by the driver's calibration and CPU-share loops | High | distributed, heterogeneous | **fixed** |
| H1 | `cache_factor` uses the driver's L3 | Medium | distributed | **fixed** |
| H2 | Spill priced against the driver's disk; no field to carry the worker's | Medium | distributed | **fixed** |
| H6 | No live pressure signal reaches the planner | Medium | shared and containerized nodes | deferred (Carbonite's lane) |
| H5 | Calibration structurally blind to absolute machine speed | Medium | all | by design, documented |
| H3 | 13 of 15 probed worker fields discarded | Low, enabling | distributed | **2 now consumed** |
| H8 | 5 probed fields absent from the planner contract | Low, enabling | all | **1 added** (`storage_class`) |
| H7 | Local profile never re-probed | Low | long-lived sessions | rejected (`reset_hardware_probes` exists) |
| H9 | Reservoir sampling and stats merging unreachable from Python | Low, enabling | all | deferred |
| H10 | 50 learned values keyed by the driver's machine class, not the workers' | Medium | autoscaling / mixed fleets | **fixed** |
| H11 | Payload column widths never learned; the real multimodal width gap | Medium | wide/multimodal data | **fixed** |
| H12 | Scan collision poisons a shipped selectivity loop: 500x, worse than cold | **High** | any two same-shape filters | **fixed at the root** |
| B | Six per-operator hardware counters drive no decision | Low, enabling | all | corrected: four, one gate |

## Part 2: where the rule surface is empty

From the live registry (`PYTHONPATH=python python -c "from batcher.kyber.registry import
DEFAULT_REGISTRY; ..."`):

| Phase | Rules |
|---|---|
| NORMALIZE | 546 |
| REWRITE | 81 |
| PUSHDOWN | 68 |
| FUSION | 22 |
| SELECTION | **3** |
| ENFORCE | **2** |
| JOIN_REORDER | **1** |

723 rules, and 75% of them are scalar expression algebra. The three SELECTION rules are
`adaptive_build_side`, `size_gpu_map_batches`, and `split_expensive_filter`. Physical algorithm
choice, the half of an optimizer that measurement is *for*, is three rules wide.

That asymmetry is the thesis of Part 3. The algebraic surface is dense and mostly saturated:
another twenty `CASE` simplifications are worth little, because the engine's competitors have
those too and they are not where queries spend time. The measurement-driven surface is nearly
empty and is the one place Batcher has inputs its competitors do not.

## Part 3: the catalog

124 proposals. Each names the metadata it consumes, and marks whether that metadata is already
collected (**have**), needs a new probe or sketch (**new**), or is collected and currently
discarded (**dropped**, i.e. Part 1).

Legend for "home": the phase or module the work belongs in.

### A. Hardware plumbing that is also new capability (10)

These close Part 1 and are prerequisites for much of the rest.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| A1 | Thread `HardwareProfile.l3_cache_bytes` into `cache_factor`, as `spill_io` already takes `budget` | have | `cost/terms` |
| A2 | Add `storage_class` to `HardwareProfile`; price spill against the binding worker's device | dropped (H2) | `plan/resource`, `storage_cost` |
| A3 | `op_stats_by_kind(fingerprint=...)`; Kyber calibrates against the binding worker's class | dropped (H4) | `metadata/hub`, `calibration` |
| A4 | Carry the binding worker's fingerprint on `HardwareProfile` so A3 has an argument to pass | dropped (H3) | `plan/resource` |
| A5 | Carry `physical_cores`; size hash-build DOP by physical cores and scan DOP by logical | dropped (H8) | `annotate`, `cost` |
| A6 | Carry `numa_nodes`; charge a crossing term when a build exceeds one node's memory share | dropped (H8) | `cost/terms` |
| A7 | Carry `simd_bits`; scale the JIT speedup prior by vector width instead of a flat 4.0 | dropped (H8) | `expr_cost/jit` |
| A8 | Carry `memory_per_core_bytes`; choose memory-heavy vs parallelism-heavy plans by machine shape | dropped (H8) | SELECTION |
| A9 | Absolute calibration anchor: a fixed micro-benchmark per fingerprint, so machine speed is learnable | new | `calibration` |
| A10 | Re-probe the local profile when the cgroup limit changes, instead of memoizing for the process | have | `_internal/hardware` |

### B. The six dead per-operator counters (12)

Nothing in this section is possible in DuckDB, Polars, or Spark's optimizer, because none of
them records per-operator page faults or context switches across executions. Batcher already
does, and spends them on a log line.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| B1 | Pressure-confirmed spill budget: a family whose history shows major faults at a given state size gets a lower effective budget than config claims | `major_faults` (dropped) | `cost/terms` |
| B2 | Demote a broadcast that historically paged, even when its byte size says it fits | `major_faults` + `m_peak_bytes` | SELECTION |
| B3 | Cap plan DOP from measured involuntary-preemption rate per machine class | `invol_ctx_switches` | `annotate` |
| B4 | Separate "IO-bound, pack more" from "contended, back off" using voluntary vs involuntary switches. `cpu_shares.py` documents this ambiguity as the reason it suppresses learning; the two counters resolve it | both switch counters | `cpu_shares` |
| B5 | Measured read amplification per source (`io_read_bytes` vs `result_bytes`) exposes projection pushdown that is not reaching the reader | `io_read_bytes` | PUSHDOWN, diagnostics |
| B6 | Measured bytes per row per source, replacing the learned `avg_bytes` estimate for IO costing | `io_read_bytes` + rows | `cost/model` |
| B7 | Detect hidden materialization: `io_write_bytes` on an operator that did not spill | `io_write_bytes` | diagnostics, FUSION |
| B8 | Working-set locality score per family from the minor/major fault ratio, driving morsel size | both fault counters | `annotate` |
| B9 | Trigger adaptive re-optimization on fault rate, not only on row-count divergence | `major_faults` | `api/adaptive/gating` |
| B10 | Invalidate a cached plan when the same plan's counters degrade materially | all six | `plan_cache` |
| B11 | Infer page-cache residency per source from the fault mix; prefer the resident side as the join build | fault counters | JOIN_REORDER |
| B12 | Suppress bandit arm updates from runs with anomalous contention, so one noisy run cannot poison an arm | switch counters | `learned_tuning/bandit` |

### C. Measured memory (12)

`m_peak_bytes`, `spill_bytes`, `peak_rss_bytes`, `result_bytes`, and `n_build` are recorded on
every operator of every execution. Kyber reads none of them; Carbonite reads some. Kyber's
memory reasoning is entirely estimated.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| C1 | Broadcast eligibility from the *measured* build-side peak for that signature, not an estimated width | `m_peak_bytes` (dropped) | SELECTION |
| C2 | Measured bytes per build row per join shape, superseding propagated width estimates | `m_peak_bytes` / `n_build` | `cost/model` |
| C3 | A "this shape spills" flag that changes join strategy before the run rather than during it | `spill_bytes` | SELECTION |
| C4 | Learn the true spill threshold from where spilling actually began, instead of trusting config | `spill_bytes` | `cost/terms` |
| C5 | Measured allocator overhead per machine class: `peak_rss_bytes` against summed `m_peak_bytes` | `peak_rss_bytes` | `cost/terms` |
| C6 | Enumerate join orders for minimum *peak memory* when history shows this shape spilling | `m_peak_bytes` | JOIN_REORDER |
| C7 | Price pipeline-breaker placement by the measured cost of materializing that intermediate | `result_bytes` | FUSION |
| C8 | Measured output width per operator, replacing width estimates that decay through joins | `result_bytes` / rows | `stats/derived` |
| C9 | Plan to a memory *quantile* rather than the median, so tail runs stop OOMing | distribution of `m_peak_bytes` | `annotate` |
| C10 | Memory-aware DOP: peak scales with partitions, so choose DOP from measured peak against the envelope | `m_peak_bytes` | `annotate` |
| C11 | Correct the external-merge fan-in from measured merge-pass counts instead of the analytic formula | `spill_bytes`, `t_op_ms` | `cost/terms` |
| C12 | Serialize two breakers in one stage when their measured peaks sum over budget | `m_peak_bytes` | SELECTION |

### D. Physical algorithm selection (18)

The three-rule phase. Note that the Rust runtime already switches some algorithms adaptively
(`bc-interp/src/agg_par.rs` on measured reduction ratio, per the scorecard). These proposals
are about Kyber *choosing*, which is what lets the choice be costed, explained, and learned
across queries rather than rediscovered inside each execution.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| D1 | Aggregate algorithm choice (hash / sort / partial-then-sort) from measured reduction ratio per signature | `selectivity`, `n_actual` | SELECTION |
| D2 | Distinct algorithm choice from measured NDV-to-rows ratio | NDV table | SELECTION |
| D3 | Sort strategy (full / top-N heap / partial) from the measured k-to-n ratio | `topn_bound` (partly landed) | SELECTION |
| D4 | Window execution (sorted vs hash-partitioned) from measured partition count and skew | MCV, NDV | SELECTION |
| D5 | Semi-join implementation (hash / bloom / IN-list) from measured build NDV | NDV | SELECTION |
| D6 | Union implementation (concat vs merge) from the propagated ordering property | `properties.py` | SELECTION |
| D7 | Choose the execution tier per operator from measured `backend` history, not only for GPU | `backend` (dropped for CPU tiers) | SELECTION |
| D8 | Per-family morsel size, from measured cache behavior, instead of a fixed 16,384 | B8, `t_op_ms` | `annotate` |
| D9 | Per-operator batch size from recorded `batch_size` history beyond the UDF path | `batch_size` | `annotate` |
| D10 | Materialize vs recompute a common subexpression, from measured expression cost and result bytes | `expr_factor`, `result_bytes` | FUSION |
| D11 | Late materialization: decide which columns to carry through a join from measured widths and downstream use | column widths | PUSHDOWN |
| D12 | Choose the build side by measured probe cost rather than estimated rows (`adaptive_build_side` today uses the estimate) | `t_op_ms` per side | SELECTION |
| D13 | Choose spill compression from measured spill throughput on that device class | `spill_bytes`, `t_op_ms` | `cost`, Carbonite hand-off |
| D14 | Per-source prefetch depth from measured throughput and latency | `io_stats` | `annotate` |
| D15 | Per-source scan fan-out from measured throughput, extending `relative_read_cost` from pricing to parallelism | `io_stats` | `annotate` |
| D16 | Choose zonemap vs bloom vs plain scan per predicate from the measured pruning yield of each | new: pruning yield | SELECTION |
| D17 | Build a runtime filter only when its measured historical selectivity justifies the build cost | `selectivity` | ENFORCE |
| D18 | Route per *operator* to GPU or CPU from a per-kind measured crossover, not per plan | `backend`, `gpu/adaptive` | SELECTION |

### E. Join ordering and enumeration (14)

One rule today. This is the classic optimizer surface, and Batcher's measured q-error history
makes several variants possible that a static optimizer cannot express.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| E1 | A cost-based enumerator over measured intermediate sizes, replacing the single heuristic rule | learned cardinality | JOIN_REORDER |
| E2 | Admit bushy plans when measured intermediates justify them | learned cardinality | JOIN_REORDER |
| E3 | Prefer orders whose estimates are historically *reliable*, using per-signature q-error variance | `correction.py` confidence | JOIN_REORDER |
| E4 | Robust ordering: minimize worst case over the measured error distribution, not expected cost | q-error distribution | JOIN_REORDER |
| E5 | Order by measured build memory when C3 says the shape spills | `m_peak_bytes` | JOIN_REORDER |
| E6 | Reuse a measured sub-plan cardinality for any structurally matching sub-plan during enumeration | signatures | `cardinality` |
| E7 | Remember the winning order per query signature across queries, so enumeration is paid once | new: order memo | `plan_cache` |
| E8 | Re-order the remaining joins at the first breaker from the measured first-stage size | adaptive loop | `api/adaptive` |
| E9 | Detect star schemas from measured NDV ratios and inferred key evidence | NDV, `join_elim/evidence` | JOIN_REORDER |
| E10 | Choose the semi-join reduction order from measured filter selectivities | `measured_selectivity` | PUSHDOWN |
| E11 | Skew-aware ordering: place the skewed side where the measured MCV table says it costs least | MCV | JOIN_REORDER |
| E12 | Drive from the table with the most selective *measured* predicate, not the most selective estimated one | `measured_selectivity` | JOIN_REORDER |
| E13 | Session-level ordering: when a sub-plan is shared across queries, order for reuse | signatures | `plan_cache` |
| E14 | Joint selectivity of a conjunction from measured history, dropping the independence assumption entirely | `measured_selectivity` per signature | `stats/selectivity` |

**E14 is already landed, and this entry was wrong.** `kyber/measured_selectivity.py` records the
measured selectivity of a *whole* filter under its structural signature, and a filter whose
predicate is `a AND b` is one operator — so the stored value already *is* the joint rate, with
no independence assumption, and `StatsEstimator._selectivity` already prefers it over the
structural guess. Independence between predicates is the largest source of cardinality error in
the published literature, and Batcher had removed it before this catalog proposed removing it.

What is genuinely open is the case that module's own "What this does not fix" names: a shape's
**first** execution, which has nothing measured yet. That is F1 and F2, not E14. Recorded at
length because re-implementing landed work is the specific failure this document's sibling
ledger warns about, and this catalog nearly did it in its top-ranked item.

### F. Sketches not yet spent (14)

`bc-sketches` ships bloom, count-min, ddsketch, frequent-items, HLL, KLL, reservoir, and
t-digest. Kyber consumes bloom, HLL-derived NDV, KLL-derived quantiles, and an MCV table.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| F1 | Wire `reservoir_sample`, which has an FFI entry point and zero callers, into a per-source column sample | new (H9) | `stats` |
| F2 | Evaluate a conjunct against the sample for joint selectivity when no measured history exists yet (the cold-start half of E14) | F1 | `stats/selectivity` |
| F3 | Discover column correlations and functional dependencies from the sample, feeding the FD rules that today need a declared key | F1 | REWRITE |
| F4 | Count-min over join keys to predict skew before the run rather than detecting it during | count-min (unwired) | JOIN_REORDER, `dist` |
| F5 | Predict the *maximum* partition size from a t-digest tail instead of the mean | t-digest | `dist`, `cost/imbalance` |
| F6 | KLL boundaries for range partitioning in sorts and shuffles (partly landed: commit `3ac2e287` added a sort range grid) | KLL | `dist` |
| F7 | Join cardinality by HLL inclusion-exclusion over key sketches, instead of the NDV-ratio formula | HLL | `stats/join_columns` |
| F8 | Persist the build-side bloom across queries for cross-query semi-join pruning | bloom | `learning` |
| F9 | Sketch a scan once per session and reuse it for every downstream operator | all | `stats/estimator` |
| F10 | Multi-column NDV sketch for group-by cardinality, replacing the product-of-NDVs overestimate | new: joint HLL | `stats/aggregate_columns` |
| F11 | Frequent-items on group keys to decide the partial-aggregation payoff exactly | frequent-items | REWRITE (`eager_aggregation`) |
| F12 | DDSketch over per-operator latency so plans can be ranked at p95, not the median calibration uses | ddsketch | `calibration` |
| F13 | Synthesize zone maps for formats that publish none (CSV, JSON) after the first read | `column_tables` | `zonemap_pruning` |
| F14 | Cost a UDF by measuring it on the sample instead of assuming a constant | F1, `expr_cost` | `expr_cost` |

### G. The adaptive loop (12)

The scorecard is explicit that this loop is stage-boundary adaptation, the same granularity as
Spark AQE, and off below 20M input rows. These proposals are what would make the finer claim
true.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| G1 | Lower the size gate further by making re-planning cheap enough to always afford (the plan cache is the enabler). Partly done: the flat 20M floor is now 5M **per pipeline breaker the loop would cut at**, so a two-breaker plan engages at 10M — the rest of the item is making a cut cheap enough that the floor can fall further | `plan_cache` | `api/adaptive/gating` |
| G2 | Re-optimize on measured *memory*, not only measured rows | `m_peak_bytes` | `api/adaptive/gating` |
| G3 | Re-optimize on selectivity divergence, which is detectable earlier than row-count divergence | `selectivity` | `api/adaptive/gating` |
| G4 | Re-optimize at morsel boundaries within a scan, which is the granularity the "continuous" claim needs | new | `api/adaptive` |
| G5 | Abort and re-plan when measured q-error exceeds what the correction model's confidence band allows | `correction.py` | `api/adaptive` |
| G6 | Speculatively run two algorithm arms on the first morsels and keep the winner | `bandit` | `api/adaptive` |
| G7 | Progressive optimization: emit a partial result, then re-plan the remainder against what it revealed | new | `api/adaptive` |
| G8 | Adapt the spill threshold to live memory during the run | J1 | Carbonite hand-off |
| G9 | Re-plan the remaining joins after each build completes, using the true build size | adaptive loop | `api/adaptive/plan_surgery` |
| G10 | Feed each adaptive decision back to the bandit as a labeled sample, so intra-query learning compounds across queries | `bandit` | `learned_tuning` |
| G11 | Extend adaptive conjunct reordering (landed for filters) to join predicates and window frames | `ConjunctOrder` | Rust + SELECTION |
| G12 | Choose the partition count for later shuffles from the first shuffle's measured skew | `dist/skew` | `dist` |

### I. Cross-query and workload level (12)

This is the section with no analogue in a static optimizer, because it requires a durable
learned store, which Batcher has and DuckDB and Polars do not.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| I1 | Cluster query signatures into a workload fingerprint and pre-warm the plans of a recurring workload | signatures | `learning` |
| I2 | Recommend a materialized intermediate when a sub-plan signature recurs across queries | signatures | new pass |
| I3 | Result cache keyed on lowered IR plus source version | `plan_cache` | `api/terminal` |
| I4 | Recommend the physical sort order for a source, from the predicates measured against it | `measured_selectivity` | advisory |
| I5 | Recommend sink partitioning from the filters measured downstream of it | `measured_selectivity` | advisory |
| I6 | Choose which columns get zone maps from measured predicate frequency | `measured_selectivity` | advisory |
| I7 | Run the UCB1 bandit over whole plan alternatives, not only join strategies | `bandit` | `learned_tuning` |
| I8 | Record counterfactual regret when a plan loses, so the arm value is informative rather than merely a mean | `bandit` | `learned_tuning` |
| I9 | Transfer a prior across machine classes by a measured speed ratio, instead of starting cold. `hardware_scope` currently drops everything on a new class, which is right for correctness and expensive on an autoscaling fleet that mints new classes | A9 | `hardware_scope` |
| I10 | Detect workload drift when measured selectivity leaves the correction window, and say so | `correction.py` | `observe` |
| I11 | Report predicted against measured cost per operator on every run, so the cost model is continuously validated rather than trusted | `t_op_ms` | `observe` |
| I12 | Admit to the plan cache by measured planning cost against measured execution cost | `plan_cache` | `plan_cache` |

### J. Live pressure (8)

All of these read signals Carbonite already collects and Kyber cannot see (H6).

| # | Proposal | Metadata | Home |
|---|---|---|---|
| J1 | Reduce the spill budget under PSI memory pressure at plan time | `cgroup_pressure` | `cost/terms` |
| J2 | Cap DOP when the container is CFS-throttled, which nominal core count cannot express | `cgroup_throttled_ratio` | `annotate` |
| J3 | Admit a broadcast only if it fits in memory *now*, not in nominal capacity | `available_bytes` | SELECTION |
| J4 | Size against live GPU memory rather than nominal VRAM | NVML `device_telemetry` | `gpu/sizing` |
| J5 | Share one envelope across concurrent queries in a session rather than planning each as if alone | new: session registry | `annotate` |
| J6 | Treat a thermally throttled machine as a temporarily different speed class | new probe | `calibration` |
| J7 | Choose shuffle against broadcast on live fabric utilization | `fabric_usage` | SELECTION |
| J8 | Price spill against live disk queue depth | new probe | `storage_cost` |

### K. Distributed and shuffle (12)

Lettered K rather than H so an item id can never be confused with an audit finding above.

| # | Proposal | Metadata | Home |
|---|---|---|---|
| K1 | Use the locality cost per candidate order inside the enumerator, not only as a plan-level factor | `cost/locality` | JOIN_REORDER |
| K2 | Broadcast vs shuffle by topology (NVLink domain, rack) rather than bytes alone | `ClusterShape` | SELECTION |
| K3 | Aggregation-tree depth from measured combine cost rather than a fixed fan-in | `t_op_ms` | `dist` |
| K4 | Skew splitting from measured partition sizes, on by default (the scorecard records the machinery as existing and disabled) | `dist/skew` | `dist` |
| K5 | Straggler-aware DOP from per-worker time variance, which the per-worker feedback rows already carry | `t_op_ms` per worker | `annotate` |
| K6 | Per-node-class plans on a heterogeneous fleet, instead of one plan sized to the weakest node | `cluster_hardware_profiles` (dropped, H3) | `dist`, `plan/resource` |
| K7 | Shuffle compression choice from measured fabric throughput | `fabric`, `io_stats` | Carbonite hand-off |
| K8 | Reuse an existing co-partitioning across queries in a session | `properties` distribution | `plan_cache` |
| K9 | Predict shuffle bytes from measured `result_bytes` instead of estimated widths | `result_bytes` (dropped) | `cost/shuffle` |
| K10 | Price spot-preemption risk as a cost term, since the migration machinery already exists | `carbonite/resilience` | `cost` |
| K11 | Per-rail exchange cost, which `gpu/exchange.py` computes and the relational path does not use | `fabric/rails` | `cost/fabric` |
| K12 | Per-worker-class `num_cpus`, which the H4 fix unblocks | H4 fix | `cpu_shares` |

## Part 4: triage — a verdict on every proposal

Written **before** implementing anything, and it changed the plan substantially. Of 124
proposals, 9 were already landed, 4 are defects worth fixing, 14 are worth building, 44 are
deferred pending a measurement or a prerequisite, and 53 are rejected. A catalog assembled by
asking "what could the metadata support" produces a lot of things that sound good and should
not be built, and saying which is the point of this section.

Four rejection reasons recur, and each is a rule this repository already states:

* **Already landed.** The catalog re-proposed nine mechanisms that exist. See E14 above.
* **Nothing to choose between.** Kyber can only select among algorithms the data plane
  implements. `bc_ir::JoinStrategy` offers three (`Hash`, `Broadcast`, `SortMerge`) and is
  already chosen by a UCB1 bandit and a learned crossover. For aggregate, distinct, window and
  sort there is one algorithm family, and the runtime switches within it on data the planner
  cannot see (`bc-interp/src/agg_par.rs` adapts on *measured* reduction ratio mid-flight).
  Adding a Kyber-side "choice" there builds a worse decision-maker with strictly less
  information. This retires most of section D.
* **Speculative generality.** `CLAUDE.md` bans a field, flag, or abstraction with no current
  caller. Several proposals amount to carrying a probed number into a contract in the hope
  something later reads it.
* **Wrong lane.** "Core measures, Kyber decides, Carbonite protects." A proposal that has
  Kyber run a micro-benchmark, or decide a scheduling question Carbonite owns, is not a small
  version of a good idea; it is the failure mode `.claude/rules/architecture.md` exists to
  prevent.

### A. Hardware plumbing

| # | Verdict | Reason |
|---|---|---|
| A1 | **Landed** | Fixed H1: `cache_factor` takes the target node's cache. |
| A2 | **Landed** | Fixed H2: `HardwareProfile.storage_class` carries the binding worker's spill device. |
| A3 | **Landed** | Fixed H4: `op_stats_by_kind(hw_fingerprint)` buckets per machine class. |
| A4 | **Landed** | `HardwareProfile.fingerprint` carries the class the plan targets. |
| A5 | Rejected | `physical_cores` has no consumer, and "hash builds scale with physical cores" is unmeasured here. Carrying a field for a future reader is the speculative generality the contract bans. |
| A6 | Rejected | Same, plus a new NUMA cost term would need calibration that H5 shows a single new term cannot get. |
| A7 | Rejected | **Redundant.** `calibration._measured_jit_speedup` already fits `jit_speedup` from the measured `backend` tag, machine-scoped by the fingerprint. A prior scaled by SIMD width would fight a fit that already self-corrects. |
| A8 | Rejected | `memory_per_core_bytes` has no consumer. |
| A9 | Rejected | A micro-benchmark is *execution*, which a Kyber pass must never do. It could live in Core, but H5 is a documented design property rather than a defect: for **ranking** two plans a relative model is sufficient, which is exactly why the anchor preserves scale. |
| A10 | Rejected | `reset_hardware_probes()` already exists as the escape hatch, and a staleness check on `fingerprint()` (11 calls per warm query) costs more than the rare mid-session resize it would catch. |

### B. The unspent per-operator counters

| # | Verdict | Reason |
|---|---|---|
| B2 | Deferred | Reversed at the code: the evidence is per-*family* and noisy (a major fault can come from anything on the box) while the action — refusing a broadcast — is drastic. No measurement here shows it would fire correctly. |
| B3 | Rejected | Reversed at the code, wrong lane. Carbonite already clamps requested parallelism to the CPU budget and `policies/cpu_budget.py` already reads the CFS quota. |
| B4 | Rejected | Reversed at the code, no-op. `cpu_shares` suppresses only when `oversubscribed()` fires; when it does not, the low utilization is already trusted. |
| B12 | Rejected | Reversed at the code, misread the estimator. `record_arm` folds a **decayed running mean**, so one contended sample moves it by `1/n` and decays out — which is what an averaging estimator is for. |
| B1 | Deferred | The spill budget is now worker-aware (A2, and `memory_budget`); a second fault-driven adjustment on top needs evidence it is not double-counting. |
| B5, B6 | Deferred | **`io_read_bytes` is physical process IO, not logical bytes.** A page-cached read reports near zero, so it cannot substitute for a width estimate; it would report a compressed columnar source as infinitely narrow. Real signal, wrong units for the proposed use. |
| B9 | Deferred | The adaptive loop is gated at 5M rows per pipeline breaker; changing its trigger is entangled with G1. |
| B11 | Deferred | Same units problem as B5/B6, and build side is chosen by size for good reasons. |
| B7 | Rejected | Detecting hidden materialization is a diagnostic, and `observe/insights/resources.py` is where diagnostics live. Not a plan decision. |
| B8 | Rejected | Morsel size from a fault ratio is unsupported by any measurement here. |
| B10 | Rejected | The plan cache measurably costs 350 ms on a miss against 160 ms on a hit; keying it on a noisy runtime signal buys thrash. |

### C. Measured memory

| # | Verdict | Reason |
|---|---|---|
| C1 | Deferred | Subsumed: broadcast eligibility is `rows x width`, so it is answered by a correct measured width. Blocked on the H12 key fix. |
| C2, C8 | **Landed** | Built, reverted for its key, and re-landed once H12 fixed that key. `result_bytes / n_actual` per signature is the measured intermediate width `cost/model.py` asks for in a comment. It carries the `max`-only combination rule and the credibility bound it originally shipped with — both were always sound — plus the concentration gate, for the deliberate in-memory residual. |
| C3 | Deferred | With widths measured the spill prediction improves on its own; a second override needs evidence it is not double-counting. |
| C10 | **Landed** | Fan-out sized by the state a breaker holds, which neither the row nor the input-byte term can see. |
| C7 | Rejected | Two reasons, either sufficient. A breaker's *placement* is not a free choice — a sort, an aggregate and a join build are breakers because the algebra says so, not because a rule sited them. And pricing the materialization needs a measured `result_bytes` per intermediate, which is the key H12 shows is collided. |
| C4 | Rejected | The spill threshold is Carbonite's *policy*. Learning it from where spilling began learns the policy back, not a property of the data. |
| C5, C11 | Deferred | Low value, and hard to attribute to one operator. |
| C6 | Deferred | Depends on an enumerator change (E2); worth doing together or not at all. |
| C9 | Deferred | Carbonite's `LearnedMemoryModel` already fits a quantile of `m_peak_bytes`. Kyber cannot import it, so this wants that value lifted to a neutral layer first — a real change, but a structural one. |
| C12 | Rejected | Serializing two breakers is scheduling, which Carbonite owns. |

### D. Physical selection

The section the triage cut hardest, for the "nothing to choose between" reason above.

| # | Verdict | Reason |
|---|---|---|
| D12 | Rejected | Reversed at the code, already measured. `adaptive_build_side` ranks with the cost model, a learned broadcast threshold, a learned sort-merge crossover and a UCB1 bandit. |
| D17 | Rejected | Reversed at the code, misread the mechanism. The runtime filters are plan-time *superset* filters derived from statistics that sink to the scan; there is no build cost to admit against. |
| D9, D14, D15 | Deferred | Real, but these are *sizing* knobs (batch size, prefetch depth, scan fan-out), not algorithm choices, and each needs a benchmark to move safely. |
| D16 | Deferred | Needs a measured pruning yield, which nothing records yet. |
| D1 | Rejected | `bc-interp/src/agg_par.rs` already switches on the *measured* reduction ratio during the run. A plan-time guess is strictly worse information. |
| D2, D4 | Rejected | One algorithm family exists for distinct and window; there is nothing to select between. |
| D3 | Rejected | Landed as `topn_fusion`, `collapse_topn_over_topn` and `TopNBound`. |
| D5 | Rejected | The Rust hash join already applies its own probe-side bloom pre-filter internally (`rules/joins/agg_semijoin.py` says so). |
| D6 | Rejected | `properties.py` already propagates ordering, and `eliminate_sort_in_distinct_union_branch` already exploits it. |
| D7 | Rejected | Already the basis of the fitted `jit_speedup`; the tier itself must stay the engine's choice, since it falls back per batch on nulls. |
| D8 | Rejected | Morsel size is a Carbonite policy (`carbonite/policies/morsel.py`). |
| D10, D11 | Deferred | `project_common_subexpression` and the projection-pushdown family cover much of this; the remainder needs a measurement. |
| D13 | Rejected | Spill compression is Carbonite's (`policies/spill_advice.py`), and it already reads the same device table. |
| D18 | Rejected | `kyber/gpu/adaptive.py` already fits a per-kind GPU/CPU crossover from measured timings. |

### E. Join ordering

| # | Verdict | Reason |
|---|---|---|
| E14 | **Landed** | See the correction above. This was the catalog's top-ranked item and it already existed. |
| E1, E6, E12 | **Landed** | The estimator already consumes learned per-signature cardinality and measured selectivity, and `join_reorder` already ranks by cost over it. |
| E7 | **Landed** | `kyber/plan_cache.py` memoizes the whole optimized plan, which subsumes memoizing an order. |
| E8 | **Landed** | That is the stage-boundary adaptive loop in `api/adaptive/`. |
| E2 | Deferred | Bushy enumeration is a genuine gap and a genuine risk: it is the change most likely to regress TPC-H, and "no performance regressions" is an invariant. Wants a benchmark harness run, not a blind edit. |
| E3, E4 | Deferred | Robust ordering over the measured q-error distribution is the most interesting idea in the section and the least safe to land unbenchmarked. `correction.py` already has the confidence machinery it would need. |
| E5, E9, E10, E11 | Deferred | Each is partly covered (`cost/imbalance.py`, `stats/skew.py`, `split_expensive_filter`) and none has a measurement showing the remainder pays. |
| E13 | Rejected | Session-wide ordering for reuse is speculative with no caller. |

### F. Sketches

| # | Verdict | Reason |
|---|---|---|
| F6 | **Landed** | Commit `3ac2e287` added the sort range grid. |
| F11 | Rejected | The runtime already adapts partial aggregation on the measured reduction ratio. |
| F1, F2, F3, F14 | Deferred | The reservoir has no consumer and the FFI entry point is real, but sampling is *execution*: it needs a Core-side pass and a storage shape. Its value is bounded to the **first** execution of each shape, because `measured_selectivities` covers every later one — worth building, worth sizing honestly first. |
| F7, F10 | Deferred | Both need per-column **sketch persistence**, which the hub docstring lists as not yet implemented (`column_tables` stores an NDV *number*, not an HLL). Prerequisite first. |
| F4, F5 | Deferred | `dist/skew.py` already learns hot keys; the increment is the tail estimate, and it needs a cluster run to evaluate. |
| F13 | Deferred | Real and valuable, but it is an IO-layer feature (writing statistics after a first read), not a Kyber rule. |
| F8 | **Rejected** | A persisted build-side bloom whose source has since changed prunes rows that exist. The failure mode is a **wrong answer**, not a slow query, and the guard (source versioning) is the hard part. Not worth the correctness surface. |
| F9 | Rejected | The estimator already caches per optimize. |
| F12 | Rejected | Calibration uses medians deliberately, for robustness to exactly the noise a p95 would chase. |

### G. The adaptive loop

Every entry here is a substantial change to `api/adaptive/`, and the scorecard is explicit that
the loop is stage-boundary adaptation off below 5M rows per pipeline breaker.

| # | Verdict | Reason |
|---|---|---|
| G11 | **Landed** | Adaptive conjunct reordering shipped (`ConjunctOrder`); the extension to join predicates is real but unmeasured. |
| G1, G2, G3, G5, G9, G10, G12 | Deferred | Each is a real improvement and each changes when re-planning fires. Moving that trigger without a cluster benchmark is how a latency regression ships. G4 and G7 are the two that would make a stronger claim than AQE honest, and they are the largest. |
| G4, G7 | Deferred | Genuine moat work. Out of scope for a pass that must not regress; recorded as the headline item for a dedicated effort. |
| G6 | Rejected | Speculative dual execution doubles the work on the first morsels to decide something the bandit already decides across runs. |
| G8 | Rejected | Live spill adaptation is Carbonite's; it already does it. |

### I. Cross-query and workload

| # | Verdict | Reason |
|---|---|---|
| I11 | Deferred | The cardinality half already exists and is consumed (`correction.py`). The cost half needs a predicted-cost field on the physical-plan wire contract for a diagnostic with no optimizer consumer. |
| I3 | Deferred | A result cache needs a source-version contract to be safe. |
| I10 | Deferred | Drift detection is observability; worth doing after I11, which supplies the same plumbing. |
| I1, I2, I4, I5, I6 | Rejected | Advisory "recommend an index / partitioning / sort order" is a new product surface with no caller, not an optimizer rule. |
| I7, I8 | Rejected | A bandit over whole plans has an unbounded arm set; UCB1 over three join strategies works because the arm set is three. |
| I9 | Rejected | Depends on A9, which is rejected. |
| I12 | Rejected | The plan cache already throttles by refit epoch; admission by cost is a second mechanism for the same job. |

### J. Live pressure

| # | Verdict | Reason |
|---|---|---|
| J2 | Deferred | The strongest of the eight, and it belongs to **Carbonite**: `policies/cpu_budget.py` already reads the CFS quota and `cpu_oversubscription`. Kyber's `n_max_parallelism` is a hint Carbonite turns into an envelope, so the throttle check goes there, not here. |
| J7 | Deferred | `carbonite/transfer/fabric_usage.py` measures live fabric; `cost/fabric.py` uses the static rate. Bridging them crosses a subsystem boundary and needs a neutral home. |
| J1, J3, J5 | Rejected | Admission against live memory is Carbonite's job and it does it. A plan-time reading is also racy: the plan may run long after it was made. |
| J4 | Rejected | `carbonite/accel/allocator.py` already reads live device telemetry. |
| J6, J8 | Rejected | No probe exists for thermal state or disk queue depth, and inventing a plausible number is what `HardwareProfile` documents itself as never doing. |

### K. Distributed

| # | Verdict | Reason |
|---|---|---|
| K12 | **Landed** | A3 and A4 unblocked per-worker-class CPU shares. |
| K9 | Deferred | Falls out of a *correct* measured width, which C2/C8 was not. Waits on H11. |
| K4 | Deferred | The scorecard records the skew-split machinery as existing and disabled. Enabling it is a cluster-run decision, and CI runs no cluster. |
| K1, K3, K5, K6, K7, K11 | Deferred | Each needs a recorded multi-node run to evaluate, which `CLAUDE.md` requires for any `dist/` change and which this pass cannot produce. |
| K2 | Deferred | `cost/locality.py` and `gpu/exchange.py` already price topology; the relational broadcast decision could read it, and that is a real gap. |
| K8 | Deferred | Co-partition reuse needs the distribution property to survive across queries, which `properties.py` does not yet persist. |
| K10 | Rejected | Preemption risk as a cost term prices a probability the plan cannot act on; `carbonite/resilience/preemption.py` already migrates proactively. |

### Second pass: seven "build" verdicts reversed at the code

The triage above was written from the audit. Implementing it meant reading each target
properly, and seven entries did not survive that. They are recorded rather than quietly
dropped, because a reader who only sees the survivors cannot tell a considered rejection from
an oversight — and because five of the seven were rejected for the *same* reason the first pass
already named and then failed to apply to itself.

* **B3** (cap fan-out under measured preemption) — **rejected, wrong lane.** Carbonite already
  clamps the requested parallelism to the CPU budget, and `carbonite/policies/cpu_budget.py`
  already reads the CFS quota and oversubscription. Kyber lowering it too is a second mechanism
  for one job, in the layer that does not own it.
* **B4** (voluntary switches to identify an IO-bound family) — **rejected, no-op.**
  `cpu_shares` suppresses a shrink only when `oversubscribed()` fires; when it does not, the
  low utilization is *already* trusted. The proposal adds a signal to reach a state the code is
  in by default.
* **B12** (suppress a bandit update from a contended run) — **rejected, misread the
  estimator.** `record_arm` folds into a **decayed running mean** per arm. One contended sample
  moves it by `1/n` and decays out, which is precisely what an averaging estimator is for. The
  proposal described a latch; the code has a mean. Systematic contention, the real version of
  the worry, shifts every arm together and so cannot change the argmax.
* **B2** (demote a broadcast whose history paged) — **deferred.** The evidence is per-*family*
  and noisy (a major fault can come from anything on the box), while the action — refusing a
  broadcast — is drastic. No measurement here shows it would fire correctly.
* **D12** (build side by measured probe cost) — **rejected, already measured.**
  `adaptive_build_side` already ranks with the cost model, a learned broadcast threshold, a
  learned sort-merge crossover and a UCB1 strategy bandit. Its byte sizing also improves for
  free once widths are measured correctly (H11).
* **D17** (admit a runtime filter only when it pays) — **rejected, misread the mechanism.**
  The runtime filters are plan-time *superset* filters derived from statistics (a key range, a
  bloom, a mirrored `IN` list) that sink to the scan. There is no runtime build cost to
  amortize, so there is nothing to admit against.
* **I11** (predicted against measured cost) — **deferred.** The *cardinality* half already
  exists and is consumed: `correction.py` folds the q-error per signature. The *cost* half
  needs a predicted-cost field on the physical-plan wire contract for a diagnostic with no
  optimizer consumer, which is the speculative-generality line. Worth building when a rule
  needs it.
* **C1** (broadcast eligibility from a measured peak) and **K9** (shuffle volume from measured
  width) — **subsumed.** Both size in bytes as `rows x width`, so both are answered the moment
  the width is measured. No separate change.
* **C3** (a "this shape spilled" flag) — **deferred.** With widths measured, the state estimate
  the spill prediction runs on is materially better; a second override needs evidence it is not
  now double-counting.

### What this leaves

What Part 4's triage left to build, on top of the four defects the audit had already found:

* **C2/C8** — measured intermediate row width. `cost/model.py` asks for this in a comment.
  ~~It delivers C1 and K9 with it.~~ **Built and reverted**: the per-signature key it used
  cannot tell two sources apart. The underlying defect is real and is now H11, which needs a
  source ⟂ column key instead. C1 and K9 wait on that.
* **C10** — fan-out sized by the state a breaker *holds*, not only by what flows into it.

The honest summary of a 124-item catalog is that **eight** changes were worth making, and seven
of them are defects — including three (H10, H11, H12) that the catalog never proposed and that
were found only by trying to implement the ones it did. Exactly **one** catalogued optimizer
feature survived contact with the code: C10.

That ratio is the finding, not a disappointment. The optimizer's measured-metadata loops are
further along than an audit of *field consumption* suggests, because the consumption is
concentrated in a few well-built modules (`measured_selectivity`, `correction`,
`learned_tuning`, `calibration`) rather than spread across the rule set. What was broken was
never the absence of loops. It was **what indexes them** — every one of the seven defects is a
learned value filed under the wrong key: the wrong machine (H4, H10), the wrong node's hardware
(H1, H2), the wrong source (H12, and the C2/C8 revert), or no key at all because the gate was
written for a different statistic (H11).

## Part 5: what to build first

Ranked by value against effort, with the defects first because a proposal built on a broken
input inherits the break. **Items 1, 2 and 7's first half are landed** (A1, A2, A3, A4); item 3
turned out to be landed already and is corrected in place above.

1. **A3 and A4 (fix H4).** A plan ranked by coefficients from the wrong machine is the deepest
   error in the list, it is invisible, and it disables the two learning loops that matter most
   on exactly the deployment Batcher is trying to win. Small change: one optional parameter on
   `op_stats_by_kind` and one field on `HardwareProfile`.
2. **A1 (fix H1).** Three lines, and it repairs the dominant term in join ordering on every
   distributed query.
3. **E14 and F2 (joint selectivity).** The single largest source of cardinality error in any
   optimizer, and Batcher already measures the quantity that removes it. Nothing else in the
   catalog has this ratio of value to novelty risk.
4. **C1, C3, C6 (measured memory into plan choice).** `m_peak_bytes` is recorded on every
   operator of every execution and Kyber has never read it. Spill is the largest single cost
   error a plan can contain, by `terms.py`'s own account.
5. **B3, B4 (contention counters).** `cpu_shares.py` documents an ambiguity it cannot resolve
   and suppresses learning because of it. The two context-switch counters resolve it, and they
   are already in the row.
6. **D1 through D6 (fill the SELECTION phase).** The phase is three rules wide. Every entry
   here is a decision the engine is already making implicitly, made explicit so it can be
   costed, explained, and learned.
7. **A2 (fix H2), then J1 and J2.** Storage class and live pressure, in that order, because the
   static figure is wrong before the dynamic one is missing.
8. **A9 (absolute calibration anchor).** Unblocks I9 and makes calibration meaningful on
   single-family workloads, which is most ETL.

Everything in G is larger and should follow the scorecard's own framing: the adaptive loop is
stage-boundary today, and G4 or G7 is what would make a stronger claim honest. Do not restate
the retired claim before the code supports it.

## Part 6: what landed, and what it measures

Ten changes. Eight are defects, and two (C10, C2/C8) are catalogued proposals — the four the audit found, plus H10, H11 and H12, none of
which the catalog proposed and all of which surfaced only from trying to build the items it did.
One (C10) is a proposal that survived Part 4.

H12 is the one to read first: it is the only finding here that was making plans **worse than not
learning at all**, and it is in shipped code.

| Change | What it does | Pinned by |
|---|---|---|
| A1 (H1) | `cache_factor` takes the target node's L3 | `test_cost_worker_cache.py` |
| A2 (H2) | `HardwareProfile.storage_class` prices spill on the worker's volume | `test_cost_storage_awareness.py` |
| A3, A4 (H4) | `op_stats_by_kind` buckets per machine class; the planner names the class | `test_learning_scoped_to_target_machine.py` |
| C10 | Fan-out sized by the state a breaker holds | `test_parallelism_from_state_size.py` |
| H11 | Payload column widths measured, at the source ⟂ column key | `test_payload_column_widths.py` |
| H10 | Driver-side learned state keyed to the workers' class, via one ambient scope | `test_learned_state_keyed_to_target_machine.py` |
| H12a | A measured selectivity is refused when its samples are not one population | `test_selectivity_confidence_gate.py` |
| H12b | The scan token carries the source's identity, so signatures name the relation | `test_scan_signature_identity.py` |
| C2/C8 | Measured intermediate row width, re-landed on the corrected key | `test_measured_row_width.py` |

C2/C8 was built, verified on the live engine, and then **reverted** — its per-signature key
cannot distinguish two tables with the same query shape, so it applied one source's width to
another. The revert is byte-exact. The defect it was aimed at is real, and chasing it properly
produced **H11**: the same symptom, at the key a width actually belongs to (source ⟂ column),
with the sketch-marker trap handled. That sequence — build, verify, discover the key is wrong,
revert, re-derive — is the most useful thing in this document, and it is why the catalog's own
confidence in an item is worth very little before someone tries to implement it.

### The width measurement, on the live path (reverted — see H11)

Run against the real engine, not a constructed hub. A `filter` over a 4 KiB payload column,
four executions, reading the estimator between them:

```
width before any run : 44.0 B/row
width after 4 runs   : 4108.0 B/row
true payload width   : 4096 B/row (plus the int64 key)
```

A **93x under-estimate**, corrected to within 0.1% of truth. That is the failure
`_desired_parallelism` already documents in a table — a 768-dim embedding costed at the flat
64 B assumption sizes 12 GB tasks, and a 1080p frame sizes 25 TB ones — now measured instead of
assumed. C10 is what turns the corrected width into a fan-out.

The same run on **fixed-width numeric** columns learns `16.0` B/row for a two-`int64` output,
which is exactly what the type priors already say.

That looked like the property that bounded its risk, and it was not. What it actually showed is
that the mechanism reads the right *quantity*; the key it files it under is the unsound part,
which only a second table with the same query shape reveals. See H11.

### Which changes a benchmark can even reach

**Corrected.** An earlier version of this section said "every surviving change is inert on a
single node by construction". That was true of the four changes that had shipped when it was
written and is **false** of the ten that shipped in the end: H11, H12a, H12b and C2/C8 all
change single-node planning, and three of them do so on every query. Splitting the set:

| Change | Single node | Why |
|---|---|---|
| A1 | inert | `cache_bytes=None` probes this machine, as before |
| A2 | inert | `storage_class=""` resolves this process's spill volume, as before |
| A3, A4 | inert | the local fingerprint *is* the target class; the view is the same rows |
| C10 | inert | the task budget is the whole machine, so the state term never binds |
| H11 | **active** | payload column widths are learned on every resident scan |
| H12b | **active** | the scan token changes for every file-backed source |
| H12a | **active** | a wide-spread selectivity is refused rather than applied |
| C2/C8 | **active** | a measured intermediate width raises `row_width` after 3 runs |

C10's inertness was measured rather than assumed:

```
single-node task memory budget: 27.0 GiB
fan-out with local profile : 1
fan-out with no profile    : 1
C10 changes single-node fan-out: False
```

An A/B was written and armed to run when the machine went quiet, then **stopped**: on a single
node it would compare a change against itself and report a null result, at the cost of hours on
a contended box. Reasoning about which paths a change can reach is better evidence than timing
paths it provably does not.

A **recorded cluster run** is still owed for the four inert-single-node changes: they alter
distributed planning only, `CLAUDE.md` requires a recorded multi-node run for exactly that, and
CI installs no Ray. For those four the correctness argument is the tests plus the design, and
the performance argument is unmade.

### The benchmark, finally run

Once the machine went quiet (1-minute load 4.7, against 33 for most of this work — TPC-H q1
runs at 50 ms here against 366 ms under contention), the four single-node-active changes were
A/B'd against themselves: same tree, same engine, same `benchmarks/` code, the four disabled by
monkeypatch in the "off" arm. Three interleaved off/on rounds, TPC-H sf1, medians of the
per-round best-of-5:

```
queries compared : 22        (correctness clean in all 6 runs)
median delta     : -0.9%
total off/on     : 1027 ms / 1005 ms  (-2.2%)
```

**No regression.** And the per-query deltas are not readable: within-arm spread on every query
that moved more than 8% exceeds the between-arm difference — `q15` alone ranges 4.3 to 11.0 ms
*inside* the "off" arm, `q20`'s two arms span overlapping 20-27 and 23-32 ms bands. The -2.2%
total is noise in the favorable direction, not a speedup, and must not be quoted as one.

**TPC-H also cannot show the benefit, and that is expected.** Its schema is fixed-width columns
and strings — exactly where a measured width equals the type prior, verified directly: a
two-`int64` output learns `16.0` B/row against a structural `16.0`. The changes act where the
priors are wrong, which is wide and multimodal payloads, and TPC-H has none. So this run answers
"did it regress" (no) and is structurally unable to answer "did it help". The 44 B/row to
4,104 B/row correction on a 4 KiB payload column is the evidence for the second question, and it
is a plan-level measurement rather than a wall clock.

### On the wall-clock benchmark that was attempted earlier, and discarded

A TPC-H sf1 run against this build passed every correctness check. It is **not** reported as a
performance result, and the attempted before/after comparison is discarded rather than quoted,
for two reasons that are worth recording so nobody repeats them:

1. The baseline arm was built by pointing `PYTHONPATH` at a `HEAD` checkout of `python/` while
   using the **working tree's** `benchmarks/`. Another session had changed the SQL translator,
   so twelve of twenty-two queries failed with `AttributeError: '_Translator' object has no
   attribute '_star_sources'`. A baseline that errors on half its queries is not a baseline.
2. The machine was running a 14,000-test suite and another session's refactor throughout. The
   surviving queries moved by up to 5x *between arms that differ by a width estimate that
   provably does not change on TPC-H's schema*, which measures contention, not the change.

The direct plan-level measurement above is the evidence instead: it isolates the quantity the
change actually moves. A wall-clock comparison should be re-run on a quiet machine, with both
arms built from one `benchmarks/` tree, before any performance claim is attached to this work.

## Part 7: what remains open, and the one thing that unblocks most of it

Every one of the 124 proposals now carries a verdict, and every deferral names the prerequisite
that blocks it — there is nothing left in limbo. The dispositions:

| Disposition | Count |
|---|---|
| Landed — 9 already were before this pass, 7 were built in it (A1–A4, C10, C2/C8) | 16 |
| Rejected, with a recorded reason | 50 |
| Deferred, with a named prerequisite | 58 |
| **Total** | **124** |

Four of the ten changes this pass shipped — **H10, H11, H12a, H12b** — are not in that table at
all, because the catalog never proposed them. They were found by trying to build the items it did.
That is the single most useful thing to know about this document. Its hit rate on optimizer
*features* was two of 124 (C10, and C2/C8 only after a defect the catalog never saw was fixed),
and its value came overwhelmingly from the defects that surfaced while testing its proposals.

The deferrals clustered onto four prerequisites. One has since been removed; the remaining
three are worth naming because none of them is optimizer work at all:

1. ~~**A correct plan signature**~~ — **removed.** This was the top blocker, on C2/C8, C7, K9
   and parts of E and F. The scan token now carries the source's data-stable identity (H12), so
   a learned value keyed by plan signature belongs to the relation it was measured from. The
   feared cost — threading a parameter through 22 call sites — turned out not to be the fix at
   all: the identity belongs on the `Scan` node, and every call site then gets it for free.

   **C2/C8 (measured intermediate row width) was the item this unblocked, and it has since
   re-landed.** Built, verified at a 93x correction, reverted purely because its per-signature
   key could not tell two sources apart, and restored once that key was correct — with the
   `max`-only rule and credibility bound it always had, plus the concentration gate for the
   in-memory residual. The case that forced the revert is now its first test: four runs of a
   4 KiB-payload table leave an unrelated 16-byte table's width at 16.
2. **A recorded cluster run** (blocks most of K, plus K4's skew splitting and the
   straggler/placement work). CI installs no Ray, and `CLAUDE.md` requires a recorded multi-node
   run for any `dist/` behavior change. Nothing here can substitute for it.
3. **A quiet machine and a benchmark harness** (blocks E2's bushy enumeration, E3/E4's robust
   ordering, D9/D14/D15's sizing knobs). Each is a plausible win and each is exactly the kind of
   change that regresses TPC-H if landed unmeasured.
4. **Sketch persistence** (blocks F7, F10, F1's reservoir). `column_tables` stores an NDV
   *number*, not an HLL; the hub's own docstring lists sketch persistence as not yet built.

The remaining deferrals sit in `api/adaptive/` (section G), and the scorecard already frames
those honestly: the loop is stage-boundary adaptation off below 5M rows per pipeline breaker, and G4 or G7 is what
would make a stronger claim than AQE true. That is a project, not a backlog item.

## Verification notes

The two demonstrations in Part 1 are reproducible from the repository root with
`PYTHONPATH=python`. H4's script constructs a hub, records 500 operator rows stamped with a
foreign hardware fingerprint, and reads back `op_stats_by_kind()` and
`cpu_shares.load_cpu_utilization()`. H5's script records two operator families whose measured
times differ from the shipped ratio and reads back the fitted coefficients. Neither needs Ray,
a GPU, or a built engine beyond the one in the tree.

None of the proposals in Part 3 has been implemented or benchmarked. Any that lands must carry
a differential test against DuckDB per `.claude/rules/testing.md`, and any that touches the
distributed path needs a recorded cluster run per `CLAUDE.md`, since CI executes neither.
