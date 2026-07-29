# Decision-quality ledger — Kyber, Carbonite, and the contracts they share

A running record of improvements that make Batcher's **decisions** right across the two
dimensions it claims to span:

* **Every data type** — structured (fixed-width columns, strings), semi-structured (JSON,
  maps, unions, nested lists), unstructured (text and binary blobs), and multimodal
  (images, audio, video, embeddings, tensors). A decision tuned on `int64` columns is
  routinely off by three or four orders of magnitude on a decoded image column, and the
  failure is silent: the plan is produced, it runs, and it OOMs or spills for reasons
  nothing in the plan explains.
* **Every scale** — a sub-second single-node query, a GB working set, a TB table, a PB
  table, and a fleet of tens of thousands of nodes. A model with no term that grows with
  the fleet ranks a three-shuffle plan and a one-shuffle plan as a factor of three apart
  when at fleet scale the real ratio is far worse.

Entries are numbered `D<n>` continuously and never reused, so the count is a count of
*distinct* improvements. Category tags: **bug** (a decision that is wrong, not merely
imprecise), **fidelity** (an estimate that was off by orders of magnitude on a real data
shape), **scale** (a term that was missing or mis-shaped at cluster/PB scale),
**robustness**, **perf**, **test** (coverage that pins a contract), **hygiene**, **docs**.

This page is not published (`exclude_patterns` in `docs/conf.py`); it is a working index
for anyone touching the optimizer or the resource manager.

---

## Column widths — the number under every byte-valued decision (`plan/types/widths.py`)

Every byte axis in the engine bottoms out here: the memory envelope Carbonite admits
against, broadcast eligibility, spill volume, morsel sizing, and the `io`/`net` cost axes.
A width that is wrong by orders of magnitude makes all of them wrong at once, and the
Arrow types it was wrong on are exactly the ones that carry unstructured and multimodal
payloads.

| # | Cat | Improvement |
|---|-----|-------------|
| D1 | fidelity | **An extension type was sized at the 32-byte variable-length prior, and every multimodal column in Batcher is an extension type.** Decoded images, audio waveforms, video frame stacks, embeddings, and model outputs are all produced as the canonical `arrow.fixed_shape_tensor` type (`io/formats/ml/tensor.py`, `ml/decode/media.py`, `core/udf/call.py`), and none of the `pa.types.is_*` predicates see through an extension label — so a 224x224x3 `uint8` image was costed at **32 B/row against a true 150,528**, a 4,704x under-estimate. It ran in the one direction a bound must not fail: a memory envelope too small to hold the operator, and a build side that looks broadcastable when replicating it would OOM every worker. Extension types now unwrap to their storage type, which makes the tensor case *exact* (a fixed-shape tensor's storage is a `fixed_size_list` whose length is in the type). |
| D2 | fidelity | A `map` matched none of the list predicates and fell through to the same flat scalar prior. Maps are how every semi-structured source lands in Arrow — JSON objects with open-ended keys, Parquet `MAP` groups, Avro maps — so a `map<string, string>` of eight entries was sized at 32 B/row against a true ~600. Now sized as the list of key/value entries it is, scaled by both element types. |
| D3 | fidelity | Arrow's `null` type is pure metadata with no value buffer, and was charged the 32-byte prior for a column that occupies none. That inflated the width of every relation carrying a not-yet-typed column — the shape a JSON or CSV source produces for a field it saw only nulls in, and a schema-evolution placeholder. Now `0.0`. |
| D4 | fidelity | A run-end-encoded column fell through to the prior. Now bounded at one run per row (`run_end_type + value_type`) — an honest worst case rather than an invented compression ratio, and still well below the prior it used to hit. |
| D5 | fidelity | Union types fell through to the prior, which under-reads a *sparse* union by roughly its arity — the shape an Avro union or a polymorphic JSON field takes. Now sized by layout: a sparse union allocates every variant per row, a dense union only the chosen one plus a type code and an offset. |
| D6 | fidelity | `string_view` / `binary_view` were sized as though they carried an offset pair. A view type carries a 16-byte view struct that inlines short values instead, so the per-row bookkeeping is four times what was charged. |
| D7 | fidelity | `list_view` / `large_list_view` matched no list predicate and were sized as opaque scalars, ignoring their element type entirely — the same class of error D2 fixes. Now recognized as list types, and charged for the offset **and** size buffers a view list carries. |

## The `net` cost axis — what a plan costs to move (`kyber/cost/shuffle.py`)

`Cost.net` and `CostWeights.net` (2.0 — a shuffled byte costs twice a local one) existed
from the day the cost model did, and **nothing ever wrote to them**. Every plan Kyber has
ranked was ranked as though the network were free. On one machine that is exactly right;
on a cluster it is the term that decides the plan.

| # | Cat | Improvement |
|---|-----|-------------|
| D8 | scale | **The `net` axis is populated.** Shuffle volume, fan-out, and the co-partitioning that avoids both are now visible to join ordering, join strategy, and every cost-based rule. Charged only when `HardwareProfile.worker_count > 1`, so a single-node plan is ranked bit-for-bit as it was before (pinned by `test_single_node_charges_no_net_anywhere`). |
| D9 | scale | Shuffle volume is discounted by the `1/W` share that hashes to the bucket already on its own node. At two workers half a shuffle is free and at ten thousand essentially none of it is — the honest shape, and the reason a shuffle that is nearly free on a four-node cluster is not on a large one. |
| D10 | scale | **The fan-out term: a shuffle between `P` producers and `R` reducers opens `P x R` fragments.** At `P = R = 10,000` that is a hundred million fragments to open, frame, and drain before a useful byte moves, and it is the term that actually stops a shuffle-heavy plan from reaching fleet scale. Without it the model says three shuffle stages cost three times one stage; the real ratio at scale is far worse. It is also why a broadcast join is not merely "a cheaper shuffle" — it is `W` fragments against `W^2`, so its advantage grows with the *square* of the cluster. |
| D11 | scale | An aggregate is charged its **partial** volume, not its input. The mergeable `partial -> combine` form means each worker reduces locally, so what crosses the wire is bounded by `min(input, groups x workers)`. A two-group aggregate over a trillion rows shuffles two rows per worker; costing it at the input volume over-charges it by the whole relation and would make the optimizer avoid the cheapest stage in the plan. |
| D12 | scale | An aggregate or distinct whose input already delivers a hash partitioning on a subset of its keys is charged **zero** net. This is the property `dist` already relies on (a hash join leaves its output partitioned by the join key, so a following `group_by` on that key computes complete groups locally) — now visible to the cost model, so the enumerator prefers the plan shape it should. |
| D13 | scale | A join is charged the **minimum** of its two strategies — replicate the build side (`W-1` copies, `W` fragments) or repartition both sides (`W^2` fragments) — because that is the one the engine will pick (`rules/selection.py`). Ranking a join order by a strategy the physical plan will not use ranks it by a cost nobody pays, the same error `join_op_cost` already corrects for build-side orientation. |
| D14 | scale | A join whose two inputs are already co-partitioned on its keys is charged zero net. |
| D15 | scale | A **top-N** gathers only `workers x k` rows, not the relation: each worker ranks locally and forwards its own `k`. The reason to fuse a limit into a sort now survives into the network axis, where the difference between the two is unbounded. |
| D16 | scale | An **unpartitioned** window is charged a `W -> 1` gather of its whole input, because its frame spans the relation and every row must reach one worker. That is the operator that will not scale, and costing it at zero let the optimizer treat a global `row_number()` as free. |

## Structure and shared rules

| # | Cat | Improvement |
|---|-----|-------------|
| D17 | hygiene | `kyber/cost.py` stood at 494 lines against a 500-line limit, which blocked any further cost work. Package-ized into `kyber/cost/` (`model`, `terms`, `shuffle`) with the public surface re-exported unchanged — proved by an empty `surface-diff`. It also takes `python/batcher/kyber/` from 19 top-level modules to 18 against its allowlisted debt entry, since subdirectories do not count. |
| D18 | hygiene | The machine-shaped multipliers (cache residency, spill volume, external-merge passes, sort comparison counts) were private methods on `CostModel`, so the four operators that all build a hash table — `Join`, `Aggregate`, `Distinct`, distinct `Union` — depended on them through the class rather than on a stated rule. Lifted into `cost/terms.py` as one definition each. |
| D19 | hygiene | A sort's external-merge IO was open-coded arithmetic at its one call site (`passes x state x device factor`), so it could drift from the flat `spill_io` term the hash operators use. Named as `terms.merge_io`. |
| D20 | robustness | `texput.log`, a stray pdfTeX log tracked at the repository root, failed `lint-structure` — which runs over the whole tree, so it blocked the pre-commit hook for **every** session, not just the one that created it. Removed. |

## The memory envelope, and the plan shape that decides it (`kyber/annotate.py`, `carbonite/memory`)

The envelope is not advisory: admission checks feasibility against it, the spill decision
reads it, and the distributed per-task memory grant is derived from it. Two errors in it
were large and pointed in opposite directions.

| # | Cat | Improvement |
|---|-----|-------------|
| D21 | bug | **A hash join was budgeted at its *output* rows, where its resident state is the hash table over its *build* side.** In a star schema those differ by the fan-out ratio: measured on a 100,000-row fact joined to a 100-row dimension, `cost.py` sizes the table at 1,600 bytes and `annotate_ops` handed Carbonite **2,400,000** — 1,500x, on the most common join shape in analytics. So the query was pushed toward spilling and toward a rejected admission for a hash table that fits in a page. The two subsystems now agree by construction (`test_the_join_envelope_agrees_with_the_cost_model`), which matters beyond the arithmetic: a plan *ranked* on one number and *admitted* against another is the Kyber/Carbonite loop coming apart. |
| D22 | bug | A fused `Sort` + `Limit` was budgeted at the whole relation rather than its `limit`-row heap — the same error one level down, and the entire reason to fuse the two. |
| D23 | scale | **`PhysicalOp.inputs` was hardcoded empty**, and `carbonite/memory/estimator.py` named the consequence in its own docstring: with no tree, a plan's envelope can only be its largest single breaker. A bushy plan holds several at once (a join's build side stays resident while the probe side runs), and on a four-way bushy join with tables at 18.2 / 9.1 / 9.1 MB the largest-single reading is 18.2 where the concurrent one is 27.4 — a 1.5x under-count, in the direction that over-admits and OOMs. That docstring ended "Populating `inputs` is Kyber's to do"; `annotate_ops` now does it. |
| D24 | scale | With the tree in hand, `peak_operator_bytes` walks the *schedule* instead of taking a `max`: `peak(join) = max(peak(build), resident(join) + peak(probe))`, `peak(unary) = max(peak(input), resident(node))`. A linear plan is byte-for-byte what it was (a unary breaker's input has finished and released by the time its own state is full), and a plan carrying no `inputs` — every hand-built `PhysicalPlan` and test double — falls back to exactly the previous reading. |

## Morsel sizing for wide rows (`carbonite/policies/morsel.py`)

A morsel is the unit of the streaming working set. Its byte budget was unenforceable on
precisely the columns that need it.

| # | Cat | Improvement |
|---|-----|-------------|
| D25 | bug | **`MIN_MORSEL_ROWS` (1,024) silently overrode the byte budget it was meant to accompany.** The floor is a narrow-row concern — per-batch overhead — and applied unconditionally it puts the crossover at `morsel_bytes / 1024` = **1,024 bytes per row**, below essentially every unstructured or multimodal column: a 768-dim `float32` embedding overshoots 3x, a 224x224x3 image **147x**, and one 1080p RGB frame **6,000x** (6 GiB against a 1 MiB budget), multiplied again by the in-flight morsel count. The floor is now itself bounded by the byte target, falling to a single row for a value larger than the whole budget — which is the only morsel that exists for it. On narrow rows it is `MIN_MORSEL_ROWS` exactly as before. |
| D26 | bug | The same inversion was pinned *green* by a test asserting `rows == max(1024, expected)` while its own comment said "cap rows so rows*width stays within morsel_bytes" — so at a 4 KiB width it accepted a 4 MiB morsel against a 1 MiB budget. The assertion now states the property the comment always claimed. |
| D27 | fidelity | The width cap was read only from the **learned** memory model, which is empty on a cold store — and the first run of a multimodal pipeline is the one with nothing measured and the one that OOMs. The width was knowable the whole time: it is a property of the column types the plan already carries, and it is *exact* for the tensor columns that hold decoded images, audio, and video. A tensor-column plan is now cut to 6 rows per morsel before a single row is read, where it previously used 16,384. |
| D28 | fidelity | The learned and planned widths are combined by taking the **more binding** cap rather than preferring either. They cover each other's blind spots: the learned width is filed by operator *family* and cannot tell this plan's image scan from an earlier narrow one, while the planned width cannot price a variable-length payload beyond its prior. Taking the smaller is safe in both directions — a morsel only batches data, so an over-tight one costs throughput and an over-loose one costs the process. |

## Cardinality for nested and semi-structured shapes (`kyber/stats/estimator.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| D29 | fidelity | **An `Unnest` was estimated at a 1x fan-out on every cold run**, on the stated grounds that average list length is a property of the data. True of a variable-length list; false of a `fixed_size_list`, whose length is in the type — the embedding and fixed-shape-vector column of every AI pipeline, and the same fact `column_bytes` already reads to size the column's bytes. Exploding a `fixed_size_list<float32, 768>` now estimates 768 rows per input row instead of 1, pinned against what the engine actually emits. Variable-length lists still fall to the learning loop, unchanged. |
| D30 | fidelity | The exact fan-out carries its input's provenance rather than `DEFAULT`. Provenance is the marker admission reads to tell an estimate from a placeholder, so a fan-out read off a type — a proof, not a guess — must not be filed as one. |
| D31 | fidelity | The fan-out unwraps an extension type to its storage, so a decoded tensor column explodes by its flattened length. Same blind spot as D1: no `pa.types.is_*` predicate sees through an extension label, and every multimodal column in Batcher wears one. |

## Text-pattern selectivity — the filter of unstructured data (`kyber/stats/selectivity/patterns.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| D32 | fidelity | **Every regex got the flat substring prior**, which is right for `'error'` and wrong for the anchored patterns Batcher's own public API generates. `is_alpha`, `is_numeric`, `is_alnum`, `is_space`, `is_url`, and `is_email` all lower to an anchored `regexp_matches` (`'^[A-Za-z]+$'` and friends), and none of them scans for a substring — they classify the *whole* value. `regexp_matches` now gets the same pattern reading `LIKE` always had: exact / anchored / substring. |
| D33 | fidelity | A fully literal, doubly anchored regex (`'^foo$'`) is equality and gets the equality estimate — a measured skew frequency where one exists, else `1/ndv`, which is typically orders of magnitude below any pattern prior. |
| D34 | robustness | `'[0-9]+\$'` is a search for a price anywhere in the text, not a whole-value match. An escaped `\$` is no longer read as an end anchor, and a bare `'^'` (which constrains nothing) no longer invents selectivity out of a pattern that has none. |
| D35 | hygiene | `starts_with`, `ends_with`, anchored `LIKE`, and anchored regex are one shape read from four call sites. Named once as `anchored_selectivity`, so they cannot drift apart. |
| D36 | docs | **A flagged inconsistency, deliberately not "fixed".** An anchored match is a strict subset of the floating one (`'foo%'` implies `'%foo%'`), so `P(anchored) <= P(substring)` holds by construction — and the shipped defaults assert the reverse (0.10 against 0.05). The one-line clamp that restores the containment was tried, and it makes the *absolute* error worse on the query that exercises it: TPC-H Q14's `p_type LIKE 'PROMO%'` really keeps about 20% of `part`, so 0.10 is a 2x under-estimate and the clamped 0.05 is a 4x one. Which prior is mis-tuned is a question for `benchmarks/run.py`, not for a containment argument. Recorded in `anchored_selectivity`'s docstring so it is visible rather than silently inherited. |

## Distributed fan-out for wide rows (`kyber/annotate.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| D37 | scale | **A breaker's desired parallelism was `rows / target_rows_per_task` with no width term.** At the shipped four million rows and the flat 64 B/row that target was tuned against, a task holds a sensible 256 MiB — and on anything wider it sizes tasks that cannot exist: 12 GB for a 768-dim embedding column, **602 GB** for a decoded 224x224x3 image, 25 TB for 1080p frames. So a multimodal pipeline was fanned out as though every row were sixteen bytes and each task was asked to hold hundreds of gigabytes. `n_max_parallelism` now takes the larger of the row- and byte-derived counts against `optimizer.target_bytes_per_task`. |
| D38 | bug | That is not a new rule — it is the rule the rest of the engine already follows. `api/tuning/decisions.py::auto_num_partitions` and `dist/executors/map.py` both take `max(row_parts, byte_parts)`, and `docs/deep-dives/distributed-scheduling.md` documents it as how a stage is sized, naming video frames and embeddings as the reason. Kyber's `n_max_parallelism` — which is what `SchedulingEnvelope.n_tasks` is actually derived from — was the one place still counting only rows, so the two answers to "how many tasks" disagreed on exactly the data the documented one was written for. Pinned by `test_kyber_agrees_with_the_partition_sizer_it_documents`. |
| D39 | docs | `_MAX_TASK_FANOUT` (100,000) was justified as "far above any real fan-out", which was true while fan-out counted rows. A petabyte of 1080p frames at 256 MiB per task legitimately wants about a million, so hitting the cap is now a signal about the plan rather than only about the estimate. Recorded in the constant's comment rather than raised, because what the ceiling should be on a real fleet is a measurement. |
