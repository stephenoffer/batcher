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

## The inference stage — the operator both halves of the loop could not see (`kyber/gpu`, `kyber/annotate.py`, `kyber/stats/estimator.py`)

`map_batches` is the centre of every AI pipeline and the widest data in the engine flows
through it. It was invisible to the width estimate, to the memory envelope, and to the GPU
batch seed at once, so a stage really holding gigabytes reported one megabyte.

| # | Cat | Improvement |
|---|-----|-------------|
| D40 | fidelity | **A `map_batches`'s row width fell back to the flat 64 B/row constant.** It is executed in Python and never lowered, so it publishes no output schema and `row_width` had nothing to read. The rows crossing it are the widest in the engine — a decoded 224x224x3 image is 147 KiB — so the estimate was off by three orders of magnitude for the memory envelope, the morsel cap, and the GPU batch seed simultaneously. It now prices its **input's** width when `output_columns` is omitted, which is not a new guess: `MapBatches.available_columns` already implements exactly that contract ("if omitted, the input columns are assumed to pass through"), so this is the existing assumption costed rather than counted. A declared `output_columns` keeps the flat default, because there the shape genuinely changed and the plan knows nothing about the columns the UDF invented. |
| D41 | robustness | Fixed as a *width*, deliberately **not** by giving `MapBatches` an `available_schema`. That method feeds type inference and expression validation, where asserting the input's types survive a UDF free to rewrite them converts an estimate into a wrong answer. A width only feeds cost and memory, where being closer is strictly better. Pinned by `test_the_width_is_not_asserted_as_a_schema`. |
| D42 | bug | **A `map_batches` carrying an explicit `batch_size` was budgeted at one morsel.** It re-batches its input to exactly that many rows regardless of the morsel it was handed, so the morsel byte cap does not bound it — and `kyber/gpu/sizing.py` seeds batch sizes of tens of thousands of rows from VRAM headroom. On an image column that is gigabytes reported to Carbonite as one megabyte, an under-count in the direction that admits a query the node cannot run. Measured end to end: an 8,192-row stage over a 224x224x3 column now reports 1.23 GB where it reported 1 MB. Only the *input* batch is charged; a UDF's output is arbitrary Python and claiming a multiple would be a fabricated number inside a memory bound. |
| D43 | bug | **The GPU batch-size seed charged activations and treated the input tensor as free.** A batch occupies the device twice over, and only the activation prior (a flat 64 KiB/row, described as suiting "typical vision/embedding activations") was budgeted. A decoded image row is 147 KiB *before* a single activation and a 1080p RGB frame is 5.9 MiB, so the seeded batch demanded far more VRAM than the device has: measured on a 24 GB device, a 1080p stage's inputs alone came to ~200 GB. Now `activation + input_row_bytes`, taking the seed from 32,043 rows to 334 for that stage and from 32,043 to 9,719 for images. Charged at the **Arrow** width — a model that upcasts `uint8` to `float32` on device occupies four times this, but that is a property of the user's model rather than of the plan, and inventing a multiplier would put a fabricated number inside a memory bound. |
| D44 | test | The default (`input_row_bytes=0.0`) reproduces the activation-only budget byte for byte, so every caller without an estimator is unchanged; a narrow numeric row moves the seed by under 1%. |

## The width a source already measured (`plan/source_stats.py`, `kyber/stats/estimator.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| D45 | fidelity | **`SourceStatistics.byte_size` was read by three consumers and not by the one that needed it most.** The storage shortcut, the read-cost predictor, and the distributed map sizer all read it; `row_width` — the single number under every byte axis in the engine — did not. `io/formats/multimodal/media.py` reports the exact total size and file count from its listing, so a directory of 200 MB videos is a *measured* 200 MB per row, while `column_bytes` could only offer its 36-byte prior for the `binary` column those bytes land in. Six orders of magnitude, for a figure already in hand. `row_width` now takes it as a floor at a `Scan`. |
| D46 | perf | **The obvious version of D45 was a blocking regression, and the benchmark is what said so.** Taking *any* source's `byte_size` as a floor moved TPC-H sf1's type-derived width from 88 to 142 B/row — closer to the true 139 — and made the benchmark **worse**: dimension build sides crossed the broadcast threshold and q9 went **55.8 ms to 127.9** (0.84x to 1.60x against DuckDB) with ten other queries slower. A sharper estimate against a threshold tuned for the blunter one is a re-tuning, not an improvement. So the floor is gated on a new `SourceStatistics.content_byte_size`, which a connector sets when `byte_size` measures the rows' own **content** (a media/text listing: one row is one file) rather than their stored encoding (a Parquet footer's row-group-padded `total_byte_size`). Re-measured with the gate: q9 back to 53.9 ms and every query at or below its baseline. Re-tuning `broadcast_max_bytes` against a sharper width is a separate, benchmark-driven change. |
| D47 | fidelity | `content_byte_size=True` set on the media source (one row per file) and on the text source in both modes (a row is a whole file, or a line whose bytes the total covers). |
| D48 | bug | **Three `SourceStatistics` qualifiers were dropped by the persisted round trip.** `content_byte_size`, `bounds_include_nan`, and `row_group_count` were encoded by neither `_encode` nor `_decode`, so a cached statistic reloaded with each at its conservative default. None produces a wrong result — which is what makes it hard to notice: a media corpus silently reverts to the 36-byte type prior on every reload, a float `max()` that could be answered from bounds goes back to executing, and a prune loses the ability to report what it skipped. |

## Footer statistics for nested columns (`io/stats/columnar_footer.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| D49 | bug | **A struct field's footer bounds were filed under its bare field name, so they merged into any top-level column sharing it.** Parquet stores one column chunk per *leaf*: `ParquetSchema.names` reports each leaf's bare name while `path_in_schema` reports its dotted path, and the Python accumulator keyed by the former. For a flat table the two are the same string — which is why this held — and for a nested one they are not. Measured on a table with a top-level `a` of 1..3 beside a struct `s{a}` of 1000..3000, the Python path reported `a` in **[1, 3000]**. Numeric footer min/max carries `Provenance.EXACT`, and that provenance is exactly what lets Kyber answer `max(a)` from metadata *without reading the data* — so this is a wrong answer, not a loose bound. Now keyed by `path_in_schema`; a flat schema is byte-for-byte unchanged. |
| D50 | bug | The **two implementations of one statistic disagreed**, which is the failure the module's own docstring says routing both through `_finalize_columns` prevents — but that shared finalization covers provenance and NaN handling, not naming. The native Rust walk keys by the path and reported [1, 3] for the same file the Python walk reported [1, 3000] for. The Python path is the *fallback*, reached whenever the native reader declines (an fsspec backend, a read-through byte cache, a declared `sorting_columns`, or any native failure), so it must be correct on its own. Pinned by `test_the_two_footer_paths_agree`. |
| D51 | feature | Nested leaves now reach `SourceStatistics.columns` under their real paths (`s.a`, `l.list.element`) instead of colliding or being lost. **On the Python path only, and that qualifier matters**: the native Rust walk is flat-only *by design* — `bc-io/src/footer_stats.rs::parquet_column_index` requires a single-part leaf path and skips anything nested — and it is the path a local or object-store read normally takes. So the correct attribution is the whole of D49's fix, while nested statistics as a *capability* still need the Rust side. A consumer for them (a nested-column zone-map prune, the obvious next use) is therefore **not** built here: it would be dead for most reads, which is the speculative-generality trap the contract names. |

## Keeping the envelope's *consumers* honest about it (`carbonite/memory`, `carbonite/policies/admission.py`)

D24 turned the envelope from a `max` into a sum over co-resident operators. That is the
right figure, and it silently invalidated an assumption downstream.

| # | Cat | Improvement |
|---|-----|-------------|
| D52 | bug | **Admission's "is this a guess?" test read the wrong operator once the envelope became a sum.** The contract is that a plan sized from a guess may be routed out-of-core but must never be *failed*, and it was enforced by reading the provenance of the single largest operator. On a bushy plan the peak is a sum over operators alive at the same moment, and a sum of an EXACT term and a guessed one is a guess — so reading only the larger term would fail a legitimate query on the strength of the smaller. `peak_contributors` now names every operator in the winning combination and the verdict is advisory if *any* of them is `DEFAULT`. On a linear plan the contributor set is the single dominant breaker, so the rule is byte-for-byte the previous one. |
| D53 | hygiene | The peak walk returns `(bytes, contributing ids)` rather than a bare int, so "how big" and "who" come from one traversal and cannot disagree. The no-`inputs` fallback returns the same pair shape, so a hand-built `PhysicalPlan` still gets exactly the pre-tree reading. |
| D54 | bug | `binding_operator` promised "the operator whose memory estimate *is* the plan's envelope", and once the envelope became a sum that could be false: the largest single operator can sit in a branch the peak does not come from — a build subtree whose own peak lost to the concurrent reading on the other side — so the message pointed a reader at an operator that is not why their query does not fit. It now names the largest *contributor*, falling back to the largest sized operator when nothing contributes (the pre-tree behavior exactly). |

## The explode's real footprint (`kyber/annotate.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| D55 | bug | **A row-*expanding* operator was budgeted at one morsel, and is not byte-bounded.** Measured rather than assumed: exploding 4,000 rows of a `fixed_size_list<float32, 768>` produces a **single output batch of 3,072,000 rows** — 12 MB against a 1 MiB morsel budget, 11.7x it — because the operator emits its whole fan-out for a morsel in one go and nothing re-cuts it. At a full 16,384-row morsel that is ~100 MB from a 1 MB input, budgeted at 65 KB: a 1,536x under-count on the RAG/embedding pipeline shape, in the direction that lets admission accept a query the node cannot run. The fan-out now multiplies the in-flight rows and the morsel byte cap is deliberately not applied — capping there reports the budget rather than the operator. |
| D56 | test | The measurement the budget rests on is pinned (`test_the_explode_output_really_is_one_unbounded_batch`), so if the engine ever starts re-morselizing an explode the test fails and the estimate is corrected rather than silently drifting from it. A one-to-one explode is unchanged. |
| D57 | hygiene | The fan-out is read from the estimator's own propagated counts (output rows over input rows), so it is exact wherever the type proves it (D29) and carries any learned correction otherwise — one source of truth rather than a second rule that could drift from the cardinality estimate. |
| D58 | bug | `Unpivot` has the same shape and the same under-count: 4,000 rows over 20 columns emit one 80,000-row batch. The rule is stated as a property of the operator's **fan-out** rather than of a node type, so it covers `Unnest`, `Unpivot`, and any future expander without a list to keep in sync — while a `Filter`, `Project`, or `Limit`, which cannot expand, keeps the byte cap exactly (pinned by `test_a_row_shrinking_operator_is_still_byte_capped`). |

## Coordination note — the morsel cut has two halves

`carbonite/policies/morsel.py` decides *what* a morsel's row target should be (D25-D28) and
`io/base/source.py::FileSource._normalize` is where an oversized source batch is actually
*cut* to it. They compose: `_normalize` reads `active_config().execution.morsel_rows`, and
the conductor's scoped config (`ResourceManager.recommended_config`) rewrites exactly that
field from the byte-aware cap. So a source whose reader parses a whole file into one chunk
is cut to a row count that fits the byte budget rather than to a flat 16,384 rows, which on
a decoded tensor column is the difference between 900 KB and 2.4 GB per batch.

Recorded because the two halves live in different subsystems and were built independently;
neither is complete on its own.

## The adaptive gate — a work threshold measured in rows (`api/adaptive/gating.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| D59 | scale | **The adaptive re-optimization floor was a pure row count, and its own rationale is about work.** `_ADAPTIVE_MIN_INPUT_ROWS` (20M) exists because staging trades a ~20-40 ms re-plan for a better downstream join choice, which pays only once a mis-estimated plan would cost more than that. Rows proxy for work only while a row is the ~64 bytes `optimizer.row_bytes` assumes, and across the modality range the proxy inverts at **both** ends: 20M rows of two `int64` keys is 320 MB and turned adaptation *on*, while 1M rows of decoded 224x224x3 images is **150 GB** and turned it *off*. The single most expensive query class in the engine was the one class the adaptive loop never ran on. The floor is now cleared by rows **or** bytes. |
| D60 | scale | The byte floor is derived from the two existing knobs (`min rows x row_bytes`) rather than added as a third, and the two are combined with **OR**, so a narrow query clears exactly the floor it always did and nothing that used the one-shot route is moved off it. The size floor also remains one condition among several — `_adaptive_would_help` still requires a join with a breaker-produced operand whose size is genuinely unknown — so a trivial wide pipeline does not qualify merely by being wide (pinned by `test_a_join_is_still_required`). |
| D61 | fidelity | The size probe reports `(rows, bytes)` from one walk instead of rows alone, and takes the width from the **scan's** own schema and measurements rather than from a guessed intermediate — the same discipline the row count already followed, extended to the other axis. |

### Open, recorded rather than changed

**`Unnest` is classified CPU-light while now being budgeted at ~100 MB.** `_CPU_LIGHT_KINDS`
marks an explode as IO/decode-bound so it asks for a fractional `cpu_share_io` and the
cluster packs several per core. That was defensible while the operator was believed to hold
one morsel; with D55 it holds its fan-out, so packing `n` of them per core is `n x 100 MB`
on one node. An explode is also not obviously IO-bound — it is a gather that materializes
`fanout x` the rows, which is memory-bandwidth work, not waiting.

Not changed, and the reason was **checked rather than asserted** — the claim "the loop
already corrects it" is exactly the kind of note that is worth nothing unverified.

Measured end to end: an explode records `op_stats` under the `unnest` tag with a positive
`cpu_utilization` on every run, and `load_cpu_utilization` surfaces it — *after* the refresh
throttle. That last part is the non-obvious bit and cost some time to find: the first
optimize of a process caches an empty map against a cold hub, and `_REFRESH_AFTER` (64 hub
version bumps) holds it there, so twenty-five runs of a query still read `{}` and only around
sixty runs refresh it. That is the documented throttle working, not a fault, but it means
"the loop corrects it" is true on a *warm* store and not on a lightly-exercised one.

What the measurement then says is the opposite of the worry: for the shape measured the
learned utilization is ~0.011 of a core, which supports the CPU-light classification rather
than contradicting it. That is one shape at one size and the figure is overhead-dominated,
so it does not settle the general case — but the concern narrows to the **memory** side
(`n` co-packed explodes at ~100 MB each), which is a packing-density question for
`bench-dist` rather than an argument.

## The correction that switched itself off when warm (`carbonite/memory`)

| # | Cat | Improvement |
|---|-----|-------------|
| D62 | bug | **D24's concurrent-peak walk applied only on a cold store.** `learned_plan_peak` routed the warm path to `LearnedMemoryModel.plan_peak`, which takes a flat `max` over blended operators — the pre-`inputs` reading. So the schedule walk was live only until the hub had learned anything about any operator family, which is to say only until the engine stopped being cold. A correction that silently stops applying once a system is warm is worse than one never written, because every cold-path test covering it stays green. The blend is now applied **per operator** and the schedule walked over the result, so the two are orthogonal: `warm == 2 * cold` under a doubling model *and* still above the largest blended single operator. |
| D63 | hygiene | `LearnedMemoryModel.plan_peak` is deleted rather than left beside its replacement. It had no production caller after the fix, and it is exactly the flat `max` a future caller would reach for — leaving it is how the bug comes back. The envelope's per-operator sizer is now a seam on the one walk (`_peak(plan, size_of)`), so "how big is this operator" has a single answer whatever is asking. |
| D64 | test | The test double that stood in for the model implemented only the whole-plan aggregate, which is why the flat reading looked correct. It now implements `blend_peak`, the per-operator primitive, and the reason is stated in it: a model that can only answer for a whole plan forces the walk to be flattened. |

## Containment is one question asked of three container types (`kyber/stats/selectivity`)

| # | Cat | Improvement |
|---|-----|-------------|
| D65 | fidelity | **`json.contains` and `list.contains` fell to the "no information" prior while `str.contains` got the substring one.** All three ask the same question — does this value occur *inside* this composite value? — and none can be answered from column statistics without element-level histograms, so they share a prior because they share a shape. `json_contains` is a `StrFunc` that simply was not in the family table; `ListContains` reached no dispatch branch at all and took the trailing `default_filter_selectivity`. Both therefore estimated **ten times** as many survivors as the identical predicate over a string — a difference in the *container's type*, not in the question — on exactly the semi-structured data where a bad cardinality is hardest to recover from downstream. |
| D66 | docs | **`json.exists` is deliberately left at the no-information prior**, and the reason is recorded beside the table rather than left to be re-derived. It asks whether a *path is present*, which is a schema question rather than a value search: in a schema-on-read corpus a field someone queries is often in most documents and sometimes in almost none, and that spread is genuinely unknown. Putting a number there would be inventing one. |

### Closed: two veins that turned out not to be worth building

**Nested-column zone-map pruning.** The consumer was scoped and then *not* built. The
producer only half exists: `bc-io/src/footer_stats.rs::parquet_column_index` requires a
single-part leaf path and skips anything nested, so the native walk — the path a local or
object-store read normally takes — is flat-only **by design**. A `StructField`-chain
resolver feeding nested statistics into selectivity and pruning would be correct and dead
for most reads, which is the speculative-generality trap. It becomes worth building the day
the Rust walk carries nested leaves; until then D49's fix (nested bounds no longer
*polluting* a top-level column) is the whole of the correct attribution.

**Retuning `broadcast_max_bytes` against sharper widths.** Raised by D46 and closed by it:
the widths only sharpened on sources that set `content_byte_size`, and TPC-H reads Parquet,
which does not. There is nothing for the threshold to be re-tuned *against* until a
columnar source's width estimate genuinely moves.

## The ensemble, not the rule (`tests/unit/test_decision_surface_matrix.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| D67 | test | **A cross-product test over `{narrow, embedding, image, video_frame} x {1, 8, 1024 workers}`.** Every other test in this ledger pins one decision; the failures it exists for were never in one rule. A width, a memory envelope, a morsel, a task count, and a shuffle cost all read the same data, each looked reasonable on its own, and together they sized a multimodal pipeline by three orders of magnitude wrong. This is `CLAUDE.md`'s `{collect, spill, iter_batches, distributed} x {nulls, empty, NaN, descending}` discipline applied to *decisions* rather than results: it asserts the invariants that must hold in **every** cell — the width is within an order of magnitude of the truth, a morsel never exceeds its byte budget, a task never holds more than the byte target, the envelope is monotone in the modality, single-node is charged no network, and shuffle cost is monotone in both the fleet and the row width. |
| D68 | test | **The matrix was checked against the pre-session commit and 13 of its 27 cells fail there**, which is what makes it worth having rather than a set of assertions true by construction. Two of those (the image and video-frame width cells) fail on *behaviour* — a decoded frame was sized at the 32-byte variable-length prior. The other eleven fail because the API did not exist: there was no `net` axis to be monotone in, and `morsel_target` took no plan to size a cold store from. Both are meaningful and they are not the same thing, so the distinction is recorded rather than reported as thirteen caught regressions. |

| D69 | test | The open item above was verified rather than left as a plausible note. The learned CPU-share override does activate for `Unnest` (`op_stats` records `cpu_utilization` on every explode, `class_ir_tag` maps the family, `load_cpu_utilization` surfaces it) — but only past `_REFRESH_AFTER`, because the first optimize of a process caches an empty map against a cold hub and holds it for 64 version bumps. Twenty-five runs still read `{}`; around sixty refresh it. Worth recording because "a measurement will correct this" reads as immediate and is not. |

---

## Where to continue

The entries above share one shape, and naming it is more useful to the next reader than the
list: **a decision tuned on `int64` columns, applied unchanged to data three to six orders
of magnitude wider, failing silently.** The plan is produced, it runs, and it OOMs or spills
for a reason nothing in the plan explains. Four questions found most of them, and they are
worth asking of any decision surface not yet covered here:

1. **Is the threshold in rows?** A row count assumes a row width. Every row-only threshold
   examined so far inverted across the modality range (D25, D37, D59). The fix is never to
   replace the row term but to take `max(rows-derived, bytes-derived)`, so narrow data is
   untouched by construction.
2. **Does the type prior see through the label?** No `pa.types.is_*` predicate sees through
   an extension type, and every multimodal column in Batcher wears one (D1, D31).
3. **Does the correction survive going warm?** A fix on the cold path that a learned model
   routes around applies only until the engine learns something, and every cold-path test
   stays green while it does (D62). Ask what the *warm* path does.
4. **Is the same question answered differently by container type?** Containment over a
   string, a document, and a list is one question and got three answers (D65). So did "how
   many tasks" (D38) and "how big is this join" (D21).

Concrete work these leave open, in the order their evidence is strongest:

- **Nested statistics need the Rust side first.** `bc-io/src/footer_stats.rs::parquet_column_index`
  requires a single-part leaf path, so the native walk is flat-only by design and nested
  footer bounds reach only the Python fallback. A `StructField`-chain resolver feeding
  selectivity and zone-map pruning is scoped and deliberately unbuilt until then.
- **The anchored/substring priors are provably inconsistent** and the containment argument
  says to clamp them, while TPC-H Q14 says the clamp makes the absolute error worse (D36).
  Settling it needs `benchmarks/run.py`, not reasoning.
- **Packing density for a row-expanding operator.** With D55 an explode is budgeted at its
  fan-out, so `n` co-packed explodes are `n x ~100 MB` on a node. The CPU classification has
  measured support (D69); the memory side is a `bench-dist` question.
