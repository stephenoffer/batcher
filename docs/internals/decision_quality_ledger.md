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
