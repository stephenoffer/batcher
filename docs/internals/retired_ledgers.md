# Retired working ledgers

The engineering ledgers that used to live here and at the repository root were **retired
on 2026-07-29**, deliberately and all at once. They were not lost, and they must not be
restored.

If you are reading this because a ledger you expected is missing, that is why. Two of them
carry a note from a session that read an earlier stage of this same removal as an unexplained
truncation and restored them three times. That note is wrong about the cause: nothing is
silently reverting the tree. The files were removed on purpose.

## What was retired

Fourteen files, each a running record of one improvement campaign:

| Retired | Was at |
|---|---|
| AI / LLM workload | `AI_WORKLOAD_LEDGER.md` |
| DL / LLM workload | `DL_LLM_WORKLOAD_LEDGER.md` |
| DS / ML workload | `DS_ML_WORKLOAD_LEDGER.md` |
| Engine parity | `ENGINE_PARITY_LEDGER.md` |
| Metadata improvements | `METADATA_IMPROVEMENTS_LEDGER.md` |
| Architecture audit | `docs/internals/audit_ledger.md` |
| Defect hunt | `docs/internals/bug_hunt_ledger.md` |
| Daft parity | `docs/internals/daft_parity_ledger.md` |
| Decision quality | `docs/internals/decision_quality_ledger.md` |
| Engine improvements | `docs/internals/engine_improvements_ledger.md` |
| Estimator mathematics | `docs/internals/math_improvements_ledger.md` |
| Spill / OOM | `docs/internals/spill_oom_improvements_ledger.md` |
| Streaming improvements | `docs/internals/streaming_improvements_ledger.md` |
| UDF improvements | `docs/internals/udf_improvements_ledger.md` |

Every entry described a change that is in the code with a test, so the content is recoverable
from git history and the behaviour is pinned by the suite either way. Read the history if you
need the reasoning behind a specific entry; `git log --diff-filter=D --name-only` finds the
removal commit for any path above.

## What was still open when they were retired

Nothing was half-landed. Both items either ledger marked "needs a human" had already been
fixed by the time of the audit: the zone-map `NOT` unsoundness
(`kyber/rules/zonemap_pruning.py`) and the `hex` / `lpad` / `rpad` constant folds
(`kyber/rules/exprs/text_folds/`). The rest of the open register was work argued and
deliberately declined, each for a reason worth keeping:

- **No authentication.** A `Principal` is caller-asserted, so in-process code can claim any
  identity; the trust boundary is the process. Stated in the user's language in
  `docs/user-guide/hardening.md`.
- **Sampling is biased on duplicate-heavy data.** `sample(fraction=)` hashes row *content*,
  so duplicates share one coin flip. Determinism and partition-independence
  (single-node == distributed) are mathematically incompatible with an unbiased Bernoulli
  sample of a multiset. A fix needs a per-row identity, not a content hash.
- **Measured RSS is collected but unused.** `OperatorFeedback.peak_rss_bytes` is a delta in a
  process-wide counter, so under concurrency it attributes one operator's growth to whichever
  was running. Fitting Carbonite's memory model on it would inflate every reservation from a
  mis-attributed sample. Needs per-operator attribution first.
- **Footer-derived column widths are a near miss.** A Parquet footer's
  `total_uncompressed_size / num_values` is an average width for free, but it measures the
  *encoded* size, so a dictionary-encoded string column reads far narrower than its Arrow
  width. Feeding it to the width estimator would understate exactly the columns that matter
  most. The right source is the measurement Core already takes.
- **Learned state still lags on the distributed path**, though less than when this was
  written. `record_execution` / `record_selectivity` now fire on the distributed relational
  route (`stages._record_distributed_cardinality`, counting only where the count is already
  known so learning never forces a materialization) and on the distributed `map_batches`
  branch. What remains unmetered: the Flight transport paths, `dist/executors/write.py`, and
  `reduce_join_paths_spilling`; `learn_column_stats` and `record_run_feedback` are still
  single-node. "Single-node == distributed" holds for *results*, not yet for all learned
  state.
- **Spill throughput is a table, not a measurement.** `kyber.storage_cost.spill_device_factor`
  prices a spilled byte from a static device-class table read out of `/sys` (10x network, 30x
  rotational, local flash the omitted baseline), while Core records `spill_bytes`,
  `io_write_bytes` and `t_op_ms` per operator — enough to measure the real sustained rate of
  the configured spill directory. Declined because that table is *calibrated against* the cost
  term it multiplies, so replacing it re-tunes two coupled numbers rather than sharpening one
  — the same trap `SourceStatistics.content_byte_size` records, where a more accurate width
  produced worse plans. Needs `bench-ops` and `test_spilling.py` under real memory pressure,
  re-tuning both together.
- **The general two-inequality join still runs IEJoin** and loses above roughly 1M rows. Band
  detection closed the single-key case; block pruning is the fix for the general shape.
- **Token n-gram metrics need a Rust primitive.** BLEU, ROUGE-N, METEOR and Distinct-2/3 need
  token n-gram counting; `.str.chunk` gives character n-grams only, and there is no multiset
  op over a token list. chrF is reachable and shipped.
- **A zero-row projection can report an all-`Null` schema.** `Limit(_, 0)` over certain kernels
  (decimal division, list reductions) emits no typed empty batch, so a `limit(0)` schema probe
  degenerates. Worked around for the list case in `plan/types/infer.py`; the engine-side fix is
  in the empty-input projection path.

Smaller flagged-and-declined items (`product` over `bool`/`string` casting silently where
DuckDB rejects, `RelOp::from_json` recursing unbounded on a small rayon stack, streaming
`batch_size` being a max rather than an exact size) are in the history alongside their
reasoning.

## Why they are not coming back

A ledger is a good instrument while a campaign is running and a liability afterwards: it
accumulates entries faster than anyone re-reads them, it duplicates what the tests already
prove, and its "still open" register goes stale silently. The durable record of a change is
its test; the durable record of a decision is the comment next to the code that implements it.
Both outlive a ledger and neither can drift from the thing it describes.

If you are running a campaign that wants a scratch record, keep it outside the repository or
in a branch, and land the conclusions as tests and comments.
