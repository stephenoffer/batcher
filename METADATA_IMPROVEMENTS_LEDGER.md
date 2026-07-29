# Metadata improvements ledger

A running record of work on Batcher's metadata plane: what the engine *collects* about
data and execution, where it *persists* it, and how the optimizer *spends* it. Each entry
names the defect or gap, why it was invisible, and what changed.

The theme running through most of it: metadata failures are silent. A learned-stats loop
that stops working raises nothing and produces no wrong answer. Plans simply stop
improving, which is only visible to someone running a benchmark. So the entries below are
weighted toward things that were quietly switched off rather than things that broke.

## M1-M8 — the derived views over the feedback history

**M1. One corrupt row emptied the whole measured history.** `bucket_by_kind` and
`chronological_signed` each wrapped their entire scan loop in a single `try`. A truncated
write, a row from a build with a different shape, or a value another process was mid-write
on did not cost that row — it aborted the loop, and for the signed view it returned `[]`.
That reads as "this session has measured nothing", which switches off cardinality
correction and cost calibration wholesale. Decoding moved into `_rows`, which isolates per
row, skips non-object values, and caps its own warning volume.

**M2. Both views loaded from separate scans.** They are read together on every optimize —
calibration wants the by-kind buckets, correction wants the signed history — but each
loaded itself, so the first query of a process scanned `op_stats` twice and ran
`json.loads` over every stored row twice, producing the same objects both times.
`build_views` does one pass and shares the rows that land in both views.

**M3. A row recorded during a view load was dropped from it.** `record` skips folding into
a view that is still `None`; the load then assigned over the top. The row survived in the
backend but did not reach the model until the next process — and it was the newest
measurement, the one the learning loop weights most. The load now holds the hub's lock.

**M4. A stored feedback table was never pruned by a short-lived process.** `record` prunes
every 4,096 rows *it* wrote, which a process-per-request workload never reaches. So a
durable store grew a row per operator per query forever and every new process parsed all of
it before its first plan. The first view load now prunes from the keys it already has, at no
extra scan.

**M5-M8.** Per-row decode warnings are capped (`_DECODE_WARN_LIMIT`); a malformed storage
key sorts as oldest during a prune rather than raising; `bucket_by_kind` and
`chronological_signed` were deleted once `build_views` subsumed them; `_prune_op_stats`
accepts already-read keys.

## M9-M15 — the learned-parameter store

**M9. One shared cache generation across every namespace.** The parsed-read cache was
validated against a single counter, and `save_params` bumped it. Source statistics are
persisted under a namespace *per source path*, so writing one dataset invalidated the
parsed view of every unrelated namespace — the learned throughputs, the tuning posteriors,
the calibration coefficients — which the next query re-scanned and re-parsed. Generations
are now per namespace.

**M10. `load_params` re-parsed on every call.** The keyed path was cached; the whole-blob
path was not. `load_source_stats` calls it once per bound source per query, and a source
blob carries a column map with bounds, blooms, and quantile grids. Now cached under the same
per-namespace generation, and patched in place by `save_params`.

**M11. The parsed views grew without bound.** One namespace per source path meant a session
reading thousands of files retained every decoded blob for the life of the process. Bounded
at `_NAMESPACE_CACHE_MAX`, evicting a namespace from all four maps *together* — dropping a
generation counter while leaving a view behind would let the counter restart and climb back
to a number the stale view still carried.

**M12. `hub.py` split.** Feedback absorption and learned-parameter storage share only a
backend. The latter moved to `metadata/params.py` as `LearnedParams`, which the Hub composes
and delegates to.

**M13-M15.** `load_params` no longer hands back a non-mapping when the store holds a foreign
blob; the read-only contract of the returned view is documented at both call sites; a
non-dict legacy blob no longer breaks `load_keyed`.

## M16-M20 — learned scalars

**M16. One non-finite observation poisoned a learned scalar permanently.** Exponential
smoothing is `prior + step * (value - prior)`, which propagates a NaN or infinity into the
stored value and from there into every subsequent update. The producers are all ratios —
bytes over elapsed, observed over predicted, used over capacity — so a zero denominator is
the ordinary way one arises. Non-finite observations are now dropped.

**M17. A poisoned stored value was adopted on read.** Consumers divide by these scalars or
compare them against thresholds, where a NaN fails every comparison silently. A non-finite
stored value now reads as "never recorded", so the next observation restarts the estimate.

**M18. A nonsense observation count could invert the blend step.** `count` decides the step;
at `-1` it divides by zero and below `-1` it flips the sign, moving the estimate *away* from
every observation. Clamped to at least one observation.

**M19-M20.** A stored `True` no longer reads as `1.0`; `_count_of` isolates the count read.

## M21-M26 — the GPU learned figures

**M21. `ml/gpu.py` kept its own whole-blob read-modify-write.** Two more copies of the
smoothing loop `metadata/smoothed.py` exists to hold, using the static
`learning_smoothing_alpha` (0.5) rather than the running-mean-then-exponential
`max(floor, 1/(n+1))`. Under a static 0.5 the *first* observation still holds an eighth of
the value after four runs and never washes out, so the cold profiling run — the one most
likely to be unrepresentative — anchors the estimate it was supposed to be smoothing.

**M22. Two inference pipelines recording at once lost each other's update.** Loading a
namespace's whole blob, editing one key, and writing it back is a lost update, and an
autoscaled fleet records constantly. Per-key writes fix it.

**M23. Reading one model's figure parsed the whole fleet's.** One blob held every pipeline's.

**M24-M26.** Both namespaces stay hardware-`scoped`; a store written by the old whole-blob
shape keeps answering, because the per-key view merges a legacy blob underneath its own
entries; the duplication is gone, so the smoothing policy has one definition again.

## M27-M40 — the persistence backends

**M27. A SQLite store was unusable from a worker thread.** `sqlite3.connect` defaults to
refusing cross-thread use, and the hub is a process singleton whose writers are worker
threads. `record` catches the resulting `ProgrammingError` and logs it, so the symptom was
not an error but a durable store that persisted *nothing* from any pipeline off the main
thread. Reads had no such catch and raised into planning. Now `check_same_thread=False` with
the serialization duty discharged by an explicit lock.

**M28. Every `put` paid a rollback-journal fsync.** One row per operator per query, each
committing, turned learning from execution into a per-query tax. WAL plus
`synchronous=NORMAL` keeps the writes append-only. The trade — a crash can lose the newest
few commits — costs a plan fitted on slightly less history, which re-converges in a handful
of queries.

**M29. A prefix scan read the whole table.** `learned_params` holds every namespace at once.
The prefix is now a range over the primary-key index, exact for string prefixes and a
documented superset for numeric ones, with the tuple comparison kept as the authority.

**M30. SQLite could not delete, so the durable default was never pruned.**

**M31-M33. Object storage: one sequential `cat_file` per object.** A scan of the feedback
table was tens of thousands of *sequential* round-trips before the first plan of a process
— against S3 at 30 ms per GET, a cold start measured in minutes. Now chunked concurrent
`fs.cat` with `on_error="omit"`, so an object another driver prunes mid-scan is skipped.
`batch_put` uses `fs.pipe`. `delete` added.

**M34-M35. Redis: full `HSCAN` and no `HDEL`.** The prefix is now an `HSCAN MATCH` pattern
with glob metacharacters escaped — a namespace is a file path or a model name, so `[` and
`*` appear in one literally, and an unescaped `[` turns the pattern into a character class
that matches nothing, reading as "never learned" rather than as an error.

**M36-M37. The layered backend could not delete and lost a caller's cache.** `refresh()`
rebound to a fresh `InProcessBackend`, silently discarding a caller-supplied cache on the
first refresh; it now empties in place. `delete` forwards to both layers, so the cache never
keeps answering for a key the shared store has forgotten.

**M38. `InProcessBackend.scan` yielded straight out of a live dict.** It is a generator, and
its callers consume it lazily while `record` writes to the same table from another pipeline
— `RuntimeError: dictionary changed size during iteration`, raised into planning. It now
snapshots.

**M39-M40.** `InProcessBackend.clear` added for `refresh`; every backend that can forget now
does, so the hub's prune is no longer a no-op on three of the four configurations.
