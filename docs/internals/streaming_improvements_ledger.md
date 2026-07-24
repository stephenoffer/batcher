# Streaming improvements ledger

A running record of improvements, fixes, and features that make Batcher better at
**streaming workloads** — the continuous/unbounded micro-batch engine (triggers, output
modes, watermarks, windowed and stateful aggregation, checkpoint/recovery, streaming
sources and sinks) and the Tier-0 morsel-streaming executor beneath the batch path.

Each entry is a distinct, tested change. Entries are numbered `S<n>` continuously and never
reused, so the count is a count of *distinct* improvements. Category tags:
**bug** (wrong result / data loss), **robustness** (crash/leak/hang on a reachable path),
**feature** (new capability or Spark-parity gap closed), **validation** (fail-fast on bad
input), **perf**, **test** (coverage that pins a contract), **hygiene** (dead code / dedup),
**docs**.

This page is not published (`exclude_patterns` in `docs/conf.py`); it is a working index for
anyone touching the streaming subsystem.

---

## Control-plane value types and query lifecycle (`plan/streaming`, `api/streaming`)

| # | Cat | Improvement |
|---|-----|-------------|
| S1 | bug | `parse_interval_seconds` accepted a non-finite duration: `nan < 0` is False and `inf` is "non-negative", so both slipped past every check. A NaN trigger cadence makes `remaining > 0` always False and busy-loops the micro-batch thread; an infinite watermark lateness overflows the microsecond literal it lowers to. Now rejected with a `PlanError` at the one gate every duration flows through (`plan/streaming/spec.py`). |
| S2 | robustness | Two active streaming queries under the same explicit `name=` silently evicted the first from the `bt.streams` registry while it kept running — unreachable, unstoppable, still writing to its sink. Spark rejects a duplicate *active* name; a new `_register` helper now enforces it (a stopped name is free to reuse). Applied to both the single-node and distributed launchers. |
| S3 | robustness | `start_streaming_query` registered the query in `_ACTIVE` *before* `engine.start()`, which opens the sink and recovers from the checkpoint. If either raised, the never-running query lingered in the registry as a phantom active stream. `start()` is now wrapped so a failure deregisters it. Applied to both launchers via the shared `_deregister` helper. |
| S11 | feature | `StreamingQueryEngine.status()` reported `message="Stopped"` for a query that *failed*, indistinguishable from a clean stop unless the caller used `await_termination`. It now surfaces `"Failed: <type>: <msg>"` so a caller polling `status` sees the failure and its cause (`core/streaming_query.py`). |
| S18 | feature | Added `StreamingQueryProgress.output_rows_per_second` — emission throughput, distinct from `input_rows_per_second` (arrival throughput): a filter or windowed aggregate emits far fewer rows than it reads, so the two diverge and both are worth exposing (`plan/streaming/spec.py`). |

## Streaming sources and sinks (`io/formats/streaming`)

| # | Cat | Improvement |
|---|-----|-------------|
| S4 | perf | `SeenStore.confirm()` (the Auto Loader durable dedup) committed once per file — one fsync each. Added `SeenStore.mark_many`, a single `executemany` + one commit, so confirming a discovery pass of N files costs one fsync (and one atomic transaction) instead of N (`seen_store.py`, `autoloader.py`). |
| S5 | bug | The Event Hubs source read each payload with `body_as_str()` then re-encoded to UTF-8 — a binary payload (protobuf/Avro/compressed) raised `UnicodeDecodeError`, and a non-UTF-8-but-decodable one was silently mangled. Now reads raw bytes via `body_as_bytes()` through the new pure `_event_to_message` helper (`eventhubs.py`). |
| S6 | bug | The Event Hubs `partition_key` (a `str`) was passed into the `key` column declared `pa.binary()`, raising `ArrowTypeError` in `_make_batch`. Now coerced to `bytes` (and `None` preserved) by `_as_bytes` (`eventhubs.py`). |
| S7 | robustness | The Event Hubs `_poll` created a per-partition AMQP consumer (`_create_consumer`) on every poll and never closed it — a continuous stream leaked one consumer (socket + threads) per partition per poll. Each consumer is now closed in a `finally` (`eventhubs.py`). |
| S26 | bug | The Event Hubs source ignored checkpointed positions on recovery: it always rebuilt each consumer at the configured `starting_position`, silently replaying or skipping on every restart, while `kafka`/`kinesis`/`pulsar` all honor their checkpoint. Now each message carries its offset string as `resume_token`, and `_poll` resumes each partition from `_resume_from` (populated by `seek`) — real exactly-once recovery (`eventhubs.py`). |
| S8 | bug | The `socket` dev source stripped only `\n`, so a CRLF (`\r\n`) producer left a trailing `\r` in every `value`. Now `rstrip("\r\n")`; also removed the always-empty `buf` dead variable (`dev.py`). |
| S9 | bug | The Kinesis source reused a closed shard's stale iterator forever: a `GetRecords` with no `NextShardIterator` (a drained/closed shard) left the old iterator in `_iterators`, so the next poll raised `ExpiredIteratorException` in a loop. The shard is now retired into a `_closed` set and dropped from `_active_shards` (`kinesis.py`). |
| S25 | robustness | A Kinesis `ProvisionedThroughputExceededException` (the 5-TPS/shard `GetRecords` cap — routine back-pressure) crashed the whole query. Now a throttled shard is skipped for that poll (its iterator unchanged, records read next poll) with no blocking sleep, so the trigger cadence paces the retry. `_is_throttle` matches both the exception class name and the boto `ClientError` code (`kinesis.py`). |
| S13 | feature | The Kafka poll timeout was hard-coded at `1.0s`, coupling both stop-latency (a `stop()` is only observed between polls) and poll efficiency to a constant. Added a `poll_timeout` option (kept out of the confluent-kafka config so it can't leak as a bogus `poll.timeout` key) (`kafka.py`). |
| S14 | robustness | A Pub/Sub `pull` with no deadline blocks indefinitely on an idle subscription, stalling the trigger cadence and `stop()`. Added a bounded `pull_timeout`; a deadline hit with no data (`DeadlineExceeded`/`RetryError`, matched by name so `google-api-core` need not import) is treated as an empty poll, while a real error still propagates (`pubsub.py`). |
| S15 | feature | The incremental file source (Auto Loader analog) ingested an entire backlog in one micro-batch. Added `max_files_per_trigger` (Spark `maxFilesPerTrigger`) backpressure: a pass admits at most N new files, oldest-name-first, leaving the rest undiscovered for later passes so a large backlog drains across bounded epochs (`autoloader.py`). |
| S17 | feature | The Pulsar `receive` timeout was hard-coded at 1000ms, coupling stop-latency and idle-poll latency to a constant (same class as S13). Added a `receive_timeout_millis` option, kept out of the client options (`pulsar.py`). |
| S20 | perf | `RateSource._make_batch` built each batch's timestamps with a Python list comprehension over `n` `datetime` objects — O(rows) Python in the data plane, the exact hot-path tuple touch the architecture rule forbids. Vectorized to `np.arange` + a single `pc.multiply(...).cast(timestamp[us])`, proven bit-identical to the old closed form across `rps ∈ {1,2,7,1000}` (`dev.py`). |
| S38 | robustness | The base broker `_poll_loop` re-polled immediately on an empty poll (`if not messages: continue`), spinning a full core for any broker whose `_poll` returns at once when idle — Kinesis `GetRecords` always returns, and a shard skipped for throttling (S25) returns empty instantly, so the loop hammered the rate-limited API into more throttling. Added a geometric back-off between empty polls, capped at 250ms and reset on real data, so an idle fast-empty broker rests instead of spinning; a broker whose `_poll` already blocks (Kafka) rarely reaches it (`broker.py`). |
| S39 | robustness | `DeltaStreamSink.write_batch` guarded exactly-once with only a non-atomic `is_committed` pre-check, so two writers sharing an `app_id` (concurrent drivers, or a query racing its restart) both passed the check and the loser's commit crashed the query with a `CommitError` even though the batch was durably committed by the peer. The commit is now wrapped: on `CommitError` it re-reads the log and treats a now-recorded transaction as benign, re-raising only when the transaction is genuinely absent (`sinks.py`). |
| S40 | robustness | `ForeachBatchStreamSink`/`ForeachStreamSink` returned `None` tokens, unlike the File/Delta sinks whose batch-id-bearing tokens the commit-log `sink_token` column records. They now return `foreach_batch:{batch_id}`/`foreach:{batch_id}`, honoring the `StreamSink` token contract uniformly so the commit-log can dedup a replayed `foreachBatch` once the runner is wired to pass the token through (`sinks.py`). |

## Watermark stream operators (`api/terminal/stream`)

| # | Cat | Improvement |
|---|-----|-------------|
| S21 | bug | `stream_watermark_dedup` re-emitted a duplicate when the event-time column was null: a null-time row was kept before any watermark, its key folded into the seen set, then evicted on the next batch (a null fails the `>= watermark` keep), forgetting the key — so a later genuine occurrence re-emitted as new. Null-event-time rows are now dropped uniformly (matching the post-watermark late filter) (`watermark.py`). |
| S22 | bug | `stream_stream_join` matched every pair when both streams named their event-time column identically: the join suffixes the right side to `<col>_right`, but the interval filter differenced the raw pre-suffix names, so both resolved to the *left* column and `|Δt|` was always 0. Each side's true output alias is now resolved from `plan.output` (`watermark.py`). |
| S23 | validation | A nested 3+-way stream join (`a.join_stream(b).join_stream(c)`) fell through to a generic "must materialize" error. Now raises a `PlanError` naming the exact two-stream limit and how to restructure (`dispatch.py`). |
| S24 | validation | The unbounded-source-must-materialize terminal error now names the offending top-level operator (e.g. "top-level Sort") instead of a generic list, so the user knows which breaker broke streamability (`dispatch.py`). |

## Distributed / spilling global-window stream (`dist/window_stream.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| S30 | feature | `avg` was in the fallback set of the bounded-memory global-window streamer (correct but not memory-bounded). A running mean isn't a constant offset, but its running `sum` and `count` each are — so private running `sum`/`count` helpers are injected into the window IR, offset by the prior buckets' totals, divided, and dropped before yield. `avg` now streams with bounded state like `sum`/`count`, pinned differentially vs the in-memory kernel across descending × {1,16,64} buckets (`window_stream.py`). |
| S31 | bug | `LogicalPlan.to_ir()` memoizes and returns the plan's *shared* dict/list; the global-window streamer mutated it in place (`win_ir["input"] = …`, plus the new `avg` append), so a second structurally-equal run in the same process accumulated appended helpers and raised `Field "__ws_sum::a" exists N times in schema`. Now copies the dict and the `functions` list before mutating — proven to fail (KeyError) without the fix (`window_stream.py`). |
| S32 | robustness | The `avg` offset used a naive `sum + prior_total`, which carries the kernel's empty-sum NULL forward and diverges (e.g. a null-only leading bucket). It now coalesces the within-bucket running sum to 0 before adding the prior total and emits NULL only where the *global* running count is 0 (`window_stream.py`). |
| S33 | test | Added differential coverage for `first_value` in the global-window stream (descending × {1,16,64}), which streamed correctly but had no test (`tests/integration/test_spilling.py`). |

## UDF / ML streaming (`core/udf/stream.py`, `ml/streaming_sampler.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| S34 | bug | On the streaming CPU stage with `num_workers > 1`, `_parallel_units` split each morsel into `workers` even slices *even when the user set an explicit `batch_size`* — a 200-row morsel with `batch_size=100, num_workers=4` became four 50-row units, not two 100-row chunks. This violates the "explicit `batch_size` always wins" invariant and hands a batch-boundary-sensitive UDF the wrong-sized batches (silent wrong result / crash). Fixed by gating the fill-the-pool re-slice on `explicit is None` (`core/udf/stream.py`). |
| S35 | test | Pinned the `stream_eligible` routing contract — five documented shape claims (lone GPU stage needs a batch_size, CPU-only chain rejected, multi-stage-with-GPU accepted, multiprocessing stage rejected) that had no direct test (`tests/unit/test_udf_streaming_and_budget.py`). |
| S36 | test | Pinned `linear_map_chain` extraction (stages returned bottom-up; rejects a plan with no map and a non-scan root) (`tests/unit/test_udf_streaming_and_budget.py`). |
| S37 | test | Pinned `reconcile_stream`'s monotonic-schema behavior: a narrow-after-widen batch backfills the union column as a typed null, and an empty source yields nothing (only the widening case was covered before) (`tests/unit/test_udf_streaming_and_budget.py`). |

## Checkpoint / recovery (`io/formats/streaming/checkpoint`)

| # | Cat | Improvement |
|---|-----|-------------|
| S12 | robustness | The offset and commit logs grew one row per micro-batch forever — only the state store was pruned. A per-second stream accumulates millions of rows per month. Added `OffsetLog.prune`/`CommitLog.prune` and `CheckpointStore.prune_logs(keep_through)`, called after every commit (even for stateless streams). Recovery reads only the last committed batch's offsets and the commit maximum, so pruning `< last_committed` is proven safe by test (`logs.py`, `store.py`, `streaming_query.py`). |
| S16 | hygiene | Removed dead code in the state store: `StateStore.snapshot(..., meta)` wrote a `.meta.json` sidecar and `prune` deleted it, but no caller ever passed `meta` — the windowed fold's watermark rides in the Arrow schema metadata instead. Dropped the parameter, the sidecar write/delete, and the now-unused `json` import (`state_store.py`). |

## Morsel-streaming and core drivers (`core/streaming`)

| # | Cat | Improvement |
|---|-----|-------------|
| S10 | perf | `stream_topn` re-read and re-serialized `active_config().engine_config_json()` inside the per-micro-batch loop; hoisted it out (constant for the query), matching `stream_limit` (`core/streaming.py`). |
| S19 | test | The windowed watermark fold's careful cross-resolution timestamp normalization (event-time columns in `s`/`ms`/`us`/`ns`, all reconciled to the microsecond watermark/`window_start`) was claimed in comments but only ever tested at `us`. Added a resolution-parametrized eviction test proving each unit produces the batch-oracle result — reading raw non-`us` ticks would scale the watermark 1000× and drop every row (`tests/integration/test_watermark_window.py`). |
| S27 | test | The streaming top-N (`sort + limit`) and streaming `limit` drivers (`stream_topn`/`stream_limit`, the memory- and IO-bounded paths that keep only N rows / stop reading after N) had no direct test. Added integration tests pinning that top-N returns the global best N (not per-batch) and that `limit(n)` short-circuits an endless source after n rows instead of draining it (`tests/integration/test_streaming_execution.py`). |
| S28 | test | The streaming `distinct()` driver (`stream_distinct`, bounded-memory group-by over all columns) had no direct test; added one pinning that identical rows across micro-batches fold to the exact distinct set (`tests/integration/test_streaming_execution.py`). |
| S29 | test | Pinned the streaming keyless-aggregate empty-input contract: a global `count`/`sum` over an empty stream must yield one row (`0`/`NULL`), not zero rows — the incremental fold skips empty batches, so this exercises the `_empty_global_aggregate` fallback that keeps it agreeing with the DuckDB oracle (`tests/integration/test_streaming_execution.py`). |
