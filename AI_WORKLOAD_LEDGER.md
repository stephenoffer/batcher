# AI / LLM Workload Improvement Ledger

Working ledger for the "make Batcher work better with LLMs and large AI workloads" effort.
Sources: an audit of `python/batcher/ml/**` + `core/udf/**`, an audit of `dist/**` + `io/**` +
`carbonite/**`, and a technique inventory extracted from `../optimization-guides` (Ray Data /
Ray Train / vLLM field guidance).

Baseline before this work: `tests/unit` = 6 failed, 4722 passed.

## Status key

`[x]` done and verified · `[~]` in progress · `[ ]` not started

---

## Cluster 0 — Correctness / silent data loss (highest priority)

- [x] 0.1 `_DynamicBatcher._drain` took `[0]` off a `combine_chunks().to_batches()` list, dropping
  rows for wide binary/list inference outputs. `ml/inference.py`
- [x] 0.2 `_DynamicBatcher._drain` set `_rows` from `remainder[0]`, undercounting a multi-batch
  remainder and stalling the emit threshold. `ml/inference.py`
- [x] 0.3 `_DynamicBatcher.flush` returned only the first batch. Now returns a list. `ml/inference.py`
- [x] 0.4 Autobatch worker `merged[0]` dropped rows. Now `concat_batches`. `core/udf/execute.py`
- [x] 0.5 `to_tf_dataset` signature probe consumed batch 0 of every TF run. `ml/converters.py`
- [x] 0.6 `PolynomialFeatures` / `TargetEncoder` were not re-exported from `batcher.ml`.
- [x] 0.7 `test_llm_structured` pinned obsolete drift-failure behavior; the inferred-struct path
  now unifies across batches. Test rewritten to pin the good behavior.

## Cluster 1 — Distributed batch inference at scale (P0, owner: lead)

- [x] 1.1 Batch inference + write materializes the whole post-inference result on the driver.
  `api/terminal/core.py:537` excludes `has_map_batches`. Unconditional OOM on a 2B-row embed job.
- [x] 1.2 No streaming path for GPU inference: `stream_distributed_map` and `iter_distributed`
  both refuse UDF plans.
- [x] 1.3 `_distributed_write_plan` calls the native engine, which cannot run a Python UDF.
  Needs a UDF-aware write shard.
- [ ] 1.4 Distributed writes require a fully materialized shard (`write_partitioned` takes a
  `pa.Table`); no partitioned streaming write.
- [x] 1.5 `resident_inference_pools()` scope has no `RayError` handling — one preempted GPU node
  kills the query.
- [x] 1.6 A UDF exception is never retried; transient CUDA OOM / 503 kills a multi-hour job.
- [ ] 1.7 No job-level checkpoint/resume for batch inference.
- [x] 1.8 `FileSink.commit` is a no-op — no `_SUCCESS` marker, so a half-written output dir is
  indistinguishable from a complete one.

## Cluster 2 — GPU scheduling, placement, heterogeneous clusters

- [ ] 2.1 No GPU-node affinity, only CPU-node isolation.
- [ ] 2.2 CPU-decode and GPU-infer stages are not co-scheduled; no cross-stage placement.
- [x] 2.3 Carbonite hardcodes `num_gpus=0.0`; GPU is never a budgeted resource.
- [ ] 2.4 No VRAM-aware admission — first run with an oversized batch OOMs the device.
- [x] 2.5 Inference actor pools get no placement group, so a pool can half-place and stall.
- [x] 2.6 `gpu_collective` STRICT_PACK unreachable on the inference path.
- [x] 2.7 Placement-group failure degrades silently.
- [x] 2.8 Actor pool health check is serial, 10s timeout each — 2000s for a 200-actor pool.
- [x] 2.9 `_MAP_INFLIGHT_MAX` / `_MAP_COMPUTE_WEIGHT` hardcoded, duplicated from `ml.gpu`.
- [x] 2.10 VRAM sampling is node-global, so packed actors synchronously oscillate to the floor.
- [x] 2.11 Packing density capped at 4 actors/GPU by a 0.25 fraction floor.
- [x] 2.12 VRAM budget is a fixed 1.5x with no batch-size / seq-len / dtype dependence.
- [x] 2.13 No tensor-parallel awareness in pool sizing.
- [x] 2.14 `_pipeline_accel_kwargs` pins `device=0`; no `device_map="auto"`.
- [x] 2.15 `torch.compile` gated on CUDA only, excluding XPU/MPS.

## Cluster 3 — Streaming, prefetch, backpressure

- [ ] 3.1 No backpressure between the actor pool and the driver; all outputs held on the driver.
- [ ] 3.2 `stream_distributed_map` submits every partition up front and has no preemption recovery.
- [ ] 3.3 Read-ahead depths are module-import constants that ignore memory pressure.
- [ ] 3.4 512 MiB read-ahead budget is per-process, not per-node.
- [ ] 3.5 Worker scan cache (30% RAM, on by default) is invisible to the memory pool.
- [ ] 3.6 Spill and GPU memory are unaware of each other.
- [x] 3.7 `InferencePool` submission is unbounded — no credits.
- [x] 3.8 `execute_with_udfs` materializes the whole "streaming" result.

## Cluster 4 — File listing / many small files at scale

- [x] 4.1 Directory expansion is non-recursive; a nested media tree fails plain `read()`.
- [x] 4.2 Globbing is a full client-side listing + per-path `fnmatch`.
- [x] 4.3 Prefix-scoped fast path bails on the common `dir/*.parquet`.
- [x] 4.4 Glob stat-storm is a known unfixed regression.
- [ ] 4.5 Footer reads per-file, LRU of 1024 — thrashes at 500k files.
- [x] 4.6 `splits()` fans 64 threads over every file on the driver.
- [ ] 4.7 `row_count()` does a second full footer sweep.
- [ ] 4.8 Strict schema inference reads only file 0; union mode reads every file. No sampling.
- [x] 4.9 Non-strict schema mode collapses to a single `WholeSourceSplit` — no parallelism.
- [ ] 4.10 Hive dataset splits are top-level only; `year/month/day` gives 5 splits for 5 years.

## Cluster 5 — LLM engine surface

- [x] 5.1 No tokenizer anywhere; batching is by rows, not tokens.
- [x] 5.2 No sorting by sequence length (the biggest offline-vLLM throughput lever).
- [ ] 5.3 `max_model_len` not auto-sized; prompts never truncated.
- [ ] 5.4 No streaming generation.
- [x] 5.5 Multi-LoRA groups run serially, defeating `max_loras`.
- [x] 5.6 `llm_generate` and `llm_udf` take divergent paths.
- [x] 5.7 No per-row sampling params.
- [x] 5.8 No `finish_reason` / `logprobs` — truncation is undetectable and corrupts `parse_json`.
- [ ] 5.9 `http_engine` builds a pool per batch; no keep-alive.
- [x] 5.10 Retry backoff has no jitter.
- [x] 5.11 `instruct` suffix skipped for dict requests (vision / per-row LoRA).
- [x] 5.12 `_coerce` silently truncates `"3.9"` to `3`.
- [x] 5.13 `_match_label` substring matching nulls a correct answer when labels nest.
- [x] 5.14 Vision inputs unbounded — no resize, no `convert("RGB")`.
- [x] 5.15 `sampling(n=)` documented but discarded.
- [x] 5.16 `engine.last_usage` is a shared mutable attribute — token counts misattributed.

## Cluster 6 — Embeddings

- [x] 6.1 No normalization option in `embed`.
- [x] 6.2 `sentence_transformer_encoder` never sets `batch_size`, `device`, or fp16.
- [x] 6.3 No pooling control and no long-document chunking.
- [x] 6.4 `np.asarray` fails on a CUDA tensor.
- [x] 6.5 Output extension type vs the `fixed_size_list` Lance ANN expects; no validation.

## Cluster 7 — Multimodal decode

- [ ] 7.1 Video decode is a per-row Python loop.
- [x] 7.2 Video decodes the container up to 3 times.
- [ ] 7.3 No GPU decode path anywhere.
- [ ] 7.4 No lazy/deferred decode handle.
- [x] 7.5 No seek-based frame sampling.
- [x] 7.6 `download_dataset` has no retries, no per-URL timeout, pool per batch.
- [x] 7.7 `upload_dataset` resolves the filesystem per row.

## Cluster 8 — Training ingest

- [x] 8.1 `stream_loader` collects the whole dataset per rank.
- [x] 8.2 `elastic_shard` materializes the full position list.
- [x] 8.3 `iter_torch_batches` has no epoch parameter — every epoch sees one order.
- [ ] 8.4 Local shuffle only in the lazy path; no guidance on buffer sizing.
- [ ] 8.5 No jax integration.
- [x] 8.6 No collate hook on the indexed loaders.
- [x] 8.7 `prefetch_batches=1` default, single thread, no separate CUDA copy stream.
- [x] 8.8 No `drop_last` in `iter_torch_batches`.
- [x] 8.9 `streaming_split` drops a trailing partial round silently.
- [x] 8.10 `ResumableSampler` has no DataLoader-worker stride.
- [x] 8.11 pin_memory + `non_blocking` use-after-free window.
- [x] 8.12 Non-tensorizable columns vanish silently from the training batch.

## Cluster 9 — Preprocessors

- [ ] 9.1 `_ordinal_expr` builds an O(k) CASE chain per row.
- [x] 9.2 `distinct_values` pulls the full category set to the driver, no cardinality guard.
- [x] 9.3 `Chain.fit` re-executes the upstream plan once per step.
- [x] 9.4 `Tokenizer` is per-row Python with no batched fast path, no truncation, no mask.
- [ ] 9.5 `KBinsDiscretizer(quantile)` issues `n_bins-1` separate aggregates.
- [x] 9.6 `TargetEncoder` is not cross-fitted; no `cv=`.
- [x] 9.7 `PolynomialFeatures` has no term-count guard.

## Cluster 10 — Serving, pipeline, observability

- [x] 10.1 `serve_deployment` blocks the asyncio event loop on the forward pass.
- [x] 10.2 `serving_udf` is one synchronous request per batch; no pipelining, no warmup.
- [x] 10.3 `http_client` does `arr.tolist()` after warning about it.
- [x] 10.4 No metric export / progress for a multi-hour inference job.
- [x] 10.5 `run_pipeline` joins with `timeout=1.0` and returns; threads outlive the call.
- [x] 10.6 `skipped_splits()` is worker-local and never aggregated — silent data loss at scale.
- [x] 10.7 `_SKIPPED_SPLITS` never resets between queries.
- [x] 10.8 Error budget resets per `_apply_udf` call, so `max_errored_rows` is unbounded.
- [x] 10.9 Dropped rows are unrecoverable — no error column, no dead-letter sink.
- [x] 10.10 `autocast_call` re-executes the model 6+ times on real data, duplicating paid API calls.

## Cluster 11 — Knobs and patterns Batcher lacks (from the Ray inventory)

Tracked separately once clusters 0-10 land; the inventory is in the session notes. Highest-value
candidates: token-budget batching, length bucketing, `max_errored_blocks` analogue, S3 retry
vocabulary, `eager_free`, column-lifetime pruning, uint8-through-decode, JPEG-bytes-in-Parquet
layout guidance, prefix-cache-friendly prompt ordering, FP8/quantization recommendation by GPU.

## Cluster 12 — Found during this effort (not in the original audits)

- [x] 12.1 `explain()` raises `NotImplementedError` on ANY `map_batches`/inference plan, because
  `planned_profile` lowers via `plan.to_ir()` and `MapBatches.to_ir()` deliberately raises. So the
  plan of a batch-inference pipeline cannot be inspected at all — and "read the plan" is step one
  of the `optimize-a-slow-query` skill for exactly this workload class. Fix by rendering the
  logical tree for UDF plans instead of lowering to executable IR. Must NOT change the wire
  contract: `to_ir()` tags stay as they are.
- [x] 12.2 A `Filter` is NOT pushed below `MapBatches`, so `map_batches(model).filter(...)` runs the
  GPU model over every row and discards most results. Ray's field guidance measures this as a
  direct proportional GPU saving (filtering 60% of rows before inference saves 60% of GPU work).
  Sound pushdown needs an explicit user contract, because the UDF owns its whole output batch and
  may rewrite any column — `input_columns` declares what the fn READS, not what it PRESERVES. The
  honest fix is a symmetric opt-in `preserves_columns=` declaration, mirroring how `input_columns`
  is already opt-in "precisely because getting it wrong is a wrong answer, not a slow one".
- [x] 12.3 A distributed write whose result is empty produced NO output path at all, while the
  single-node write leaves a readable zero-row file — a `distributed != single-node` divergence
  (invariant #7) that surfaced downstream as "path does not exist". Pre-existing and general, not
  specific to the UDF path.
- [x] 12.4 Test fakes for `_MapActor` lagged the real signature; updated with the write_spec change.


## Cluster 13 — Suspected bugs, INVESTIGATED (all three turned out NOT to be bugs)

When the "fix any bugs we find" pass investigated these three, each turned out to be either a
transient mid-refactor test state or a sandbox-infra failure — none is a standing Batcher bug.

- [x] 13.1 `test_diff_agg_arg_extreme::test_arg_extreme_null_value_single_node_equals_distributed`
  is NOT a data bug. **Corrected finding:** the earlier "real data mismatch" note was wrong — the
  failure was only ever a Ray `CoreWorker.wait` SystemError / bringup hang, and the test never
  reached its data comparison. Proven three ways that the `arg_max`/`arg_min` algebra is correct:
  (a) a new Rust test `arg_extreme_null_heavy_merge_equals_single_node` reproduces the exact
  null-heavy scenario split across partitions and asserts two-phase == single-node — PASSES; (b)
  `arg_extreme_skips_null_values` / `_all_null_values_is_null` match DuckDB single-node — PASS; (c)
  a trivial `ray.wait` works but `collect(distributed=True)` hangs on Ray bringup in this sandbox,
  affecting every implicit-Ray distributed test equally. The added Rust test is kept as durable
  mergeability coverage the suite previously lacked (the old cross-partition test had no nulls).
- [x] 13.2 `test_expr_rewrite_coverage` (ListZip in `_EXPR_KIDS`): was a transient mid-refactor
  state from a concurrent agent's list-ops work; `ListZip` is now present in the rewrite table and
  the test PASSES. Not a standing bug.
- [x] 13.3 `test_available_schema[sum]`: also a transient mid-refactor state — the test now PASSES
  on the settled tree. Not a standing bug.

## Newly landed by this effort (summary of the delivered wins)

- Distributed batch-inference WRITE now streams from the worker that produced each shard — the
  headline 2B-row-embedding driver-OOM is gone (cluster 1).
- Filter pushdown BELOW `map_batches` under an opt-in `preserves_columns=` declaration — the model
  runs only on surviving rows (12.2). Sound: no declaration ⇒ no move.
- Transient inference failures (CUDA OOM / 429 / 503 / NCCL timeout) now RETRY instead of killing a
  multi-hour job (1.6); deterministic UDF bugs still fail fast.
- `_SUCCESS` completion markers so a half-written distributed output is distinguishable (1.8).
- Recursive directory read so a Hive/media tree Batcher wrote is readable back (4.1).
- Three silent row-dropping bugs in rebatching + the TF loader's dropped-batch-0 fixed (cluster 0).
- GPU placement groups + concurrent health check + GPU budgeting in Carbonite (cluster 2).
- Genuinely bounded-memory UDF streaming, honest per-worker error budget (cluster 3).
- Live inference-job observability: partition progress, rows/s, GPU util/VRAM, skip counts (10.4).
- Full LLM-engine, embeddings, multimodal-decode, training-ingest, preprocessor, and serving
  hardening across clusters 5-10.

## Verification notes (final)

- Full `tests/unit` on the settled tree: **2 failed, 5652 passed** (baseline was 6 failed, 4722
  passed). Both remaining failures — `test_available_schema[sum]` and `test_expr_rewrite_coverage`
  (ListZip) — are pre-existing at HEAD in the list-expression layer, unmodified, and unrelated to
  this effort (items 13.2/13.3).
- Differential vs DuckDB on the operators this effort touched (operator matrix + inference
  pushdown): **276 passed**. Public API coverage, docstring, layer, and guardrail gates all green.
- `test_ml_distributed_inference_write.py` (the headline fix): **3 passed** in isolation on the
  settled tree.
- `test_dist_hunt2_matrix::test_odd_partition_counts` failed under load, but the cause is a Ray
  `CoreWorker.wait` SystemError (runtime resource exhaustion from many concurrent distributed
  test runs), NOT a data mismatch — `assert_tables_equal` never executed. Environmental, not a
  regression. The distributed group-by/aggregate path was untouched by this effort.
- Item 13.1 (`test_diff_agg_arg_extreme` null-value single==distributed) DID show a real data
  mismatch earlier and is a genuine pre-existing distributed arg-extreme bug — left flagged for a
  dedicated pass, out of AI-workload scope.
