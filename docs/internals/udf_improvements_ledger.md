# UDF improvements ledger

A running record of improvements, fixes, and features that make Batcher better at running
**user-supplied UDFs and callables** — `map_batches` / `map` / `flat_map`, class-based model
UDFs, the batch-inference and multimodal-preprocessing plane, and the threads/processes/GPU
scheduling beneath them. The moat this serves is ML/LLM inference: a UDF here routinely calls
a flaky external service, loads a model once per worker, or runs a forward pass that must not
stall or OOM the whole query.

Each entry is a distinct, tested change. Entries are numbered `U<n>` continuously and never
reused, so the count is a count of *distinct* improvements. Category tags:
**bug** (wrong result / data loss), **robustness** (crash/leak/hang on a reachable path),
**feature** (new capability or Ray Data / Daft parity gap closed), **validation**
(fail-fast on bad input), **perf**, **test** (coverage that pins a contract), **docs**.

This page is not published (`exclude_patterns` in `docs/conf.py`); it is a working index for
anyone touching the UDF / inference plane (`core/udf/`, `api/dataset/ml.py`, `dist/executors/map.py`).

---

## Resilience: retry and timeout for flaky/external UDFs (`core/udf/resilience.py`)

A flaky `fn` — an LLM API call, a vector-DB upsert, a model that intermittently OOMs — used
to fail the whole query on the first transient error, or hang it forever on a stuck call. A
new resilience layer wraps the raw per-batch call *outside* the error-budget bisection, so a
transient error is retried first and only a failure that survives every retry is charged to
`max_errored_rows`. It preserves the `batch -> result` signature, so every scheduling path
(sequential, threads, the streaming stages, the GPU autobatch pool) gets it by wrapping once.

| # | Cat | Improvement |
|---|-----|-------------|
| U1 | feature | `max_retries` — a batch whose `fn` raises a retryable error is retried up to N times with exponential backoff before the failure propagates. The universal LLM-API / external-service pattern (`429`, connection reset), previously absent: any transient killed the job. |
| U2 | feature | `retry_on` — restricts retries to specific exception type(s) (a single type or a tuple), so a real bug (a `TypeError` from a schema mismatch) fails fast on the first call instead of burning the retry budget and delaying the failure. Empty = retry any `Exception`. |
| U3 | feature | `retry_backoff` — configurable base backoff; attempt `k` waits `retry_backoff * 2**k` seconds, capped at 30s so a large `max_retries` cannot sleep for minutes (`0.5 * 2**12` is 34 minutes) waiting out an outage rather than riding out a transient. |
| U4 | feature | `timeout` — a per-batch wall-clock ceiling; a call exceeding it raises `TimeoutError` instead of hanging the query on a stuck external call. Runs the call on a dedicated thread and abandons it on timeout (Python cannot preempt a running call), documented as a stall-guard, not a hard kill. |
| U5 | robustness | A `timeout` the user opted into is always retryable, even when `retry_on` restricts the set to specific transient types — `TimeoutError` is added to the retry set unless it is already covered — so a timed-out call is retried rather than propagating past a narrow `retry_on`. |
| U6 | feature | Retry/timeout apply uniformly on the materializing (`execute._apply_udf`), streaming (`stream._apply_udf_stream`), and GPU-autobatch (`execute._apply_udf_autobatch`) paths — a `fn` behaves identically whichever shape the plan routes it through. |
| U7 | robustness | Retry/timeout is layered strictly *outside* the GPU OOM-halving and the `max_errored_rows` failure bisection, so a permanent per-row failure is retried to exhaustion and only then isolated/charged — the two knobs (`max_retries` for transients, `max_errored_rows` for dirty rows) compose without double-counting. |
| U8 | validation | `_normalize_retry` rejects a negative `timeout`/`max_retries`/`retry_backoff` and a `retry_on` that is not an exception type (or tuple of them) with an eager `PlanError` at the API edge, turning a deferred worker-side failure into an actionable one. |
| U9 | feature | Every retry is announced on the observability bus (`map_batches` LOG event: attempt, backoff, `fn` name, error type+message) so a running job reports transient failures as they happen. Only the exception type and message are published, never the batch's values (which may be the malformed/sensitive row). |
| U10 | robustness | The retry-event publish is best-effort (wrapped in a bare `except`), so an observability failure can never turn a successfully-retried transient into a hard query failure. |
| U11 | feature | Exposed on both `ds.ml.map_batches` and the top-level `ds.map_batches` sugar, and shipped whole to Ray workers on the frozen `MapBatches` node, so the distributed path gets the same resilience with no extra wiring. |
| U12 | test | `tests/integration/test_udf_resilience.py` — retry-then-succeed, retry-exhausted-propagates, `retry_on` restriction (type and tuple), timeout-raises, timeout-retried-recovers, retry∘error-budget composition, and the validation matrix. |
| U13 | docs | `map_batches` docstrings (ml + frame) document the retry/timeout knobs, their layering with `max_errored_rows`, and the timeout preemption caveat; this ledger records the pass. |

## Async UDFs: concurrent I/O-bound calls on one event loop (`core/udf/async_udf.py`)

An LLM-inference / API-enrichment `fn` is I/O-bound — it awaits a remote service. The thread
path couples concurrency to a thread per in-flight batch; an `async def` `fn` runs many
batches' awaits concurrently on one event loop, bounded by a semaphore — the way you issue
thousands of concurrent LLM requests without thousands of threads. Detection is from the
callable, so a user opts in just by writing `async def`; the synchronous path is untouched.

| # | Cat | Improvement |
|---|-----|-------------|
| U14 | feature | `is_async_udf` detects a coroutine `fn` — an `async def`, directly or via a class instance's `async def __call__` — so a class model UDF and a plain function are both recognized, and the resolved per-batch callable is what is inspected (matching how the sync path calls it). |
| U15 | feature | `run_async_batches` dispatches the batches concurrently on one event loop under an `asyncio.Semaphore`, so an I/O-bound stage overlaps its awaits (the Ray Data async-`map_batches` model) instead of serializing on a thread pool. Verified to hit exactly the concurrency cap and to beat the serial sum. |
| U16 | feature | Results are returned in input-batch order regardless of completion order (`asyncio.gather`), so async concurrency never reorders a relation — the per-batch contract holds. |
| U17 | feature | `max_concurrency` bounds in-flight coroutines (default 32), so a slow tail cannot unbounded-buffer the input and a remote service is not hammered past its rate limit by default. |
| U18 | feature | On the async path `timeout` is a *real* cancel (`asyncio.wait_for` cancels the pending coroutine at its next await), unlike the thread-based timeout for a synchronous `fn` which can only stop waiting on a call it cannot preempt. |
| U19 | feature | The async runner reuses the same retry policy (backoff, `retry_on`, `max_retries`) as the sync path, so an async `fn` gets identical resilience semantics — a transient is retried, a timeout is retryable, a real bug fails fast. |
| U20 | feature | `batch_format` (numpy/pandas/torch) conversion happens around the await, so an async `fn` can take a framework object just like a sync one; the engine boundary stays Arrow. |
| U21 | robustness | An async stage is excluded from the synchronous stage-overlap streaming path (`stream_eligible`), which would otherwise hand the stage an un-awaited coroutine; it stays on the materializing path that routes it to the event loop. Async wins over the GPU-autobatch and multiprocessing routes in `_apply_udf` for the same reason. |
| U22 | validation | `max_concurrency` is validated (`>= 0`) with an eager `PlanError` at the API edge. |
| U23 | test | `tests/integration/test_udf_async.py` — transform+order, class UDF, concurrency-cap, overlap-beats-serial, timeout-cancels, retry-recovers, numpy format, empty input. |
| U24 | perf | Fixed a double model-load the async check introduced: `is_async_udf` inspects the *unbuilt* `op.fn` (a class's `async def __call__` is visible without instantiating), so a class model is not loaded once just to check async-ness and again to run — the build happens exactly once per call, in the branch that runs it. |

## Class-UDF lifecycle: deterministic teardown (`core/udf/execute.py::teardown_udf`)

A class `fn` is a load-once factory: `__init__` loads the model, `__call__` scores each batch.
Until now the instance was only ever reclaimed by the garbage collector, so a GPU allocation /
HTTP session / DB connection lingered past the stage. A class may now define `close()`, called
when the call that built the instance owns it.

| # | Cat | Improvement |
|---|-----|-------------|
| U25 | feature | A class UDF's optional `close()` is called at stage end, giving a load-once model a deterministic place to free a GPU allocation / connection / session instead of waiting for GC — a lifecycle hook neither Ray Data actors nor Daft UDFs offer. |
| U26 | robustness | Teardown is gated by *ownership*: `close()` runs only when `op.fn` was a class this path built. A prebuilt instance passed in by a long-lived streaming loop or distributed actor is that owner's to tear down at its own lifetime end, so the model is never closed out from under a reuse. |
| U27 | robustness | Teardown runs in a `finally`, so the model is released even when the `fn` raises mid-stage; and `close()` is best-effort (a failing teardown is swallowed) so it cannot fail a query whose results are already produced. |
| U28 | feature | Applied across every owned path — the sync/threads path (extracted to `_run_sync_udf`), the async runner, and the GPU autobatch pool — each building once and closing once. |
| U29 | test | `tests/integration/test_udf_lifecycle.py` — load-once-close-once, close-raises-is-swallowed, no-close-is-fine, prebuilt-instance-not-closed, teardown-survives-a-raising-fn. |

## Validation and ergonomics (`api/dataset/ml.py`)

Turn the two most common UDF foot-guns into an eager, actionable `PlanError` at the API edge
instead of an opaque failure deep in a worker.

| # | Cat | Improvement |
|---|-----|-------------|
| U30 | validation | A non-callable `fn` (an int, a string, a bare object) is rejected with a clear message naming the expected shape, before any work starts, instead of a `TypeError` surfacing per batch inside a worker. |
| U31 | validation | A class `fn` whose instances are not callable — a model class that forgot `def __call__(self, batch)` — is rejected by name, the exact mistake stated ("its instances are not callable"), instead of failing once loaded per worker. The check walks the MRO (excluding `object`) so an inherited `__call__` is correctly accepted. |
| U32 | validation | `map` and `flat_map` validate the *user's* row `fn` before wrapping it in the callable row adapter, so a non-callable per-row `fn` is caught at the edge rather than slipping through the adapter. |
| U33 | test | `tests/integration/test_udf_validation.py` — non-callable matrix, class-without-`__call__`, class-with-`__call__`, inherited-`__call__`, and the `map`/`flat_map` inner-fn checks. |

## Structure: the UDF plane split into cohesive modules (`core/udf/`)

The new capabilities grew `execute.py` past the 500-line module limit. Rather than allowlist an
over-limit file, the per-stage application logic was extracted so each module has one job.

| # | Cat | Improvement |
|---|-----|-------------|
| U34 | maintainability | Extracted `core/udf/lifecycle.py` (`build_udf_callable` + `teardown_udf`) — the two ends of a model instance's life, depending on nothing else in `core.udf`, so it sits below every module that uses it and breaks what would be an import cycle between `execute`, `apply`, `stream`, and `strategy`. |
| U35 | maintainability | Extracted `core/udf/apply.py` (`apply_udf`, `rechunk`, and the sync/async/autobatch runners) — *how* one stage runs — leaving `execute.py` (266 lines) as purely the plan-tree walk. Both are back under the 500-line limit with no allowlist. |
| U36 | maintainability | Repointed the moved-name couplings (the `execute._rechunk` / `execute._apply_udf` / `execute._apply_udf_autobatch` references in `tests/unit/` and the doc comments in `stream`/`async_udf`) so nothing dangles at the old home, and `stream`/`strategy` now import the lifecycle helpers from their true module. Layer-independence and the `core.udf` façade import path are unchanged. |

## Async per-row `map` / `flat_map` (`api/dataset/callbacks.py`)

The async story extended from whole-batch `map_batches` down to the per-row `map`/`flat_map`
callbacks — the per-row LLM/API-enrichment pattern, where each row is its own remote call.

| # | Cat | Improvement |
|---|-----|-------------|
| U37 | feature | `map` / `flat_map` accept an ``async def`` row `fn`; a batch's rows are awaited concurrently on one event loop (`_AsyncRowMap` / `_AsyncRowFlatMap`) instead of one at a time — issuing many concurrent per-row requests without a thread per row. |
| U38 | feature | The async row adapter runs its event loop *inside* a synchronous `__call__`, so it rides the existing thread path as a plain batch UDF: batches parallelize across `num_workers` while each batch overlaps its rows. No new scheduling path. |
| U39 | feature | `max_concurrency` on `map`/`flat_map` bounds the in-flight per-row awaits within a batch (default 32), so a per-row API call cannot fire an unbounded fan-out or trip a rate limit. Verified to hold exactly at the cap. |
| U40 | feature | Per-row output order is preserved across the concurrent awaits (`asyncio.gather`), and the one-to-many `flat_map` variant flattens in row order — same contract as the synchronous adapters. |
| U41 | test | Added async `map`/`flat_map` cases to `tests/integration/test_udf_async.py` — transform+order, one-to-many flatten, and the within-batch concurrency cap. |

## More validation and ergonomics (`api/dataset/ml.py`)

| # | Cat | Improvement |
|---|-----|-------------|
| U42 | validation | `output_columns` is checked at the API edge for empty/non-string names and duplicates, naming the offender — instead of the opaque Arrow schema error (or a silently shadowed column) a duplicate/blank name otherwise causes deep in the engine. |
| U43 | validation | An ``async def`` `fn` passed with `multiprocessing=True` now warns (`PerformanceWarning`) that multiprocessing is ignored — async runs on one event loop and never uses the process pool, so the user's intent was being silently dropped. |
| U44 | test | `tests/integration/test_udf_validation.py` gains the bad-`output_columns` matrix and the async+multiprocessing warning check. |
| U45 | robustness | The async path now honors `max_errored_rows`: a batch that fails after retries is bisected to isolate and drop the offending row(s) against the shared per-worker budget (`_resilient_batch`, the async twin of `call._resilient_call`), the halves awaited concurrently. Previously an async `fn`'s failure ignored the budget and crashed the query — a silent gap between the async and sync dirty-data contracts, now closed. |
| U46 | test | `tests/integration/test_udf_async.py` — async row-isolation drops the bad rows and keeps the rest; async budget-exhaustion still fails fast. |
| U47 | feature | The top-level `ds.map` / `ds.flat_map` sugar now forwards `batch_size`, `num_workers`, and `max_concurrency` to `ds.ml.map`/`flat_map`, so a per-row async call through the short spelling can tune its concurrency and batching instead of being stuck at the defaults — parity with the `ds.ml` surface. |
| U48 | bug | An async class UDF bound with `fn_constructor_kwargs` (or `fn_kwargs`) was wrapped by `_bind_fn` in a *synchronous* `_BoundModel.__call__`, so `is_async_udf` saw a sync callable, routed it to the sync path, and coerced the un-awaited coroutine as a bad return type (`TypeError: got coroutine`). `_bind_fn` now generates an `async def __call__` when the base class is async, so the model stays async through the wrapper. The synchronous class path is unchanged. |
| U49 | test | `tests/integration/test_udf_async.py` — async class UDF with `fn_constructor_kwargs`, and an async plain `fn` with `fn_kwargs`, both produce the right result. |

## Composition: `@udf` decorator inherits every knob

| # | Cat | Improvement |
|---|-----|-------------|
| U50 | test | Pinned that the public `@udf(...)` decorator forwards the new resilience knobs (`max_retries`/`timeout`/…) and routes an ``async def`` decorated transform through the concurrent path — the decorator composes with everything above for free because it forwards `**config` to `map_batches`. Regression coverage in `tests/integration/test_callbacks.py`. |
| U51 | bug | An async `map_batches` / `map` driven from **inside a running event loop** (a Jupyter notebook or an async web app — the primary ML environments) crashed with `asyncio.run() cannot be called from a running event loop`. A new `run_coroutine_blocking` detects a running loop and runs the batch/row gather on a fresh loop in a short-lived worker thread, never touching the caller's loop; outside a loop it uses `asyncio.run` directly. Applied to both the whole-batch runner and the per-row `map`/`flat_map` adapter, so async UDFs work in Jupyter. |
| U52 | test | `tests/integration/test_udf_async.py` drives an async `map_batches` and an async `map` from within `asyncio.run(...)` and asserts both return the right rows rather than raising. |
| U53 | validation | An ``async def`` `fn` with `num_gpus > 0` now warns that GPU auto-batching and autocast are skipped on the event-loop path (async is for I/O-bound work, not a GPU forward pass) — the same surface-the-ignored-intent treatment as async+multiprocessing, both folded into one `_warn_async_combos`. |
| U54 | robustness | Verified the new node fields (`timeout_s`/`max_retries`/`retry_on`/`max_concurrency`) survive `pickle` unchanged, so they ship intact to Ray workers and the distributed path gets the same resilience/async behavior as single-node. |
| U55 | bug | `output_columns=[]` was silently accepted and stored as a non-None `()`, so `available_columns()` reported zero columns to the plan while the `fn` actually kept the input schema — a plan/execution mismatch that could mislead projection/lineage downstream. It is now rejected with a message pointing at `None` (the real "unchanged" spelling). |
