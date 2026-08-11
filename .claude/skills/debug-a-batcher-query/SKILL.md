---
name: debug-a-batcher-query
description: Diagnose a Batcher query that raises, returns wrong results, hangs, or OOMs — a triage tree by symptom, the typed-exception map, the shrink-and-bisect procedure against the DuckDB oracle, and how to isolate an optimizer, spill, or distributed-mergeability bug. Invoke when a query misbehaves, not when a correct query is merely slow (that is optimize-a-slow-query).
---

# Debug a Batcher query

Work the tree by *symptom*. Do not start by reading engine source — start by making the
failure small and reproducible. Most reports resolve in section A or B.

First, three facts that save an hour:

- A `Dataset` is **lazy**. A transformation that "did nothing" did nothing on purpose;
  work happens at a terminal op (`collect`, `to_pydict`, `iter_batches`, `count`,
  `show`, `write`). A traceback pointing at `collect()` may originate anywhere upstream.
- `ds.profile()` is a **data-quality column profiler** (null counts, approx distinct) —
  it is *not* a performance profiler. The measurement tools are `ds.explain(analyze=True)`
  and `ds.stats()`.
- Confirm you are on a release build: `bt.versions()["engine_profile"]`. A debug build is
  ~10x slower and can look like a hang. Rebuild with `just build-release`.

## A. It raises

Every Batcher error descends from `BatcherError` (`python/batcher/_internal/errors.py`).
The class tells you which layer failed, which tells you where to look.

| Exception | Layer | Typical cause / first move |
|---|---|---|
| `PlanError` | `plan`/`api`, build time | Unknown column, a `str` where an `Expr` belongs, schema mismatch. Check `ds.columns` at each step. |
| `ConfigError` | `config` | A tunable out of range or inconsistent; raised by `Config.validate()`. Read the message — it names the field. |
| `OptimizationError` | Kyber | The optimizer could not produce a valid physical plan. Go to section E. |
| `ResourceError` | Carbonite | The request was infeasible. Go to section C. |
| `BackpressureAbort` (`ResourceError`) | Carbonite | Backpressure could not be relieved — a stuck consumer or too-tight credits. Section C. |
| `ExecutionError` | Core / engine | An operator failed at runtime. Section B/C. |
| `BackendError` (`ExecutionError`) | Core | A backend failed; wraps the real error — read `__cause__`. Also raised by `ds.stats()` on `map_batches`/ML pipelines. |
| `CompileError` (`ExecutionError`) | `bc-codegen` | JIT compile failed. The interpreter is still a fallback, so this surfacing at all is a bug worth reporting. |
| `TransportError` | `bc-transport` | Flight/shared-memory failure. Section D. |
| `RetryableShuffleError` / `FatalShuffleError` | `bc-transport` | Retryable = worker loss (recompute+retry is automatic). Fatal = decode/protocol/auth; retrying cannot help. |
| `IOError` | `io` | Read/write/list/open failed — including a missing optional extra (e.g. `s3://` without the `cloud` extra). |
| `FormatError` / `SchemaError` / `CommitError` (`IOError`) | `io` | Unsupported or malformed format / irreconcilable schemas across files / atomic-commit conflict. |
| `DataQualityError` | `api` | A `ds.dq...fail()` expectation failed; `.violations` carries per-constraint counts. |
| `AccessDeniedError` | `governance` | No `SELECT` privilege; carries `.table` and `.columns`, never values. |

`PerformanceWarning` and `SecurityWarning` are `UserWarning`s, not errors — the query is
correct; the warning points at a better spelling. Run under `python -W error::UserWarning`
to make them stop the run.

## B. Wrong results

**First, if the symptom is "fewer rows than expected", ask whether the engine dropped them
on purpose.** It is the one wrong answer that shrinking cannot reproduce, because the rows
that vanish are the malformed ones and a shrunk input rarely contains any. Three tolerance
flags delete rows by design, and all three report it:

```python
from batcher.observe import metrics
metrics.start_metrics()
ds.to_pydict()
snap = metrics.metrics_snapshot()["skipped"]
snap["total"]                     # whole inputs dropped by on_error="skip"
snap["malformed_rows_total"]      # rows dropped by on_bad_lines= or max_errored_rows=
snap["malformed_rows_by_source"]  # {"csv": 12, "map_batches": 3} — which one did it
```

A non-zero `total` also means the row count is short by an *unmeasured* amount: an
unreadable file's row count is exactly what could not be read. `source.corrupt_files()`
names the paths. Only once all three read zero is the answer wrong for a reason worth
bisecting.

Then shrink, then bisect. Never debug a wrong answer at full scale.

1. **Shrink the input** until the wrong row still appears — target tens of rows, in a
   literal `bt.from_pydict({...})`. Deterministic and pasteable is the goal.
2. **Bisect the chain.** Collect after each step and eyeball where the row count or a
   value first goes wrong. Bisecting the *plan* beats reading the engine.
3. **Ask the oracle.** DuckDB decides. Batcher is wrong until proven otherwise.

```python
import duckdb, batcher as bt
from conftest import assert_same, assert_same_ordered   # tests/differential/conftest.py

con = duckdb.connect()
con.register("t", table)                       # `duck_materialize` instead for FLOAT+NaN
assert_same(bt.from_arrow(table).filter(bt.col("k") > 1).collect(), con.sql("SELECT * FROM t WHERE k > 1"))
```

`assert_same` is order-*independent*: it **cannot see a sort bug**. For anything with an
`ORDER BY`/`LIMIT` contract use `assert_same_ordered`. To compare Batcher against
*itself* across execution paths, use `assert_tables_equal(actual, expected, ordered=...)`.
Numeric tolerance is 9 decimal places; integers are compared exactly; NaN equals NaN.

4. **Cover the cross-product.** A green suite means nothing if nothing combined a
   non-default flag with a non-default path. Run the operator across
   `{collect(), collect(spill=True), iter_batches(), distributed}` x
   `{nulls, empty, one row, duplicates, -0.0/NaN, descending}`:

```python
base   = ds.collect()
spill  = ds.collect(spill=True)
stream = pa.Table.from_batches(list(ds.iter_batches())) if base.num_rows else base.slice(0, 0)
assert_tables_equal(spill,  base)
assert_tables_equal(stream, base)
```

`tests/differential/test_diff_operator_matrix.py` already *is* this cross-product (48
operator shapes x `base`/`empty`/`single`, plus sort over every
`descending` x `nulls_first`). Extend it rather than writing a one-off. Its docstring
names the four real bugs it exists for — spilled descending sort emitting nulls
mid-result, a `-0.0`/`0.0` float key splitting one group into two, a shuffled nullable
key splitting groups across reducers, and a keyless aggregate over empty input returning
0 rows from `iter_batches()` but 1 from `collect()`.

```bash
pytest tests/differential/test_diff_operator_matrix.py -x -q
pytest -m differential
```

## C. It hangs or OOMs

Memory lives on `MemoryConfig`; backpressure on `FlowControlConfig`.

```python
from batcher.config import Config, MemoryConfig, config_context
with config_context(Config().replace(memory=MemoryConfig(max_memory_bytes=2 << 30))):
    out = ds.collect(spill=True)
```

- `memory.max_memory_bytes` (default `None` = auto-sensed; `default_total_bytes` 8 GiB
  fallback). `soft_limit` 0.85 throttles, `hard_limit` 0.90 spills. The effective budget
  is `max_memory_bytes x hard_limit`.
- **Spill is off by default** — `collect(spill=True)` is the switch for aggregation, join
  and sort. `memory.spill_dir` picks the scratch path; `spill_compression` defaults to
  `"auto"`; `spill_bucket_max_bytes` is 128 MiB compressed.
- `memory.unbounded_memory=True` opts *out* of spilling — a way to confirm spill itself
  is implicated, not a fix.
- **A distributed hang: read the log before touching credits.** There are two hangs and they
  look identical from outside. The first is *scheduling*: the tasks were never placed, because
  they ask for more than any node has or because another job holds the cluster. The engine says
  so after two minutes on the map and inference barrier, and immediately when a placement group
  times out — `no node can host one task: this stage asks for 32 CPU per task and the widest of
  8 node(s) is short on CPU (16 available, 32 needed)`, or `the cluster is short of free
  capacity: 16 outstanding at 8 CPU each … 15 CPU free between them`. Neither is fixed by
  anything on this page: the first needs a smaller grant, the second needs the co-tenant to
  finish. `ray status` corroborates both. Check also whether the fleet came up narrower than it
  asked for (`shuffle fleet came up narrower than requested`) and which cluster the process
  actually attached to (`attached to Ray … nodes=1` on a job you meant to distribute is the
  whole answer).
- The second is **credit starvation**, and only then are the knobs below relevant: a consumer
  stopped draining, so producers block at 0 credits and never abort.
  `flow_control.default_credits` is 16 batch slots, ceiling
  `default_credits x credit_ceiling_factor` (64), also clamped by `credit_byte_budget`
  (256 MiB) / `execution.morsel_bytes`. Raising credits masks the symptom; find the stalled
  consumer.
- `BackpressureAbort` is the *good* outcome — it means the deadlock was detected.

Turn on logging before guessing. There is **no `RUST_LOG` or `BATCHER_DEBUG`** in this
repo; the Rust tracing level is driven from the same `log_level`:

```bash
BATCHER_OBSERVABILITY_LOG_LEVEL=DEBUG python repro.py     # env: BATCHER_<SECTION>_<FIELD>
```

Loggers are `batcher`, `batcher.kyber`, `batcher.carbonite`, `batcher.core`, `batcher.io`,
`batcher.api`. `observability.log_format="json"` and `log_file=...` for a durable trace.

## D. Single-node and distributed disagree

That is a **mergeability bug**, and it is a hard-invariant violation: one implementation
(`partial -> combine -> finalize`) must serve one core and many machines. `combine` must
be associative *and* commutative. The usual culprits are a key whose identity differs
between the local and shuffled path (float `-0.0` vs `0.0`, nullable ints), or state that
finalizes correctly only when it sees every row at once.

```python
one  = ds.collect(distributed=False)
many = ds.collect(distributed=True, num_workers=4, num_partitions=16)
assert_tables_equal(many, one)          # ordered=True if the query has an ORDER BY
```

Vary `num_partitions` (1 / 4 / 64) — a bug that appears only above 1 partition is in
`combine`; one that appears only at high partition counts is usually key derivation.
Confirm against the in-repo suites, which are selected **by path** (they carry no
`integration` marker):

```bash
pytest tests/integration/test_distributed.py -x -q      # *_matches_single_node family
pytest tests/integration/test_spilling.py -x -q         # *_matches_in_memory family
pytest tests/integration/test_flight_shuffle.py -x -q   # credit-bounded transport
just bench-dist                                         # equivalence + timing
```

`config.distributed.transport` (`"auto"`/`"flight"`/`"disk"`) — if `disk` agrees and
`flight` does not, the bug is in transport, not the algebra.

## E. Isolating an optimizer bug

If the unoptimized shape is right and the optimized one is wrong, a Kyber rule broke
semantics. Inspect the plan and the IR:

```python
print(ds.explain())                                  # est≈N (exact | learned | default)
print(ds.explain(analyze=True))                      # runs it: actual vs est, ms, cpu, backend, decisions
ir = ds._plan.to_ir()                                # logical JSON IR (internal handle)

import json
prof = json.loads(ds.explain(analyze=True, format="json"))
prof["logical_ir"], prof["optimized_ir"], prof["decisions"], prof["adaptive_stages"]
```

`format` accepts only `"text"` and `"json"`. Diff `logical_ir` against `optimized_ir` —
the offending rewrite is usually visible as one moved or dropped node.

**There is no global "disable the optimizer" flag and no per-rule toggle** in `config` or
on `collect()`. Do not cite one. The real seams are:

```python
from batcher.kyber import Optimizer
baseline = Optimizer(rules=[]).optimize(ds._plan).ir      # no rules at all
full     = Optimizer().optimize(ds._plan).ir
```

Bisect by passing a shrinking `rules=` list until the wrong IR appears — that names the
rule. Narrower kill switches, all real: `collect(adaptive=False)` (stage-boundary
re-optimization), `execution.fuse_linear=False` (pins the operator-at-a-time path),
`execution.streaming=False` (forces the materializing executor — an explicit bisecting
escape hatch, not a tuning knob), `optimizer.plan_cache_entries=0` (disables plan
memoization, use when a query is right on first run and wrong after).

## F. A GPU stage is slow, and the plan looks right

On an accelerator fleet the usual suspects are not in the plan at all. Four of them produce
*no error* — the query returns the right rows, slower — so they are invisible unless you look:

| Symptom | Check | Cause |
|---|---|---|
| One stage far slower than an identical earlier run | `bt.accelerators()` -> `devices[].throttled` | The driver is clamping a device: thermal (cooling has failed) or power (your own cap doing its job). A clamped device is derated, not removed. |
| Wrong numbers from an otherwise deterministic stage | `devices[].ecc_uncorrected` | Uncorrectable ECC: a tensor read back is not the tensor written. Enable `accelerator.health` and the device is quarantined. |
| Devices idle, stage wall-clock long | `batcher.ml.devices.device_feed_advice()` | The pipeline is starving them. The lever is upstream (prefetch, batch size, fewer devices), not a faster kernel. |
| Fan-out smaller than the cluster's GPU count | `accelerator.energy.power_budget_watts` | Carbonite clamped the grant to what the budget can power. `carbonite.accel.validate_fleet_power` reports the counter-offer. |
| A multi-device collective slower than expected | the `dist` log line naming `world_size` and `widest_domain` | The collective is wider than any NVLink domain the fleet has, so its all-reduce left the fast path. |
| A stage queueing on an apparently idle fleet | `fabric.residency_report(...)` | A residency rule narrowed the fleet to one region's nodes. The report gives the before/after device counts. |

`bt.measure_energy()` around the pipeline plus `observe.format_energy_report` turns most of
this into one table: per-stage joules, utilization, the idle share, and whether each figure was
measured or modelled. An idle share above roughly a third means the fleet is under-fed, which
is a pipeline problem wearing a hardware costume.

## If the bug is in the engine

Land the **regression test first, and watch it fail** — the differential/matrix case or
the Rust unit test — then fix. Never weaken or delete a differential test to go green: a
legitimate Batcher/DuckDB difference is a decision to surface, not to hide.

Then verify with `/run-quality-gate`, and pick the contributor skill that matches the
layer: `add-relational-operator`, `add-expression-or-function`,
`add-kyber-optimizer-pass`, `add-distributed-operator`.

## See also

- `.claude/rules/testing.md` — the oracles and the per-change test gates
- `.claude/rules/rust-engine.md` — seq == par == JIT, mergeable algebra
- `docs/user-guide/operate/running/troubleshooting.md`, `docs/user-guide/operate/tuning/explain-plans.md`
- `docs/architecture/internals/carbonite.md` — spill, credits, admission
- `optimize-a-slow-query` — when the query is correct but slow
- `docs/user-guide/operate/running/gpu-fleets.md` — the power, fabric, health, and residency controls behind
  section F, and how to turn each on
