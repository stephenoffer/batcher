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

Shrink, then bisect. Never debug a wrong answer at full scale.

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
- A hang under distributed is usually **credit starvation**: a consumer stopped draining,
  so producers block at 0 credits and never abort. `flow_control.default_credits` is 16
  batch slots, ceiling `default_credits x credit_ceiling_factor` (64), also clamped by
  `credit_byte_budget` (256 MiB) / `execution.morsel_bytes`. Raising credits masks the
  symptom; find the stalled consumer.
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
- `docs/user-guide/troubleshooting.md`, `docs/user-guide/explain-plans.md`
- `docs/internals/carbonite.md` — spill, credits, admission
- `optimize-a-slow-query` — when the query is correct but slow
