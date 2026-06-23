# Rule: Testing (Everything Is Tested, Correctness Before Speed)

Batcher's claim is to be faster *and* correct than DuckDB/Spark/Polars/Ray Data.
That is only credible if correctness is mechanically proven against an oracle on
every change. Tests are not optional follow-up — they are part of the change.

## The oracles

Batcher has two correctness oracles. Use them; don't invent ad-hoc assertions.

1. **DuckDB (Python differential).** For any relational behavior — operators,
   expressions, SQL, optimizer rewrites — the result MUST match DuckDB on the same
   input. Harness: `tests/differential/conftest.py::assert_same` (order-independent,
   type-tolerant multiset comparison; tolerates int↔float, Decimal→float, float
   rounding). Add cases next to the existing `test_diff_*.py` files.
2. **The Tier-0 interpreter (Rust).** `bc-interp::execute` (sequential) is the
   reference. The parallel executor and the JIT MUST agree with it bit-for-bit on
   supported inputs. New `bc-runtime` primitives get a Rust unit test asserting the
   mergeable invariant: `combine_finalize(partition(partial(pₖ)))` == single-node.

## Hard gates per change type

- **New / changed relational operator or expression** → MUST add a **differential
  test vs DuckDB** covering it (incl. nulls, empties, type edges), AND keep the
  Rust seq == par == JIT agreement green. Touching the JSON IR → add a round-trip
  test that the Python `to_ir()` shape deserializes in Rust.
- **New `bc-runtime` primitive** → Rust `#[cfg(test)]` unit test + the mergeability
  invariant test. If it's stateful, test it spilled/partitioned too.
- **New Kyber pass / cost or cardinality change** → unit test in `tests/unit/`
  proving the rewrite is *semantics-preserving* (plan changes, result doesn't), PLUS
  a differential test showing the optimized query still matches DuckDB. Plan-shape
  assertions (e.g. predicate pushed below join) go in `tests/unit/`.
- **Layer/import change** → `just lint-layers` MUST stay green (independence +
  `plan` neutrality contracts).
- **Distributed path** → an equivalence test that single-node and multi-partition
  execution produce identical results (see `tests/integration/test_distributed.py`,
  `test_flight_shuffle.py`, `test_spilling.py`).

## Test layout & markers

```
tests/unit/          fast, no native engine (optimizer passes, IR validation, cost)
tests/differential/  cross-check results vs DuckDB/Polars (the correctness spine)
tests/integration/   end-to-end: I/O, adaptive re-opt, distributed, spilling, UDFs
```

Pytest markers (declare them): `unit`, `differential`, `integration`, `property`.
Property tests (`hypothesis`) are encouraged for algebraic invariants
(merge associativity, encode/decode round-trips, optimizer idempotence).

## Correctness before timing

The benchmark harness refuses to time a query whose result doesn't match the oracle
(`benchmarks/harness.py`, `FLOAT_ATOL`/`FLOAT_RTOL`). Apply the same discipline
everywhere: never report or optimize for speed on a path whose correctness isn't
proven first. A fast wrong answer is a bug.

## Coverage philosophy

- Cover the **contract**, not the implementation: every operator × {empty input,
  nulls, single row, multi-batch, type boundaries}.
- A bug fix lands **with a regression test** that fails before the fix.
- Don't delete or weaken a differential test to make a change pass — if Batcher and
  DuckDB legitimately differ, that is a decision to surface explicitly, not to hide.

## Gate before "done"

`just test` runs the CI sequence: `check → test-rust → build → test-py`. Add
`just lint-layers` and (for Python changes) `just lint-py`. See `/run-quality-gate`.
