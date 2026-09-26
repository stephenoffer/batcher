# Rule: Testing (Everything Is Tested, Correctness Before Speed)

Batcher's claim is to be faster *and* correct than DuckDB/Spark/Polars.
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

## The two gates, and the difference between them

Both are mechanical and both are blocking. They catch adjacent failures and it is worth
knowing which is which, because the second is the one that survives review.

`just lint-tests`
    A test that **cannot fail**: an ordered result compared with an order-independent
    helper, an assertion true by construction (`assert len(x) >= 0`), a test that asserts
    nothing at all.

`just lint-methodology`
    A test, example or benchmark that **can** fail but has been arranged so that it does
    not — so it reports a property nobody checked. Everything it finds runs real code,
    takes real time, and turns green, which is why reading the output cannot distinguish
    it from a working check. It flags an outermost-`ORDER BY` result compared with
    `assert_same`; a `parametrize` over a directory walk with nothing asserting the walk
    found anything; an `examples/` script that runs to completion asserting nothing; an
    engine failure turned into `pytest.skip` behind a bare `except Exception` (invisible to
    `lint-skips`, which reads module-level guards); and a token asserted *absent* from
    `explain()` output with nothing proving the token ever appears.

An absence assertion (`token not in explain()`) needs a positive control showing the token
appears when it should; without one it is a claim about the renderer. The hazard is that it
decays without anyone touching it: a change to the `explain()` rendering (`plan/profile/render/`)
can make the token unreachable, and the assertion keeps passing.

### Two things the audit tooling deliberately will not do

**It will not flag a plan shape.** Apparent redundancy is not evidence of redundancy — a
`lambda:` that looks removable may exist purely to defer a name lookup — and a rule that fires
on shape rather than behaviour manufactures regressions in proportion to how automatic it is.

**It will not report a cause it inferred.** A controlled change — move one variable, hold
the rest, watch the behaviour move — earns a *dependency* claim, not a *mechanism* claim.
Acting on an inferred mechanism can modify correct code and leave the real bug in place. An
over-read experiment is more dangerous than an unread one, because it arrives with evidence
attached.

### A knob that moves is not a knob that tests what you think

`collect(num_partitions=N)` is a **spill** knob. It visibly changes physical execution (batch
counts, timing of a spilled sort), and it does not change how work is distributed: a `LIMIT`
over an unordered `group_by` — the shape `.claude/rules/python-control-plane.md` records as
diverging between single-node and two workers — gives the same rows at every partition count.
So a test asserting "the same result at 1, 4 and 8 partitions" is a true statement about
spilling and says nothing about distribution, while reading exactly like a distributed
equivalence test. The only thing that tests single-node == distributed is `distributed=True`
against a real cluster, which CI cannot do and `just lint-skips` prices.

Two further traps sit on either side of this one:

- **Below `MIN_ROWS_TO_SHARD` nothing shards at all** (4 morsels = 65,536 rows), so a small
  fixture makes *every* parallelism knob inert and every such comparison vacuous.
- **A sweep over many operations at once inherits the weakness of its lever.** Thirty-two
  operations agreeing across an inert lever is thirty-two vacuous rows; the breadth makes it
  more convincing, not more valid.

The general form: before trusting a comparison across a setting, show the setting changes
something you can see. If you cannot make the *un*fixed version of the system fail the test,
the test is not measuring the setting.

### `distributed=True` on its own is the same trap, one layer in

**`collect(distributed=True)` with no `num_workers` defaults to one worker**, and one worker
computes what single-node computes. A matrix compared across that flag agrees for the reason an
unplugged lever agrees. Likewise `distributed` defaults to `'auto'`, so a "single-node" baseline
written as a bare `collect()` is not pinned to single-node either.

So: a distributed-equivalence test names `distributed=False` on one side and
`distributed=True, num_workers=N` with `N >= 2` on the other, and carries a control that is
known to diverge — `LIMIT 3` over an unordered `group_by` on a multi-file Parquet source is one.
Read any `distributed=True` call site under `tests/integration/` that omits `num_workers`
before trusting it.

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

The canonical gate matrix is in `CLAUDE.md` — run the rows your change touches. `just test`
runs the whole CI sequence (`check → test-rust → build → test-py → cov-gate`) if you want it
in one command. The `/run-quality-gate` skill triages failures.
