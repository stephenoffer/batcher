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
    it from a working check. What it found on its first run, all since fixed:

    - three SQL differential tests comparing an outermost-`ORDER BY` result with
      `assert_same`, which sorts both sides;
    - two `parametrize`s over a directory walk with nothing asserting it found anything —
      `tests/docs/test_examples.py` collects **510 executed example scripts** that way, and
      a moved directory turns all 510 into zero tests with the run still green;
    - seven `examples/` scripts running to completion asserting nothing, one of which
      printed `None` three times under the heading "each node summarizes its neighbours";
    - one test turning an engine failure into `pytest.skip` behind a bare
      `except Exception`, hiding 48 of its own 98 cases — and `lint-skips` reads
      module-level guards by design, so a mid-body skip is invisible to it;
    - two assertions that a token is *absent* from `explain()` output, with nothing
      anywhere proving that token ever appears.

The last shape is the one that decays without anyone touching it. When `plan/profile/render/`
turned `explain()` into a table with a header and tree glyphs, a helper that parsed it with
`line.strip().split()[0]` began returning `['query', '────', 'OPERATOR', 'sort', '└─']`.
One exact-equality assertion failed loudly, which is how it was found.

**An earlier revision of this paragraph claimed the `assert "sort" not in _ops(ds)` beside it
had become a tautology. That was asserted without being checked, and it is false.** The
broken parser still returned the *root* operator's name correctly — only *nested* operators
collapsed to `'└─'` — and the sort in that test is the root when it is present at all. Run
both ways, the assertion passes on a source whose `sorted_by` metadata makes the sort
eliminable and fails on one without it, so it discriminated exactly the two states it exists
to tell apart.

Keep the rule and correct the reasoning. The hazard was **latent, not realized**: the same
helper feeding an assertion about any operator *below* the root would have silently become a
tautology, because no nested operator's name survived the parse. That is the case a positive
control catches. So an absence assertion needs one — something showing the token appears when
it should — or it is only a claim about the renderer. And a claimed instance of a test that
cannot fail is itself a claim to be run rather than argued: this one was written into a rules
file on nothing but a plausible reading of the parser.

### Two things the audit tooling deliberately will not do

**It will not flag a plan shape.** Three "redundant-looking" constructs were raised in one
day and all three were deliberate, including a `lambda:` that existed purely to defer a name
lookup and whose removal by a ruff autofix broke `import batcher` for four sessions.
Apparent redundancy is not evidence of redundancy, and a rule that fires on shape rather
than behaviour manufactures regressions in proportion to how automatic it is.

**It will not report a cause it inferred.** A controlled change — move one variable, hold
the rest, watch the behaviour move — earns a *dependency* claim, not a *mechanism* claim.
"An interposed `Project` blocks predicate pushdown" was backed by exactly such a change and
was still wrong about the mechanism: pushdown works, and the filter came from
`runtime_join_filter` in the last phase with nothing left to sink it. Acting on the
inference would have modified correct code and left the real bug in place. An over-read
experiment is more dangerous than an unread one, because it arrives with evidence attached.

### A knob that moves is not a knob that tests what you think

`collect(num_partitions=N)` is a **spill** knob. It is the obvious reach when you want to
show an operator gives the same answer however the work is divided, and it does not show
that.

Measured here: varying it from 1 to 8 visibly changes physical execution -- a spilled
300,000-row sort came back in 61 batches against 72, with the timing to match -- and it did
not move a `LIMIT` over an unordered `group_by` at all. Not at 200 rows, not at 300,000
(both sides of `MIN_ROWS_TO_SHARD`, which is 4 morsels = 65,536), with spill on or off, from
an in-memory source or four Parquet files. That shape is precisely the one
`.claude/rules/python-control-plane.md` records as diverging between a single-node run and a
two-worker one over four Parquet files: groups 0, 1, 2 against 3, 5, 8.

So a test asserting "the same result at 1, 4 and 8 partitions" is a true statement about
spilling and says nothing about distribution, while reading exactly like a distributed
equivalence test. The only thing that tests single-node == distributed is `distributed=True`
against a real cluster, which CI cannot do and `just lint-skips` prices.

Two further traps sit on either side of this one, and both were walked into in the session
that wrote this entry:

- **Below `MIN_ROWS_TO_SHARD` nothing shards at all**, so a small fixture makes *every*
  parallelism knob inert and every such comparison vacuous. A 200-row fixture proves
  nothing about parallel execution, and the first reading of the measurement above -- "the
  lever is inert" -- was wrong for exactly that reason before it was re-run at 300,000.
- **A sweep over many operations at once inherits the weakness of its lever.** 32 graph
  algorithms were swept for "determinism across partition counts" and all 32 agreed, which
  looked like a strong result and was worth nothing: one inert lever, 32 vacuous rows. The
  breadth made it more convincing, not more valid.

The general form: before trusting a comparison across a setting, show the setting changes
something you can see. If you cannot make the *un*fixed version of the system fail the test,
the test is not measuring the setting.

### `distributed=True` on its own is the same trap, one layer in

The entry above sends you from `num_partitions` to "use `distributed=True` against a real
cluster". That is necessary and it is not sufficient, because **`collect(distributed=True)`
with no `num_workers` defaults to one worker**, and one worker computes what single-node
computes. A matrix compared across that flag agrees for the reason an unplugged lever agrees.

Measured 2026-09-22 on a local Ray 2.58 cluster, four Parquet files, 200,000 rows, using the
divergence `.claude/rules/python-control-plane.md` already records -- `LIMIT 3` over an
unordered `group_by`:

| Run | Groups returned |
|---|---|
| `collect(distributed=False)` | 0, 1, 2 |
| `collect()` (the `'auto'` default) | 0, 1, 2 |
| `collect(distributed=True)` | 0, 1, 2 |
| `collect(distributed=True, num_workers=2)` | 3, 5, 8 |

The fourth row reproduces that file's measured numbers exactly, which is what makes it a
usable control. The first three are the same run three ways. Note the second: `distributed`
defaults to `'auto'`, so a "single-node" baseline written as a bare `collect()` is not
pinned to single-node either -- name `distributed=False` rather than letting the default
decide what you are comparing against.

This was not hypothetical. A verb matrix run across the bare flag reported seven verbs
agreeing, and `Dataset.split` among them; re-run with `num_workers=2`, the same matrix showed
`split` raising `PlanError`, because a global window that is then filtered has no distributed
plan. The vacuous run was the one that looked like a clean pass.

So: a distributed-equivalence test names `distributed=False` on one side and
`distributed=True, num_workers=N` with `N >= 2` on the other, and carries a control that is
known to diverge. 230 of the ~290 `distributed=True` call sites under `tests/integration/`
already pass `num_workers`; the remainder are worth reading before they are trusted.

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
