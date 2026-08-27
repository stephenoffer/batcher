"""Reward hacking: a gate that reports success it did not measure.

`tools/audit/testing.py` catches the test that *cannot fail*. This file catches the
adjacent, harder failure — the test, example or benchmark that **can** fail but has been
arranged so that it does not, and so reports a property nobody checked. The distinction
matters because the second kind survives review: every one of these runs real code, takes
real time, and turns green.

Each rule below is a shape that has already produced a false green in a data engine, and
each is calibrated to a small, hand-verified finding set on this tree, so
`tools/lint_methodology.py` can gate on them rather than merely report.

``order-blind-sql``
    A query whose *outermost* clause is `ORDER BY`, compared with `assert_same` — which
    sorts both sides before comparing and therefore cannot see order at all. This is the
    same hole `harness/order.py` closed on the benchmark side, where it was worth closing
    because "an engine that skipped the sort passes the gate and is then timed on the work
    it did not do". The suite side had no equivalent. `testing.py::order-blind-test` sees
    only the *DataFrame* spelling (`ds.sort(...)`); a SQL string asks for the same
    guarantee and went unchecked.

``xfail-not-strict``
    `pytest.mark.xfail` without `strict=True`. A non-strict xfail is green whether the bug
    is still there or not, so the day the bug is fixed nothing says so and the test stays
    quarantined forever — while a test that starts passing *for the wrong reason* is
    equally invisible. `strict=True` turns "unexpectedly passed" back into a failure, which
    is the only thing that ever retires an xfail.

``skip-launders-engine-failure``
    `pytest.skip()` reached from a broad `except Exception` that neither names an optional
    third-party backend nor re-raises. Standing down because a backend is absent is correct
    and this suite does it everywhere; standing down because *anything at all* went wrong
    converts a regression into a silent loss of coverage — the test stops running, the run
    still says green, and `tools/lint_skips.py` cannot see it (it reads module-level guards,
    by design).

    The first version of this rule additionally required the `try` body to call `bt.` or
    `batcher.`, which was the wrong test and missed the worst instance:
    `test_prop_optimizer_result_invariance` reaches the engine through a local
    `run_with_rules(...)` helper, so the rule saw no engine call — while **36 of that test's
    81 pairs** stood down behind a bare `except Exception`, on the one property that says an
    optimizer rule never changes an answer. What makes such a handler safe is that it
    re-raises or fails for the case it does not recognise, which is the check now used.

``discovered-parametrize``
    `@pytest.mark.parametrize` over a helper that *discovers* its own cases by walking the
    tree (`rglob`, `iterdir`, ...), with nothing asserting it found any. An empty list
    collects zero tests and reports green — so a helper that stops finding its inputs (a
    moved directory, a renamed glob) deletes a whole suite with no failure anywhere.
    `tests/docs/test_examples.py` is the live example: `_example_files()` returns `[]` when
    `examples/` is absent, and 510 executed examples become zero silently.

    The rule is deliberately narrow. Flagging every *computed* parametrize argument
    produced 516 findings here, almost all inline lambda lists that cannot be empty by
    accident — a detector at that false-positive rate does not get triaged, it gets
    ignored.

``example-without-assertion``
    A script under `examples/` that runs to completion asserting nothing. The suite runs
    every example (`tests/docs/test_examples.py`) precisely so a demonstrated workflow
    cannot rot — but an example with no assertion only proves the API still *exists*, not
    that it still returns the right answer, while counting as a passing test.

``uncontrolled-negative-render``
    ``assert "<token>" not in <something>.explain()`` where **no test anywhere** asserts that
    the same token *does* appear in a rendering. A negative substring assertion against
    human-facing output has no positive control: nothing proves the token would have shown up
    had the thing under test gone wrong, so the day the renderer stops spelling it that way
    the assertion silently becomes a tautology and keeps passing.

    This is not hypothetical here. `plan/profile/render/` turned `explain()` into a table with
    a header and `└─` tree glyphs, and a helper in `test_parquet_sortedness.py` that parsed it
    with ``line.strip().split()[0]`` began returning ``['query', '────', 'OPERATOR', ...]``.
    One exact-equality assertion failed loudly; the ``assert "sort" not in _ops(ds)`` beside it
    kept passing, and would from then on have passed whether or not the sort was eliminated.

    The fix is a positive control — one test asserting the token appears when it should — which
    is why the rule looks suite-wide rather than per-file.

``benchmark-order-unchecked``
    A `compare()` call in `benchmarks/` that does not pass `ordered_by`. `run.py` passes
    each case's, so the sort check applies; a second call site that drops it re-opens the
    hole `harness/order.py` exists to close, for the cases routed through it.

``differential-without-oracle``
    A file under `tests/differential/` that neither compares against an oracle nor declares
    one. The directory name is a claim — the contract calls it "the correctness spine" — so
    a file here asserting only hand-written values is a unit test wearing a differential
    test's name. A *declared* alternative is accepted, because several are legitimate
    (`zoneinfo` for timezones, the standard library's codecs for compression, the Tier-0
    interpreter for path equivalence); silence is not.

``benchmark-order-unregistered``
    A case registered with no `ordered_by` when it should have one — either by constructing
    a `Case` by hand (which bypasses `Suite`, whose job is deriving the keys from the query)
    or by a native `.case(...)` in a module that carries an ordered query. This is where the
    check was actually lost rather than at the call site: TPC-H built its `Case`s directly,
    so **all 22 carried `ordered_by=()` while 18 end in an `ORDER BY`**, and the sort gate
    had never run on the project's flagship benchmark. The scan suite lost its three top-N
    cases the second way — `Shape` carries its SQL, and the registration dropped it.

``benchmark-unguarded-build``
    A standalone timing entry point under `benchmarks/` that never calls
    `require_release_build`. **60 of 64 of them**, measured — so nearly every number the
    tree can produce outside `run.py` could come from a debug build, which its own guard
    puts at 8-60x slower. Ratcheted rather than gated, because 60 findings would make this
    file permanently red and a permanently-red gate is one everybody learns to walk past.

``rust-ignore-without-reason``
    A `#[ignore]` in the Rust crates carrying no explanation. Python's equivalent is already
    ratcheted at zero by `lint_skips`; Rust had nothing, and `cargo test`'s "N ignored" sits
    in a line everyone reads as success. Every ignored test here is genuinely a timing study,
    but that is only knowable by reading each body — and a bare `#[ignore]` looks exactly
    like a test quarantined because it broke.

``single-sample-timing``
    An assertion comparing two *measured durations* where neither is a repeated-sample
    estimate. One sample of each arm on a box this repo documents as routinely carrying
    three concurrent sessions measures the neighbour's load, not the property — so the test
    passes or fails by coin-flip and, being green most of the time, reads as a guarantee.

    Contention can only ever *add* time, so `min` over a handful of runs is the estimator
    that survives a shared machine. Two live examples:
    `test_hive_partition_write.py::test_cost_does_not_grow_with_the_partition_count` failed
    in a full run and passed standalone on the same commit; and
    `test_observe_hardening.py::test_progress_events_do_not_slow_a_stream_measurably`
    allowed the instrumented arm 20x the baseline **plus 0.5 seconds** against arms
    measuring ~0.0005s, which no input could fail.

``patch-target-not-bound``
    A `monkeypatch.setattr(<module>, "name", ...)` whose target module does not bind `name`
    at module level. The patch applies to something nothing reads, so the test goes on
    running the real code — taking real time, exercising the real path, and passing.
    `.claude/rules/concurrent-agents.md` records the class (a patch on `module.attr` becomes
    a no-op when `attr` moves to `module.sub`, "and the test keeps passing while testing
    nothing"); nothing checked it. `raising=False` is what makes it silent: without it pytest
    raises `AttributeError` and the test fails loudly, which needs no rule.

    The two instances that motivated it fail in **opposite** directions, which is worth
    stating because a reader who has seen one will assume the failure mode is one-sided.
    A spy on the *package* `dist.spill_breakers` missed `execute_spilling_sort`'s call to the
    module-global in `spill_breakers/sort.py`, so operators that spilled perfectly reported
    "never spilled". A spy on the *submodule* `dist.global_window.disk` missed `_dispatch`'s
    function-local import from `batcher.dist.global_window`, which resolves the package
    attribute through a PEP 562 hook — so an operator that ran correctly reported "reached
    nothing". Both assertions were correctly formed and pointed at nothing.

    Two live instances, both fixed: `test_autoscale_wait.py` patched
    `scaling._ray_initialized`, a name that has never existed (the engine calls
    `ray.is_initialized()` directly) — entirely inert while looking load-bearing, and the
    proof was that removing it changed nothing. `test_accelerator_observability.py` patched
    `diagnosis.device_window`, which `diagnosis.py` imports inside a function so never binds;
    the line above it patched `series.device_window`, which was the effective one.

## Calibrating a rule: two ways the measurement lies

Every rule here is calibrated by running it and reading the findings, so the calibration is
itself an experiment and can fail the same way its subjects do. Two shapes have already
produced a wrong number on this tree, and they need different defences.

**A wrong-shaped sample.** The experiment measures the right thing on inputs that cannot
exhibit the defect. A tightening of `assert_same` to positional column comparison was declared
safe on 2,334 green SQL differential tests, not one of which was a `SELECT *` over a join —
where Batcher follows SQL:2016 §7.7 and DuckDB does not, so the full suite then returned 49
failures against Batcher for being correct. A helper that read its input handle twice shipped
correct for all three of its callers, every one of which passed a DuckDB *relation*; a cursor
is consumed by the first read and answers `None`. A control built to show that a swapped value
was invisible without a unique key had a unique key in it. The defence is to check that the
sample *can* exhibit what you are looking for before believing it does not.

**An instrument that formats the answer away.** Rarer, harder to see, and not a sampling
problem at all: the experiment is correctly shaped and the tool you read it through omits the
category you were measuring. A sandbox was reported as *not* staging `.github` on the strength
of an `ls` — which hides dotfiles — and the conclusion drawn was that a required dependency
could be dropped. `ls` answered a question adjacent to the one asked and looked like it had
answered the one asked. The same class: a `| head -N` that truncates before the interesting
row, `grep` without `-a` on a file with a NUL byte, `git diff --stat` showing files where the
question was hunks, `grep -c "^FAILED"` on a log whose summary has not printed yet, `tail`
supplying a pipeline's exit status. And the sharpest instance, because it was written *into*
the guidance warning about this class: `rg -l ... | xargs rg -l ...`, where `rg` is a shell
function from the Claude Code snapshot rather than a binary — so `xargs` cannot exec it, the
error goes to stderr, and the pipeline **exits 0**, printing an empty list and reporting
success. The regeneration step certified the absence it could not see.

The general form is worth stating plainly, because no
amount of re-sampling catches it: **a tool that formats away a category of its input is not a
witness to that category's absence.**

**The largest instance in this repo is not a shell tool at all — it is the comparator the
correctness spine runs on.** `tests/_harness.py::assert_same` compares rows as a sorted
multiset and columns as a *set*, so it formats away both row order and column order. Measured:
**2,105 call sites across 402 test files**, against 337 uses of the three order-aware
alternatives the same harness provides (`assert_same_ordered`, `assert_same_for_query`,
`assert_tables_equal(ordered=True)`).

**That is not an indictment of `assert_same`, and reading it as one would be the wrong
lesson.** Order-independence is correct and deliberate at the overwhelming majority of those
sites: a query with no `ORDER BY` promises a multiset, and demanding an order it never owed
would fail on rewrites that are perfectly sound. The defect is only ever *reaching for it where
the property under test is the one it discards* — and the tree has the alternatives, so this is
a choice at each call site rather than a gap in the harness. `CLAUDE.md` states the specific
rule ("never assert a sort with an order-independent comparison") forcefully; what it does not
say is that this is one instance of a class, and the class is what generalises.

Two of this file's own findings are the evidence rather than the argument. A window
column-order divergence and an optimizer rule that **transposed** `available_columns()` both
survived because the comparator sorted the property away — and `test_diff_relational_windows.py`
shows how ordinary the mistake is: it already uses `assert_same_ordered` for its top-N cases
and `assert_same` for its window ones, in one file, deliberately, with a docstring explaining
the split. The author reasoned correctly about *rows* and the column comparison is set-based
too. "Ordered" has two axes, and choosing the loose comparator on purpose for one of them does
not mean you chose it for the other.

The defence for the second is not a bigger sample, it is naming the instrument's blind spot
before reading its output — or reading the same fact through a second tool that formats
differently. Both errors above were caught by someone re-running a number they had been told
was already measured, which is the cheapest available check and the one most discouraged by
being handed a calibration.

## Rules this file deliberately does NOT have

There is no detector for a *suspicious plan shape* — a stacked `Filter(Filter(x))`, a
projection that looks redundant — and the omission is load-bearing. Three such things were
raised on this tree in one day and **all three were deliberate**: the filter stack is
`split_expensive_filter`'s output, because the data plane's `AND` evaluates both operands
over every row with no short-circuit, so splitting lets the expensive conjunct see only the
cheap one's survivors; the extra projection was innocent; and a `lambda:` wrapping a bare
function call existed purely to defer a name lookup past module-execution time.

Those are two different failures needing two different defences, and conflating them loses
the second:

**A module that re-exports lazily.** `patch-target-not-bound` declines outright when the
target module defines `__getattr__` (PEP 562), because its attribute set cannot be settled
without importing it, and a rule that guessed would be noisy in precisely the place the
second motivating bug lived — `dist.global_window` is one such module. That matters more
than the coverage it costs: a detector calibrated to zero on a tree that demonstrably
contains the defect *certifies* the absence, which is worse than not having the detector.

**The detector's own scope resolution is the thing to be careful with here, and it went
wrong twice before it worked.** Resolving `monkeypatch.setattr(mod, ...)` means knowing
which module `mod` names, and `ast.walk` has no notion of a scope: the first version walked
the whole file, so two helpers that each did
`from batcher.io.formats.streaming import <different module> as mod` let the later binding
answer for the earlier one's call — reporting a *correct* patch as broken. The second added
per-function alias maps and still walked the module scope with `ast.walk`, which descends
into function bodies, so every call was resolved twice and the false positive came back.
`_own_statements` stops at nested `def`/`class`, which is what finally made the code agree
with the docstring. Reporting a correct thing as broken is the failure that gets a gate
routed around, so it is worth more care than a miss.

**A person inferring intent from structure.** Defence: demand a control before reporting.
The stacked-filter hypothesis was disproved in one experiment — re-run the optimizer with
`split_expensive_filter` removed and the stack collapses — and the reason that experiment
got run at all is that the rule's docstring says its phase placement exists to stop
`merge_adjacent_filters` undoing it. The author was told what would break.

**A tool rewriting structure it cannot interpret.** No control helps here, because nobody
formed a hypothesis to test. Ruff's `PLW0108` ("unnecessary lambda") under `--unsafe-fixes`
removed that deferring lambda, the module then raised `NameError` on import, and four
sessions were blocked until it was reverted. It took a second file the same way
(`kyber/stats/selectivity/patterns.py`, the identical forward-reference shape).

The obvious defence does not exist here, and that was *measured* rather than reasoned:
**`# noqa: PLW0108` cannot be used in this repo at all.** `PLW0108` is not in the selected
rule set, and `RUF100` (unused-noqa) *is* — so a directive naming a non-enabled rule turns
`lint-py` red. Anyone reaching for the local suppression discovers that only after trying
it. So a comment saying what breaks is not the best defence available, it is the only one,
which makes such a comment load-bearing infrastructure rather than documentation.

Generalising: a rule that fires on *shape* rather than on *behaviour* is dangerous in
proportion to how automatic it is, because nobody reviews the third autofix as carefully as
the first — and in any repository running `--unsafe-fixes` over a curated rule set, the rule
can fire where the suppression cannot exist.

## A controlled change earns a dependency claim, not a mechanism claim

The sharpest lesson of the three, because it survives doing the experiment *correctly*.

A finding was reported here as "an interposed `Project` blocks predicate pushdown", backed
by a properly controlled change: interpose the node, hold everything else fixed, watch the
behaviour move. The control was valid and the dependency is real. **The mechanism was
wrong.** `push_filter_through_project` works, and the optimized plan already has the
projection above the filter; the filter at issue is inserted by `runtime_join_filter` in
`Phase.ENFORCE`, the last phase, after every pushdown pass has run — so it lands at the
join's input with nothing left to sink it, and an interposed `Project` strands it.

Moving one variable and watching the output move establishes *that* two things are
connected. It says nothing about the path between them, and in an optimizer the path is
where all the complexity lives.

The cost of over-reading it is not a wasted investigation — it is a **plausible fix aimed at
the wrong component**. "Pushdown cannot cross a `Project`" points at the pushdown rule, which
was correct; acting on it would have modified working code and left the phase-ordering bug
in place. An over-read experiment is more dangerous than an unread one, because it arrives
with evidence attached.

So: a detector may report a *correlation it measured*. It may not report a *cause it
inferred*, however good the experiment was.

## An experiment can run, produce a number, and have measured something else

The two hardest cases on this tree were not unrun checks. They ran, and they were *clean*.

**A sample of the wrong shape carrying the authority of a large one.** Tightening
`assert_same` to compare column names positionally was validated against the SQL slice of
the suite: **2,334 tests, all passing unchanged**. The conclusion drawn — that column order
always agrees, so checking it is free — was wrong. The sample was 2,334 explicit select
lists and contained no `SELECT *` over a join, which is the only shape where the engines
differ. The full suite then failed 49 join tests, and *Batcher was the correct one* (SQL:2016
§7.7 puts a `USING` join's coalesced key first; DuckDB does not). A positional oracle would
have recorded 49 failures against Batcher for conforming to the standard.

**A negative control with nothing controlling it.** A check built to show that a particular
corruption was invisible returned a confident result *contradicting* the reviewer — and it
was vacuous: the arm meant to carry no key had a unique key in it, so the arm demonstrating
"the corruption cannot be seen" was quietly testing a case where it could. It ran, produced
output, and tested nothing. That is worse than a check that fails to detect, because a
contrary result from a control is exactly the evidence one stops looking after.

So a negative control needs its own control, and the operational form is a four-line table:

    the corruption actually corrupted the input       <- else the arm is vacuous
    the checker misses it                             <- the hole being demonstrated
    the checker flags the corrupted copy              <- when it should
    the checker PASSES the correct copy               <- the line most often skipped

The fourth is the one that turns a catch into evidence: a checker that returns "violation"
unconditionally would satisfy the third and be worthless.

## A gate two parties can pass by both doing nothing

A third failure kind, distinct from the two above and the most quietly total.

`benchmarks/scenarios/formats/read.py` gated on `len(counts) == 1` — every engine's row count
agreeing. Every engine reading **zero** rows agrees perfectly, and a near-instant read then
reports as a very fast one. A wrong path, a truncated write, an empty glob: none raises.
`streaming_throughput.py` compounded it, comparing `{key: sum}` maps where `{} == {}` is
`True`, and computing `rows/sec` by dividing `args.rows` — a **constant from the command
line** — by elapsed time. An engine that drained nothing reported full throughput, correct.

It generalises past benchmarks: a rate metric that stops matching reports `0.0`, which is
exactly what clean data looks like. No exception, no odd number, nothing to notice. The
seven prompt-safety metrics in `plan/functions/metrics/safety/` fail that way, which is why
`tests/unit/test_safety_metric_precision.py` pins a *negative* case for each — a detector is
only characterized by both halves.

The fix in each case is a positive control the gate cannot satisfy vacuously: an
independently-computed expected value, not agreement between two parties who may both be
idle.

## The 2x2 that locates a regression instead of describing it

When a tightened check starts failing beside a changed implementation, the plausible story is
"the tightening surfaced an existing near-miss" — and it is plausible enough that the natural
next step is to loosen the check. Running four arms instead of two costs almost nothing and
settles it:

    new check + new implementation     FAIL
    old check + new implementation     FAIL     <- the implementation, visible either way
    new check + old implementation     PASS     <- the check is clean

That third row is the one usually skipped, because the first two already tell a story. Here
it inverted the conclusion: a tightened float comparison was *not* surfacing an old near-miss,
it was catching a **new** variance regression, and without the third row the correct response
(revert the implementation) and the tempting one (loosen the check) are indistinguishable.

## "Agrees with the oracle more often" is not a correctness metric

A variance recurrence was changed here partly on the evidence that exact agreement with
DuckDB rose from **64% to 88%** on random data. That measures whose *rounding* you match, not
which answer is right. Checked against exact rational arithmetic on the shape that mattered —
grouped `var_samp` over `1e12 + (i % 5)` — the form with the better agreement statistic was
**2.5x further from the true value**, and eight times worse at an offset of 1e15. It was
reverted.

So when an oracle and the subject are both approximating, agreement is a similarity measure
between two implementations, and the third thing — exact arithmetic — is the only one that
adjudicates.

## An oracle can disagree with itself

DuckDB 1.5.5 returns `2.0` for `var_samp` over `{2^53, 2^53+2}` through a SQL literal union
and `4.0` for the same values through a registered Arrow table, in one session, with the
values echoing back identically. Two people measured opposite answers an hour apart and both
were right.

`tests/_harness.py::duck_materialize` already exists for a related reason — a registered
Arrow table pushes a filter into the scan, where NaN comparisons follow IEEE rather than
DuckDB's own executor semantics. Treat the binding route as a variable: any differential
finding on a numerically delicate shape should be checked through both before it is believed.

## A divergence from the oracle is evidence about the pair, not about the subject

`compare()` originally had two outcomes, agree or `FAILED`, and that is not enough. Of the
divergences found on this tree, most were the *comparator* being the odd one out — Polars and
Daft dropping rows on TPC-H q6, DuckDB ordering a `USING` join against the standard, DuckDB
raising on malformed JSON where Batcher yields null. Recording those as Batcher failures is
how a correct engine gets "fixed".

`benchmarks/harness/divergences.py` now carries them as a third status, and the interesting
part for this file is that **it is exactly the mechanism someone reaches for to turn a red row
green**, so it is designed against: an entry is refused at import without a verdict, a
substantive reason and a citation; it must match a signature in the actual diff rather than
just the two engine names; and it names the engine that is *the odd one out*, so no entry can
excuse a Batcher defect. Both earlier versions of it failed in the predictable direction —
keyed on an engine *pair*, one silenced every future Batcher-vs-DuckDB mismatch.

If a rule is ever wanted for "an exemption list that can exempt the subject under test", that
module's `_validate` is the worked example of the constraints.

## A detector calibrated to zero on a tree that contains the defect is worse than none

The strongest argument against the two rules above is a *measurement*, not this reasoning.

`quickselect_matches_sorted_oracle` in `bc-runtime/src/agg/median.rs` computed its expected
value by restating the implementation's own interpolation formula inline. It was therefore a
change-detector wearing an oracle's name: it tracked the code, so it could not fail for the
defect it sat over — and the engine disagreed with DuckDB on ~23% of continuous quantiles
while that test stayed green throughout.

A call-graph detector for "the test calls into the module under test to build its
expectation" was then written and run over every `#[cfg(test)]` block in `crates/`. It found
**zero**, because the duplication was an inlined arithmetic expression rather than a call.

That is the outcome to be afraid of. A rule that reports a class clean, on a tree that
demonstrably contains an instance of the class, is worse than not having the rule — it
converts an open question into a settled one, in the wrong direction. The defence that
survives is a convention rather than a check: an oracle comes from outside the module, or
the test says in a comment that it is a change-detector and not an oracle.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from tools.audit.context import ROOT, Context, Finding, _rel

_Func = ast.FunctionDef | ast.AsyncFunctionDef

#: Comparison helpers that sort both sides, so row order cannot reach the assertion.
ORDER_BLIND = {"assert_same", "assert_tables_equal"}

#: Names a fixture/helper takes when it probes for an *optional third-party backend*. A
#: skip out of one of these is the design working, not a laundered failure.
_OPTIONAL_BACKEND_HINTS = (
    "duckdb",
    "deltalake",
    "pyiceberg",
    "iceberg",
    "spatial",
    "azure",
    "pyarrow",
    "torch",
    "cudf",
    "ray",
    "sqlalchemy",
    "driver",
)


def outermost_order_by(sql: str) -> bool:
    """Whether `sql` has an `ORDER BY` at paren depth zero.

    A subquery's `ORDER BY` constrains nothing about the statement's result and every
    engine is free to discard it, so only the outermost one is a promise about row order.
    This mirrors `benchmarks/harness/order.py::order_keys_of`, which makes the same
    distinction for the same reason.
    """
    depth = 0
    low = sql.lower()
    for i, char in enumerate(low):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and low.startswith("order by", i):
            return True
    return False


def _parametrized_strings(fn: _Func) -> dict[str, list[str]]:
    """`{parameter name -> the string literals it takes}` from this test's `parametrize`s.

    Resolving the decorator matters: the queries in this suite are overwhelmingly supplied
    as a parametrize list, so a rule that only reads literals appearing *inside* the
    function body sees almost none of them.
    """
    found: dict[str, list[str]] = {}
    for decorator in fn.decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
            and len(decorator.args) >= 2
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
        ):
            continue
        names = [n.strip() for n in decorator.args[0].value.split(",")]
        values = decorator.args[1]
        if not isinstance(values, (ast.List, ast.Tuple)):
            continue
        for item in values.elts:
            row = (
                item.elts if isinstance(item, (ast.Tuple, ast.List)) and len(names) > 1 else [item]
            )
            for name, element in zip(names, row, strict=False):
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    found.setdefault(name, []).append(element.value)
    return found


def _assigned_strings(fn: _Func) -> dict[str, str]:
    """`{variable -> literal}` for plain `q = "SELECT ..."` bindings inside the test."""
    found: dict[str, str] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
    return found


def _oracle_query(node: ast.expr, params: dict[str, list[str]], local: dict[str, str]) -> list[str]:
    """The SQL text behind a `duck.sql(...)`-shaped argument, as far as it can be resolved.

    Returns every candidate, because one parametrized test covers many queries and any of
    them carrying an outermost order is enough to make the comparison order-blind.
    """
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("sql", "execute", "query")
        and node.args
    ):
        return []
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return [arg.value]
    if isinstance(arg, ast.Name):
        return params.get(arg.id) or ([local[arg.id]] if arg.id in local else [])
    if isinstance(arg, ast.JoinedStr):
        # An f-string's literal parts are enough to see a top-level `ORDER BY`; the
        # interpolations are values, and a value cannot introduce one.
        return ["".join(v.value for v in arg.values if isinstance(v, ast.Constant))]
    return []


def _mentions_optional_backend(node: ast.AST, enclosing: str) -> bool:
    """Whether this handler is plainly probing for an optional third-party backend."""
    text = (ast.unparse(node) + " " + enclosing).lower()
    return any(hint in text for hint in _OPTIONAL_BACKEND_HINTS)


def check_test_module(path: Path, tree: ast.Module) -> Iterator[Finding]:
    """Every methodology finding in one parsed test module."""
    rel = _rel(path)
    discovery = _discovery_helpers(tree)

    for fn in ast.walk(tree):
        if not isinstance(fn, _Func):
            continue

        # --- xfail-not-strict ------------------------------------------------------ #
        for decorator in fn.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            target = call.func if call else decorator
            if not (isinstance(target, ast.Attribute) and target.attr == "xfail"):
                continue
            strict = call and any(
                kw.arg == "strict" and isinstance(kw.value, ast.Constant) and kw.value.value
                for kw in call.keywords
            )
            if not strict:
                yield Finding(
                    "xfail-not-strict",
                    "high",
                    rel,
                    decorator.lineno,
                    f"`{fn.name}` is `xfail` without `strict=True` — it is green whether the "
                    f"bug is still there or not, so nothing will ever say it was fixed",
                )

        if not fn.name.startswith("test_"):
            continue

        params = _parametrized_strings(fn)
        local = _assigned_strings(fn)

        # --- order-blind-sql ------------------------------------------------------- #
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ORDER_BLIND
            ):
                continue
            if any(kw.arg == "ordered" for kw in node.keywords):
                continue
            for arg in node.args:
                ordered = [q for q in _oracle_query(arg, params, local) if outermost_order_by(q)]
                if ordered:
                    yield Finding(
                        "order-blind-sql",
                        "high",
                        rel,
                        node.lineno,
                        f"`{fn.name}` compares the result of a query whose outermost clause is "
                        f"`ORDER BY` using `{node.func.id}`, which sorts both sides — the order "
                        f"the query asked for is never checked "
                        f"({ordered[0].strip().splitlines()[0][:60]!r}) — use "
                        f"`assert_same_ordered`",
                    )
                    break

        # --- discovered-parametrize ------------------------------------------------ #
        for decorator in fn.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
                and len(decorator.args) >= 2
            ):
                continue
            helper = _discovery_helper(decorator.args[1], discovery)
            if helper is None:
                continue
            # A module-level guard that the collection is non-empty is the fix, so a module
            # already carrying one is not a finding.
            if _has_nonempty_guard(tree, helper):
                continue
            yield Finding(
                "discovered-parametrize",
                "high",
                rel,
                decorator.lineno,
                f"`{fn.name}` is parametrized over `{helper}()`, which *discovers* its cases by "
                f"walking the tree — if it ever comes back empty (a moved directory, a renamed "
                f"glob) this file collects zero tests and the run still reports green; assert "
                f"at module level that it found something",
            )

    # --- skip-launders-engine-failure ---------------------------------------------- #
    yield from _laundered_skips(tree, rel)


#: Calls that walk the tree to find their own inputs. A helper built on one of these can
#: come back empty for a reason that is not a test failure anywhere — a directory moved, a
#: glob was renamed, a package stopped importing — and an empty parametrize list collects
#: zero tests and reports green.
_DISCOVERY_CALLS = frozenset({"rglob", "glob", "iterdir", "listdir", "walk", "scandir"})


def _discovery_helpers(tree: ast.Module) -> set[str]:
    """Zero-argument functions in this module that find their results by walking the tree.

    Narrowing to *these* is what makes the rule usable. Flagging every computed
    parametrize argument produced 516 findings on this tree, essentially all of them
    inline `[lambda e: e.audio.resample(0), ...]` lists whose lambda *bodies* happen to
    contain a call — a literal list that cannot be empty by accident and needs no guard.
    """
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, _Func) or node.args.args:
            continue
        if any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in _DISCOVERY_CALLS
            for call in ast.walk(node)
        ):
            found.add(node.name)
    return found


def _discovery_helper(values: ast.expr, discovery: set[str]) -> str | None:
    """The discovery helper a parametrize argument resolves to, unwrapping `sorted`/`list`."""
    node = values
    for _ in range(3):  # `sorted(list(_files()))` is as deep as this ever nests
        if not isinstance(node, ast.Call):
            return None
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name in discovery:
            return name
        if name in ("sorted", "list", "tuple") and node.args:
            node = node.args[0]
            continue
        return None
    return None


def _has_nonempty_guard(tree: ast.Module, helper: str) -> bool:
    """Whether anything in this module proves `helper()` found something.

    Two spellings both count, because both fail loudly on an empty discovery: a
    module-level `assert`, and a *test* that asserts it — which is how
    `tests/docs/test_skill_coverage.py` does it (`test_skills_exist`), and reading only
    module level reported that file's guard as missing.
    """
    for node in tree.body:
        if isinstance(node, ast.Assert) and helper in ast.unparse(node):
            return True
    for fn in ast.walk(tree):
        if not isinstance(fn, _Func) or not fn.name.startswith("test_"):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Assert) and helper in ast.unparse(node.test):
                return True
    return False


def _laundered_skips(tree: ast.Module, rel: str) -> Iterator[Finding]:
    """`pytest.skip()` out of a broad handler wrapped around a call into the engine."""
    enclosing: dict[int, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, _Func):
            for node in ast.walk(fn):
                enclosing[id(node)] = fn.name

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            broad = handler.type is None or (
                isinstance(handler.type, ast.Name)
                and handler.type.id in ("Exception", "BaseException")
            )
            if not broad:
                continue
            skips = [
                call
                for call in ast.walk(handler)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "skip"
            ]
            if not skips:
                continue
            if _mentions_optional_backend(handler, enclosing.get(id(handler), "")):
                continue
            # A handler that also `pytest.fail`s or re-raises is not laundering: it stands
            # down from one recognised case and lets everything else through.
            # `tests/docs/test_doc_examples` is written exactly that way. Requiring an
            # engine call to exclude it was the wrong mechanism — what makes a handler safe
            # is the `fail`, not the identity of the caller.
            if any(
                isinstance(inner, ast.Raise)
                or (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "fail"
                )
                for inner in ast.walk(handler)
            ):
                continue
            name = enclosing.get(id(handler), "<module>")
            yield Finding(
                "skip-launders-engine-failure",
                "high",
                rel,
                handler.lineno,
                f"`{name}` turns a bare `Exception` into `pytest.skip` without naming an "
                f"optional backend or re-raising — so any regression here removes the test "
                f"from the run instead of failing it, and `lint-skips` reads module-level "
                f"guards by design and cannot see a mid-body skip; catch the specific typed "
                f"error, or `pytest.fail` on anything you did not expect",
            )


def check_example(path: Path, tree: ast.Module) -> Iterator[Finding]:
    """An example script that runs but checks nothing."""
    rel = _rel(path)
    if any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
        return
    yield Finding(
        "example-without-assertion",
        "medium",
        rel,
        1,
        "this example is executed by `tests/docs/test_examples.py` and asserts nothing, so it "
        "proves the API still exists but not that it still returns the right answer — while "
        "counting as a passing test",
    )


def check_benchmark(path: Path, tree: ast.Module) -> Iterator[Finding]:
    """The two places a benchmark's sort check can be dropped: the call, and the case."""
    rel = _rel(path)
    module_has_ordered_sql = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and outermost_order_by(node.value)
        for node in ast.walk(tree)
    )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg for kw in node.keywords}

        # (a) the call site — `compare()` reached without the case's keys.
        if isinstance(node.func, ast.Name) and node.func.id == "compare":
            if "ordered_by" not in keywords:
                yield Finding(
                    "benchmark-order-unchecked",
                    "high",
                    rel,
                    node.lineno,
                    "`compare()` without `ordered_by`: the equality gate gets the rows as a "
                    "multiset, so an engine that skipped the query's `ORDER BY` is reported "
                    "correct and then timed on the work it did not do — pass the case's "
                    "`ordered_by`",
                )
            continue

        if not isinstance(node.func, ast.Attribute):
            continue

        # (b) the registration site, which is where this actually went wrong. Constructing a
        # `Case` by hand bypasses `Suite`, whose entire job is deriving the order keys from
        # the query — and TPC-H did exactly that, leaving all 22 cases with `ordered_by=()`
        # while 18 of them end in an `ORDER BY`. The sort gate had never run on the project's
        # flagship benchmark.
        if node.func.attr == "add" and any(
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Name)
            and arg.func.id == "Case"
            and not any(kw.arg == "ordered_by" for kw in arg.keywords)
            for arg in node.args
        ):
            yield Finding(
                "benchmark-order-unregistered",
                "high",
                rel,
                node.lineno,
                "`REGISTRY.add(Case(...))` without `ordered_by` bypasses `Suite`, which "
                "exists to derive the order keys from the query — register through "
                "`Suite.sql` or `Suite.sql_with_builder` so they cannot be dropped",
            )
            continue

        # (c) a *native* case in a module that contains an ordered query. A native builder
        # writes each engine's query itself, so nothing but the registration can tell the
        # harness the result must come back sorted.
        if node.func.attr == "case" and "ordered_by" not in keywords and module_has_ordered_sql:
            yield Finding(
                "benchmark-order-unregistered",
                "medium",
                rel,
                node.lineno,
                "a native `.case(...)` registered with no `ordered_by`, in a module that "
                "carries a query with an outermost `ORDER BY` — if that shape is one of "
                "these cases its ordering is never checked; pass `ordered_by=` or confirm "
                "none of this module's cases asks for an order",
            )


#: Methods that return a *rendering* — text produced for a human to read. Their format is
#: not a contract, so an assertion that parses or greps one degrades silently when it moves.
_RENDERING_CALLS = frozenset(
    {"explain", "explain_analyze", "to_string", "summary", "render", "describe", "format_plan"}
)


def _renders(node: ast.AST) -> bool:
    """Whether evaluating `node` produces a human-facing rendering."""
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in _RENDERING_CALLS
        for call in ast.walk(node)
    )


def _render_membership(tree: ast.Module) -> Iterator[tuple[bool, str, int]]:
    """Every `"<literal>" [not] in <rendering>` assertion: `(negated, token, line)`."""
    for fn in ast.walk(tree):
        if not isinstance(fn, _Func):
            continue
        # A rendering is usually bound to a local first (`text = ds.explain()`), so the
        # assertion names a variable rather than the call.
        bound = {
            target.id
            for stmt in ast.walk(fn)
            if isinstance(stmt, ast.Assign) and _renders(stmt.value)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        for assertion in ast.walk(fn):
            if not isinstance(assertion, ast.Assert):
                continue
            for compare in ast.walk(assertion.test):
                if not (isinstance(compare, ast.Compare) and len(compare.ops) == 1):
                    continue
                if not isinstance(compare.ops[0], (ast.In, ast.NotIn)):
                    continue
                subject = compare.comparators[0]
                if not (
                    _renders(subject) or (isinstance(subject, ast.Name) and subject.id in bound)
                ):
                    continue
                literal = compare.left
                if not (isinstance(literal, ast.Constant) and isinstance(literal.value, str)):
                    continue
                yield isinstance(compare.ops[0], ast.NotIn), literal.value.lower(), assertion.lineno


def _uncontrolled_negatives(
    modules: list[tuple[Path, ast.Module]],
) -> Iterator[Finding]:
    """Negative assertions against a rendering that no positive assertion controls.

    Two passes over the whole suite, because the control for a negative in one file is
    routinely a positive in another — and matching them is the entire point of the rule.
    """
    positives: set[str] = set()
    negatives: list[tuple[Path, str, int]] = []
    for path, tree in modules:
        for negated, token, line in _render_membership(tree):
            if negated:
                negatives.append((path, token, line))
            else:
                positives.add(token)

    for path, token, line in negatives:
        # Substring either way: `assert "pushed[max 2 rows]" in scan_line` is a perfectly
        # good control for `assert "max" not in text`, and demanding an exact match would
        # report five of this tree's six controlled negatives as findings.
        if any(token in control or control in token for control in positives):
            continue
        yield Finding(
            "uncontrolled-negative-render",
            "high",
            _rel(path),
            line,
            f"asserts {token!r} is absent from a rendering, and no test anywhere asserts it is "
            f"ever *present* — nothing proves the token would appear if the behaviour under "
            f"test regressed, so a change to the renderer turns this into a tautology that "
            f"still passes; add the positive control",
        )


#: Clock reads. A value derived from one of these is a duration.
_TIMERS = frozenset(
    {"perf_counter", "monotonic", "time", "process_time", "perf_counter_ns", "monotonic_ns"}
)

#: Names that mark a duration as a *repeated-sample* estimate rather than one reading.
_REPEATED = ("fastest", "best_of", "min_of", "median", "_repeat", "samples")


def _durations(fn: _Func) -> set[str]:
    """Locals in `fn` holding a measured duration, following simple derivations.

    `t0 = perf_counter()` makes `t0` a duration-ish name and `elapsed = perf_counter() - t0`
    makes `elapsed` one too, so the assertion is usually two hops from the clock.
    """
    timed: set[str] = set()
    for _ in range(3):  # a fixed point in practice; nothing here derives deeper
        for stmt in ast.walk(fn):
            if not isinstance(stmt, ast.Assign):
                continue
            reads_clock = any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in _TIMERS
                for call in ast.walk(stmt.value)
            )
            derived = any(
                isinstance(node, ast.Name) and node.id in timed for node in ast.walk(stmt.value)
            )
            if reads_clock or derived:
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        timed.add(target.id)
    return timed


def _single_sample_timings(tree: ast.Module, rel: str) -> Iterator[Finding]:
    """Assertions weighing one measured duration against another."""
    for fn in ast.walk(tree):
        if not isinstance(fn, _Func) or not fn.name.startswith("test_"):
            continue
        timed = _durations(fn)
        if not timed:
            continue
        source = ast.unparse(fn)
        if any(marker in source for marker in _REPEATED):
            continue
        for assertion in ast.walk(fn):
            if not isinstance(assertion, ast.Assert):
                continue
            for compare in ast.walk(assertion.test):
                if not (isinstance(compare, ast.Compare) and len(compare.ops) == 1):
                    continue
                if not isinstance(compare.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    continue
                # Both sides must reference a measured duration. A duration against a
                # *constant* deadline ("this must finish inside 2s") is a different and
                # much safer shape: contention can only make it fail, never pass.
                sides = [compare.left, compare.comparators[0]]
                if not all(
                    any(isinstance(n, ast.Name) and n.id in timed for n in ast.walk(side))
                    for side in sides
                ):
                    continue
                yield Finding(
                    "single-sample-timing",
                    "high",
                    rel,
                    assertion.lineno,
                    f"`{fn.name}` compares two measured durations, each from a single "
                    f"sample — on a shared box that reports the neighbour's load rather "
                    f"than the property, so it passes or fails by coin-flip; take `min` of "
                    f"several runs of each arm, since contention only ever adds time",
                )
                break


#: Minimum size for a production constant to count as an enumeration worth deriving from,
#: and for a test's hand-written list to count as an attempt to mirror one.
_SHADOW_MIN_PROD = 4
_SHADOW_MIN_TEST = 3


def _string_set(node: ast.expr) -> list[str] | None:
    """The string literals in a set/list/tuple literal, or `None` if it is not purely those."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set"}
        and node.args
    ):
        node = node.args[0]
    if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        return None
    values = [
        e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
    ]
    return values if values and len(values) == len(node.elts) else None


def production_string_sets(root: Path) -> dict[str, list[set[str]]]:
    """Module-level `_UPPER` constants under `root` holding a set of string literals.

    Module-level and upper-cased on purpose: a local named `parts` collides with half the
    test suite and was the only pure-noise hit in the first measurement.

    **Set arithmetic is resolved**, and that is not a refinement -- without it the detector
    reports a false positive it cannot explain. `arith_extra` defines
    `_ROUNDING_PROMOTES_INT = _ROUNDING - {"round", "trunc"}`, and a test tracking that
    derived constant *exactly* was matched against `_ROUNDING` instead, because the constant
    it actually tracked was invisible to a literal-only reader and the superset was the
    nearest name in the file. Resolved left-to-right within a module, which is the order
    Python itself binds them in.
    """
    found: dict[str, list[set[str]]] = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        local: dict[str, set[str]] = {}
        for node in tree.body:
            target = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target.id
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                target = node.targets[0].id
            if not target or not target.lstrip("_").isupper() or node.value is None:
                continue
            values = _resolved_set(node.value, local)
            if values is None:
                continue
            local[target] = values
            # Recorded whatever its size. `_SHADOW_MIN_PROD` gates what a finding may be
            # matched *against*, not what counts as "a set this test already tracks" -- a
            # three-member derived constant is exactly the case the suppression below exists
            # for, and filtering it here made the suppression unable to see it.
            found.setdefault(target, []).append(values)
    return found


def _resolved_set(node: ast.expr, local: dict[str, set[str]]) -> set[str] | None:
    """A string-literal set, resolving `NAME - {...}` / `|` / `&` against `local`."""
    literal = _string_set(node)
    if literal is not None:
        return set(literal)
    if isinstance(node, ast.Name):
        return local.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.BitOr, ast.BitAnd)):
        left = _resolved_set(node.left, local)
        right = _resolved_set(node.right, local)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.BitOr):
            return left | right
        return left & right
    return None


def _shadowed_production_sets(
    tree: ast.Module, source: str, rel: str, constants: dict[str, list[set[str]]]
) -> Iterator[Finding]:
    """A hand-written test list that is a strict subset of a production constant it names.

    The engine enumerates its own vocabulary -- `AGG_FNS`, `JOIN_TYPES`, `FORMATS`,
    `_OBJECT_STORE_SCHEMES` -- and a test that retypes a few of those by hand covers what
    someone thought of on the day. What it leaves out is invisible: the test passes, the
    count looks deliberate, and the gap only surfaces when a user hits it.

    One finding is a shipped engine defect and is the evidence for the rule. `FORMATS` is
    hand-listed as `["pyarrow", "numpy", "pandas"]` against six production formats, unchanged
    since `18b85ead`; the uncovered `polars` silently widened every `string` to `large_string`
    on the way through `map_batches`, so `Dataset.schema` said `string` while `collect()`
    returned `large_string` and a Parquet file written from that plan was `large_string` on
    disk. The gap predates the defect's discovery, which is what makes it predictive.

    **A claim originally made for this rule does not survive checking, and is recorded here so
    it is not repeated.** `_ROUNDING` reports `round`/`trunc` uncovered and `_IDEMPOTENT_MATH`
    reports `sign`, and `87d82730` ("trunc and sign of an integer stay integers in every
    tier") fixed exactly those three -- which reads as the detector predicting a bug. It did
    not. Running this rule against the pre-fix tree (`87d82730^`) yields **zero** findings in
    that file: before the fix `trunc` was inside the tested list, and the finding exists only
    *because* the fix moved it out into a deliberate exclusion. The right counterfactual is
    "would it have fired before the bug was fixed", not "does it name the bug now", and the
    two differ.

    **Precision is about half, and the rule cannot do better.** A narrow set is sometimes
    correct and says so: `test_relational_window_rules` parametrizes three of
    `WINDOW_RANKING`'s members and its docstring explains that the rule under test keeps a
    deliberately narrower `_PREFIX_STABLE_RANKING`, because the other members divide by a
    partition total. Nothing mechanical separates that from an oversight -- both are a short
    list beside a longer constant. Treat a finding as a question, not a verdict.
    """
    for node in ast.walk(tree):
        values = _string_set(node)
        if not values:
            continue
        subset = set(values)
        # Distinct values, not literals. A `parametrize` **row** like `("rint", "rint",
        # "rint")` is three string literals and one value -- an argument triple, not an
        # enumeration -- and counting literals matched it against every rounding constant.
        if len(subset) < _SHADOW_MIN_TEST:
            continue
        # A list that *exactly* equals some production set is tracking that set completely,
        # even if it is also a strict subset of a wider one. Suppressing here is what makes a
        # real finding self-retire: the fix is to classify the members, and a classified list
        # equals the narrower constant it now tracks.
        if any(subset == full for populations in constants.values() for full in populations):
            continue
        for name, populations in constants.items():
            if name not in source:
                continue
            for full in populations:
                if len(full) < _SHADOW_MIN_PROD or not subset < full:
                    continue
                missing = sorted(full - subset)
                yield Finding(
                    "shadowed-production-set",
                    "medium",
                    rel,
                    getattr(node, "lineno", 0),
                    f"hand-lists {len(subset)} of `{name}`'s {len(full)} members; "
                    f"{missing[:4]}{' ...' if len(missing) > 4 else ''} are never exercised — "
                    f"derive the parametrization from `{name}` and classify what it cannot "
                    f"cover, so a new member fails here instead of shipping untested",
                )
                return


#: Reads that report *how a query ran* rather than what it produced. A helper returning one
#: of these is observing the machine, not the data.
_RUN_OBSERVATIONS = frozenset(
    {
        "cpu_utilization",
        "threads",
        "peak_bytes",
        "spilled",
        "elapsed_ns",
        "cpu_ns",
        "buckets",
        "partitions",
        "rows_in",
    }
)

#: Calls that force a knob before the engine runs. A helper that sets one is *causing* the
#: difference it goes on to assert; a helper that sets none is hoping for it.
_KNOB_SETTERS = ("set_config", "set_option", "setenv", "configure", "replace(")

#: Ways a helper actually runs the engine.
_RUNS_ENGINE = ("collect(", "explain(", "iter_batches(", "execute_local", "execute_plan")


def _uncontrolled_runtime_comparisons(tree: ast.Module, rel: str) -> Iterator[Finding]:
    """Two runs compared on an observation neither run controlled.

    The shape: a test-local helper runs the engine and returns a figure describing *how* it
    ran -- CPU utilization, thread count, bucket count -- and the test calls it twice with
    different inputs and asserts one is strictly greater. That reads like a controlled
    experiment and is not one. Nothing forced the difference; the test varied an input and
    hoped the machine responded, so on a loaded box it reports the neighbour's load.

    The discriminator is mechanical rather than a judgement about intent: **did the helper
    set a knob?** `threads_at` in `test_diff_morsel_size_invariance` calls
    `set_config(morsel_rows=...)` and `run_with` in `test_spilling` sets a spill bound --
    both force the difference and then assert it reached the engine, which is a control
    proving a knob is live. A helper that sets nothing is making an observational claim in a
    controlled experiment's clothes.

    Advisory, not ratcheted: the population is three helpers tree-wide and the rule is new.
    """
    for fn in ast.walk(tree):
        if not isinstance(fn, _Func) or not fn.name.startswith("test_"):
            continue
        for helper in ast.walk(fn):
            if not isinstance(helper, _Func) or helper is fn:
                continue
            body = ast.unparse(helper)
            if not any(marker in body for marker in _RUNS_ENGINE):
                continue
            if not any(obs in body for obs in _RUN_OBSERVATIONS):
                continue
            if any(setter in body for setter in _KNOB_SETTERS):
                continue  # the helper forces the difference: a control, not a hope
            calls = [
                node
                for node in ast.walk(fn)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == helper.name
            ]
            if len(calls) < 2:
                continue
            for assertion in ast.walk(fn):
                if not isinstance(assertion, ast.Assert):
                    continue
                for compare in ast.walk(assertion.test):
                    if not (isinstance(compare, ast.Compare) and len(compare.ops) == 1):
                        continue
                    if not isinstance(compare.ops[0], (ast.Lt, ast.Gt)):
                        continue
                    yield Finding(
                        "uncontrolled-runtime-comparison",
                        "medium",
                        rel,
                        assertion.lineno,
                        f"`{fn.name}` compares two runs on `{helper.name}`, which observes "
                        f"how the engine ran rather than what it produced, and sets no knob "
                        f"to force the difference it asserts — so a loaded box reports the "
                        f"neighbour's load; set the knob and assert it reached the engine, "
                        f"or compare a property the run controls",
                    )
                    return


#: Clock reads that mark a module as *timing* something.
_BENCH_TIMERS = frozenset({"perf_counter", "monotonic", "process_time", "perf_counter_ns"})


def check_benchmark_guards(path: Path, tree: ast.Module, source: str) -> Iterator[Finding]:
    """A benchmark that publishes a timing without checking what it is timing.

    `envinfo.require_release_build` refuses to measure a dev-profile engine, because doing so
    "would compare an unoptimized Batcher against release competitors" — its own docstring
    puts the gap at 8-60x. The guard has existed as long as `envinfo` has. Measured on this
    tree, **60 of the 64 standalone timing entry points never call it**, so almost every
    number `benchmarks/` can produce outside `run.py` could come from a debug build with
    nothing said.

    That is the same failure as the missing `require_quiet_box` on `run.py`: a guard that
    exists, is correct, and is not reached from the paths that need it. A guard nobody calls
    is indistinguishable from a guard nobody wrote.
    """
    has_entry = '__name__ == "__main__"' in source or "__name__ == '__main__'" in source
    if not has_entry:
        return
    times = any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in _BENCH_TIMERS
        for call in ast.walk(tree)
    )
    if not times or "require_release_build" in source:
        return
    yield Finding(
        "benchmark-unguarded-build",
        "medium",
        _rel(path),
        1,
        "a standalone timing entry point that never calls `require_release_build` — a "
        "dev-profile engine is 8-60x slower, so this can publish a number from a debug "
        "build with no warning at all",
    )


#: Evidence, in a differential module's *code*, that it compares against something.
#: Deliberately wide: the second oracle is legitimately Python's own stdlib for a codec, or
#: the Tier-0 interpreter for a path-equivalence test, and neither mentions DuckDB.
_ORACLE_CALLS = (
    "duck",
    "polars",
    "pl.",
    "zoneinfo",
    "gzip",
    "zlib",
    "pandas",
    "pd.",
    "numpy",
    "scipy",
    "sklearn",
    "shapely",
    "pyproj",
    "assert_tables_equal",
    "spill=True",
    "distributed=True",
    "iter_batches",
    "run_with_rules",
)

#: ...or, failing that, a module docstring that says what it is checked against.
_ORACLE_DECLARED = (
    "oracle",
    "reference",
    "duckdb",
    "polars",
    "hand-computed",
    "invariant",
    "equivalence",
    "same result",
)


def check_differential_oracle(path: Path, tree: ast.Module, source: str) -> Iterator[Finding]:
    """A file under `tests/differential/` that neither uses an oracle nor names one.

    The directory name is a claim. `.claude/rules/testing.md` calls it "the correctness
    spine" and requires that any relational or expression behaviour match DuckDB on the same
    input — so a file here asserting only hand-written expected values is a unit test wearing
    a differential test's name, and the cross-check the contract mandates is simply missing.

    The rule accepts a *declared* alternative, because several are legitimate and this tree
    documents them well: `test_diff_timezone` uses Python's `zoneinfo` (DuckDB needs ICU),
    `test_diff_str_compress` uses the standard library's own codecs (DuckDB has none, and a
    round-trip against ourselves would pass on a private format), and the path-equivalence
    files use the Tier-0 interpreter, which is the project's second oracle by contract. What
    it will not accept is silence.

    Checking the seven it found was worth more than the rule: `test_diff_json` was asserting
    hand-written values where `json_extract_string` existed, and adding the oracle surfaced a
    genuine divergence — Batcher yields null for unparseable JSON, DuckDB raises and fails
    the query. That is exactly the "surface it explicitly rather than hide it" case the
    contract describes, and nothing had surfaced it.
    """
    if any(marker in source for marker in _ORACLE_CALLS):
        return
    doc = (ast.get_docstring(tree) or "").lower()
    if any(marker in doc for marker in _ORACLE_DECLARED):
        return
    yield Finding(
        "differential-without-oracle",
        "medium",
        _rel(path),
        1,
        "lives in `tests/differential/` — the correctness spine — but neither compares "
        "against an oracle nor says in its docstring what it is checked against; either add "
        "the cross-check or state which oracle stands in for DuckDB and why",
    )


def check_rust_ignores(path: Path, source: str) -> Iterator[Finding]:
    """A Rust `#[ignore]` that does not say why it is ignored.

    Python's side of this is already watched: `tools/lint_skips.py` ratchets
    ``unconditionally_skipped`` at zero, so a `@pytest.mark.skip` cannot appear unnoticed.
    Rust had no equivalent, and `cargo test` prints its ignored count as a bare number in a
    line everyone reads as success.

    Every one of this tree's ignored tests is a *timing study* rather than an assertion, which
    is a legitimate reason to exclude one — a benchmark that asserts nothing has no business
    failing a correctness run. But that is only knowable by reading each body. A bare
    `#[ignore]` and a `#[ignore]` on a test that was quarantined because it started failing
    look identical from the outside, and the second is how a known-broken path stays broken.

    `#[ignore = "why"]` is the whole fix, and `cargo test` prints the reason.
    """
    for number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#[ignore]"):
            yield Finding(
                "rust-ignore-without-reason",
                "low",
                _rel(path),
                number,
                "`#[ignore]` with no reason — indistinguishable from a test quarantined "
                'because it fails; write `#[ignore = "timing study, not an assertion"]` or '
                "whatever the actual reason is, and `cargo test` will print it",
            )


def _rust_sources() -> Iterator[tuple[Path, str]]:
    """Every `.rs` file in the workspace."""
    for path in sorted((ROOT / "crates").rglob("*.rs")):
        try:
            yield path, path.read_text()
        except (OSError, UnicodeDecodeError):
            continue


def _example_scripts() -> Iterator[tuple[Path, ast.Module]]:
    """Every executed script under `examples/` — mirroring `tests/docs/test_examples.py`.

    Underscore-prefixed parts are shared support code that the suite does not run, and a
    ``# examples: skip`` script is collected but never executed, so neither owes an
    assertion.
    """
    root = ROOT / "examples"
    for path in sorted(root.rglob("*.py")):
        if any(part.startswith("_") for part in path.relative_to(root).parts):
            continue
        text = path.read_text()
        body = [
            line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#!")
        ]
        if body and body[0].startswith("# examples: skip"):
            continue
        try:
            yield path, ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue


def _benchmark_modules() -> Iterator[tuple[Path, ast.Module]]:
    """Every module under `benchmarks/`."""
    for path in sorted((ROOT / "benchmarks").rglob("*.py")):
        try:
            yield path, ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue


# --- a patch aimed at a name the target module does not bind -------------------------------
#
# `monkeypatch.setattr(mod, "name", ...)` binds a *name*, not a function. When `name` moves to
# a submodule, is imported inside a function body rather than at module level, or was never
# there, the patch applies to something nothing reads — and the test goes on running the real
# code, taking real time, and passing. `.claude/rules/concurrent-agents.md` records the class
# -- the patch stops applying and the test keeps passing while testing nothing -- but nothing
# checked it.
#
# The two instances that motivated it are worth stating because they fail in **opposite**
# directions, and a reader who has seen only one will assume the failure mode is one-sided:
#
#   - a spy on the *package* `dist.spill_breakers` missed `execute_spilling_sort`'s call to
#     the module-global in `spill_breakers/sort.py`, so shapes that spilled perfectly
#     reported "never spilled";
#   - a spy on the *submodule* `dist.global_window.disk` missed `_dispatch`'s function-local
#     import from `batcher.dist.global_window`, which resolves the package attribute through
#     a PEP 562 hook — so a shape that ran correctly reported "reached nothing".
#
# `raising=False` is what makes it silent: without it pytest raises `AttributeError` and the
# test fails loudly, which is the outcome we want and do not need a rule for.

#: Names that mark a module as re-exporting lazily, so its attribute set cannot be settled
#: statically. Reporting those would put noise exactly where the second bug above lived, which
#: is worse than silence — see the "rules deliberately not written" note.
_LAZY_MARKERS = ("__getattr__",)


def _module_file(dotted: str) -> Path | None:
    """The source file for a dotted `batcher.*` module, or `None`."""
    base = ROOT / "python" / Path(*dotted.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    package = base / "__init__.py"
    return package if package.is_file() else None


def _bound_names(path: Path) -> set[str] | None:
    """`path`'s top-level bindings, or `None` when it re-exports lazily."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, OSError, UnicodeDecodeError):
        return None
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in _LAZY_MARKERS:
                return None
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.If):  # `if TYPE_CHECKING:` and friends
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names.update(a.asname or a.name.split(".")[0] for a in sub.names)
    return names


def _module_aliases(nodes: list[ast.AST]) -> dict[str, str]:
    """Local name -> the dotted `batcher.*` module it refers to, over `nodes`."""
    out: dict[str, str] = {}
    for root in nodes:
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("batcher"):
                        out[a.asname or a.name.split(".")[0]] = a.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                if not node.module.startswith("batcher"):
                    continue
                for a in node.names:
                    dotted = f"{node.module}.{a.name}"
                    if _module_file(dotted) is not None:
                        out[a.asname or a.name] = dotted
    return out


def _own_statements(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node in `scope` **except** those inside a nested function or class.

    Scope splitting has to be explicit, because `ast.walk` has no notion of one: walking a
    module reaches every call inside every function, and resolving those against the
    *module's* alias map is what made the first version of this report a correct patch as
    broken. Each function gets its own pass with its own imports overlaid.
    """
    stack = list(getattr(scope, "body", []))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # its own scope; visited separately
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _inert_patches(path: Path, tree: ast.Module) -> Iterator[Finding]:
    """Every `setattr(<module>, "name", ...)` whose target module does not bind `name`.

    Aliases are resolved **per scope**, not per file, via `_own_statements`. The first version
    walked the whole module, and two helpers in one file that each did
    `from batcher.io.formats.streaming import <different module> as mod` made the later
    binding answer for the earlier one's call — reporting a correct patch as broken. An
    instrument that resolves a name in the wrong scope gives a confident wrong answer rather
    than a miss, which is the same defect this rule exists to find, turned on itself.
    """
    module_level = _module_aliases(list(_own_statements(tree)))
    scopes: list[tuple[dict[str, str], ast.AST]] = [(module_level, tree)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            own = list(_own_statements(node))
            scopes.append(({**module_level, **_module_aliases(own)}, node))

    seen: set[tuple[int, str]] = set()
    for alias, scope in scopes:
        for node in _own_statements(scope):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            named = isinstance(fn, ast.Attribute) and fn.attr == "setattr"
            builtin = isinstance(fn, ast.Name) and fn.id == "setattr"
            if not (named or builtin) or len(node.args) < 2:
                continue
            attr, target = node.args[1], node.args[0]
            if not isinstance(attr, ast.Constant) or not isinstance(attr.value, str):
                continue
            if not isinstance(target, ast.Name) or target.id not in alias:
                continue
            dotted = alias[target.id]
            source = _module_file(dotted)
            if source is None:
                continue
            bound = _bound_names(source)
            if bound is None or attr.value in bound:
                continue
            key = (node.lineno, f"{dotted}.{attr.value}")
            if key in seen:
                continue
            seen.add(key)
            yield Finding(
                "patch-target-not-bound",
                "high",
                _rel(path),
                node.lineno,
                f"patches `{dotted}.{attr.value}`, which that module does not bind at "
                "module level — the patch applies to nothing and the test goes on running "
                "the real code, passing. Patch the module that actually defines the name "
                "(and, when a caller imports it inside a function, the package it "
                "resolves through as well)",
            )


def detect_methodology(ctx: Context) -> Iterator[Finding]:  # noqa: ARG001 — uniform signature
    """Gates that report a success they did not measure."""
    from tools.audit.testing import test_modules

    modules = list(test_modules())
    for path, tree in modules:
        yield from check_test_module(path, tree)
        if path.parent.name == "differential":
            yield from check_differential_oracle(path, tree, path.read_text())
    yield from _uncontrolled_negatives(modules)
    # Read the engine's own enumerations once, not per test module: `shadowed-production-set`
    # compares every test's hand-written list against all of them.
    constants = production_string_sets(Path("python") / "batcher")
    for path, tree in modules:
        yield from _single_sample_timings(tree, _rel(path))
        yield from _uncontrolled_runtime_comparisons(tree, _rel(path))
        yield from _shadowed_production_sets(tree, path.read_text(), _rel(path), constants)
        yield from _inert_patches(path, tree)
    for path, tree in _example_scripts():
        yield from check_example(path, tree)
    for path, source in _rust_sources():
        yield from check_rust_ignores(path, source)
    for path, tree in _benchmark_modules():
        yield from check_benchmark(path, tree)
        yield from check_benchmark_guards(path, tree, path.read_text())
