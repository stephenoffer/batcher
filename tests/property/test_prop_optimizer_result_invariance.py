"""Property: the full optimizer never changes a query's result.

This is *the* behavioral guard on Kyber's rule set. The optimizer carries 150+
result-preserving rewrites (boolean algebra, sargable predicates, arithmetic
reassociation, predicate inference, set-op rewrites, empty-relation folding, top-N
fusion, ...). Every one of them is supposed to change the *plan* and never the
*answer*. Example-based differential tests pin individual shapes; this searches the
space: Hypothesis generates a random table and a random-but-valid pipeline
(filter / with_columns / group-by-agg / distinct / sort / limit / union chains) and
asserts

    result(FULL optimizer)  ==  result(NO optimizer)  ==  ds.collect()

via an order-independent multiset compare. Any rule that alters a result — the exact
subtle bug an example misses — falls out as a Hypothesis counterexample.

It also checks **result-level idempotence/confluence**: optimizing an already-optimized
plan is plan-stable (its IR is a fixpoint) and re-executing it yields the same rows.

A counterexample here is a real correctness bug in a Kyber rule — minimize (shrink the
table + the pipeline), identify the offending rule, and report it; do not weaken this
test to make it pass.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col, count, lit

pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(autouse=True)
def _stable_morsel_tuning():
    """Pin the adaptive morsel-size tuning off for these tests.

    Morsel sizing is a Carbonite *tuning* concern (result-invariant by contract) and
    orthogonal to the optimizer correctness under test here; keeping it fixed makes the
    optimizer-vs-no-optimizer comparison depend only on the rule set, and avoids routing
    every ``collect()`` through the learned-memory tuning path.
    """
    from batcher.config import active_config, set_config

    prev = active_config()
    set_config(
        prev.replace(execution=dataclasses.replace(prev.execution, adaptive_morsel_sizing=False))
    )
    yield
    set_config(prev)


from _optmeta_common import (  # noqa: E402  (test-dir helper, see module docstring)
    FULL_RULES,
    NO_RULES,
    Optimizer,
    ordered_rows,
    rowset,
    run_with_rules,
)

pytestmark = [pytest.mark.property, pytest.mark.integration]

# Small, varied, typed. A dense low-cardinality `g` gives real groups; `v`/`w` are
# nullable ints (arithmetic + null 3VL); `f` float; `b` bool; `s` a small string domain.
_SCHEMA = pa.schema(
    [
        ("g", pa.int64()),
        ("v", pa.int64()),
        ("w", pa.int64()),
        ("f", pa.float64()),
        ("b", pa.bool_()),
        ("s", pa.string()),
    ]
)
_INT = st.integers(min_value=-8, max_value=8)
_NULL_INT = st.one_of(st.none(), _INT)
_NULL_FLOAT = st.one_of(
    st.none(), st.floats(min_value=-8, max_value=8, allow_nan=False, allow_infinity=False)
)
_NULL_BOOL = st.one_of(st.none(), st.booleans())
_NULL_STR = st.one_of(st.none(), st.sampled_from(["a", "b", "c"]))
_BASE_COLS = ("g", "v", "w", "f", "b", "s")


@st.composite
def _table(draw: st.DrawFn) -> pa.Table:
    """A random table on `_SCHEMA` (0..30 rows; empty/all-null/single-row all reachable)."""
    n = draw(st.integers(min_value=0, max_value=30))
    return pa.table(
        {
            "g": draw(st.lists(st.integers(min_value=0, max_value=3), min_size=n, max_size=n)),
            "v": draw(st.lists(_NULL_INT, min_size=n, max_size=n)),
            "w": draw(st.lists(_NULL_INT, min_size=n, max_size=n)),
            "f": draw(st.lists(_NULL_FLOAT, min_size=n, max_size=n)),
            "b": draw(st.lists(_NULL_BOOL, min_size=n, max_size=n)),
            "s": draw(st.lists(_NULL_STR, min_size=n, max_size=n)),
        },
        schema=_SCHEMA,
    )


@st.composite
def _nested_scalar(draw: st.DrawFn):
    """A *nested* scalar function chain over the base columns.

    The generator's derived-column stage builds integer arithmetic only, which leaves the
    expression families that rewrite *function composition* -- string, math, cast, and the
    list algebra below -- outside the randomized search entirely. Two rules shipped with
    wrong-answer bugs (`arg_sort`/`flatten` idempotence) precisely because no generated
    pipeline ever nested one call inside another of the same family.

    Chains are built two-deep so a collapse rule has a pair to act on.
    """
    kind = draw(st.sampled_from(["str", "math", "cast", "trim"]))
    if kind == "str":
        outer = draw(st.sampled_from(["upper", "lower", "reverse", "initcap"]))
        inner = draw(st.sampled_from(["upper", "lower", "reverse", "initcap"]))
        return getattr(getattr(col("s").str, inner)().str, outer)()
    if kind == "trim":
        outer = draw(st.sampled_from(["strip", "lstrip", "rstrip"]))
        inner = draw(st.sampled_from(["strip", "lstrip", "rstrip"]))
        return getattr(getattr(col("s").str, inner)().str, outer)()
    if kind == "cast":
        # `cast(int -> float) OP literal` is the unwrap-cast family's exact shape.
        op = draw(st.sampled_from(["<", "<=", ">", ">=", "==", "!="]))
        rhs = draw(st.sampled_from([0.0, 1.0, -1.5, 2.5, 3.0]))
        lhs = col("v").cast("float64")
        return {
            "<": lhs < rhs,
            "<=": lhs <= rhs,
            ">": lhs > rhs,
            ">=": lhs >= rhs,
            "==": lhs == rhs,
            "!=": lhs != rhs,
        }[op]
    outer = draw(st.sampled_from(["abs", "floor", "ceil", "trunc", "sign"]))
    inner = draw(st.sampled_from(["abs", "floor", "ceil", "trunc", "sign"]))
    return getattr(getattr(col("f"), inner)(), outer)()


@st.composite
def _nested_list(draw: st.DrawFn):
    """A two-deep list-function chain over a list built from the integer columns.

    This is the family the two shipped bugs lived in: `arg_sort` and `flatten` are not
    idempotent, and collapsing them returned wrong answers. Nothing in the generator
    could build `arr.arg_sort().arg_sort()` before, so nothing caught it.
    """
    base = bt.array(col("v"), col("w"))
    inner = draw(st.sampled_from(["sort", "reverse", "unique", "arg_sort"]))
    chained = getattr(base.list, inner)()
    outer = draw(st.sampled_from(["sort", "reverse", "unique", "arg_sort", "len", "min", "max"]))
    return getattr(chained.list, outer)()


@st.composite
def _predicate(draw: st.DrawFn, cols: tuple[str, ...], depth: int = 0):
    """A bounded random boolean predicate over `cols` — the shape the filter rules chew on.

    Mixes comparisons (`col OP lit`, `col OP col`, `lit OP col`), `AND`/`OR`/`NOT`,
    `is_null`/`is_not_null`, and `is_in` — deliberately including redundant/absorbing
    shapes (`p AND p`, `p OR NOT p`) so boolean-algebra / predicate-inference rules fire.
    """
    num_cols = [c for c in cols if c in ("g", "v", "w", "f")]

    def comparison():
        c = draw(st.sampled_from(num_cols))
        op = draw(st.sampled_from(["<", "<=", ">", ">=", "==", "!="]))
        e = col(c)
        rhs_col = draw(st.sampled_from([None, *num_cols]))
        rhs = col(rhs_col) if rhs_col is not None else lit(draw(_INT))
        return {
            "<": e < rhs,
            "<=": e <= rhs,
            ">": e > rhs,
            ">=": e >= rhs,
            "==": e == rhs,
            "!=": e != rhs,
        }[op]

    if depth >= 2:
        kind = draw(st.sampled_from(["cmp", "null", "in"]))
    else:
        kind = draw(st.sampled_from(["cmp", "null", "in", "and", "or", "not", "dup", "absorb"]))
    if kind == "cmp":
        return comparison()
    if kind == "null":
        c = draw(st.sampled_from(cols))
        return col(c).is_null() if draw(st.booleans()) else col(c).is_not_null()
    if kind == "in":
        c = draw(st.sampled_from(num_cols))
        vals = draw(st.lists(_INT, min_size=1, max_size=4))
        return col(c).is_in(vals)
    if kind == "and":
        return draw(_predicate(cols, depth + 1)) & draw(_predicate(cols, depth + 1))
    if kind == "or":
        return draw(_predicate(cols, depth + 1)) | draw(_predicate(cols, depth + 1))
    if kind == "not":
        return ~draw(_predicate(cols, depth + 1))
    if kind == "dup":  # idempotence / remove-duplicate-conjuncts bait
        p = draw(_predicate(cols, depth + 1))
        return p & p
    # absorption / complementation bait: p OR (NOT p)  (a tautology under total order)
    p = comparison()
    return p | (~p)


@st.composite
def _query(draw: st.DrawFn) -> tuple[bt.Dataset, bool]:
    """A random valid pipeline over a random table; returns (dataset, order_matters)."""
    table = draw(_table())
    ds = bt.from_arrow(table.to_batches() or table)
    cols = list(_BASE_COLS)

    # 0..2 optional filters (with an optional derived-column stage between them).
    if draw(st.booleans()):
        ds = ds.filter(draw(_predicate(tuple(cols))))
    if draw(st.booleans()):  # derived integer columns → arith_algebra / sargable bait
        ds = ds.with_columns(
            x=(col("v") + col("w")) * draw(st.integers(min_value=-3, max_value=3)),
            y=col("v") - col("w"),
        )
        cols += ["x", "y"]
    # Nested function chains -- the shapes the composition-collapsing families rewrite,
    # and the ones nothing in this generator could previously build. Added as a *terminal*
    # stage that returns immediately: the downstream terminals reference columns by name
    # (`sort` orders by all of them, `union` re-projects), and a list-valued column is not
    # a legal sort or group key. Returning here keeps the added column in the output --
    # which is what makes the optimizer-vs-no-optimizer comparison actually see it -- while
    # leaving the existing terminal coverage untouched on the other draws.
    if draw(st.booleans()):
        ds = ds.with_columns(nested=draw(_nested_scalar()))
        return ds, False
    if draw(st.booleans()):
        ds = ds.with_columns(nested_list=draw(_nested_list()))
        return ds, False
    if draw(st.booleans()):
        ds = ds.filter(draw(_predicate(tuple(cols))))

    terminal = draw(
        st.sampled_from(["agg", "distinct", "sort", "limit", "union", "select", "none"])
    )
    order_matters = False
    if terminal == "agg":
        keys = draw(st.lists(st.sampled_from(["g", "b", "s"]), min_size=1, max_size=2, unique=True))
        ds = ds.group_by(*keys).agg(
            s_sum=col("v").sum(),
            w_min=col("w").min(),
            w_max=col("w").max(),
            n=count(),
            v_uniq=col("v").n_unique(),
            b_and=col("b").bool_and(),
            b_or=col("b").bool_or(),
        )
    elif terminal == "distinct":
        subset = draw(
            st.lists(st.sampled_from([c for c in cols if c != "f"]), max_size=3, unique=True)
        )
        ds = ds.distinct(subset or None)
    elif terminal == "sort":
        # Sort by ALL current columns (random directions on a prefix) → a total order, so
        # LIMIT slices deterministically and the ordered compare is well-defined (identical
        # rows are interchangeable, hence still multiset-safe).
        perm = draw(st.permutations(cols))
        desc = [draw(st.booleans()) for _ in perm]
        ds = ds.sort(*perm, descending=desc)
        if draw(st.booleans()):
            ds = ds.limit(draw(st.integers(min_value=0, max_value=10)))
        order_matters = True
    elif terminal == "limit":
        ds = ds.limit(
            draw(st.integers(min_value=0, max_value=10)),
            offset=draw(st.integers(min_value=0, max_value=5)),
        )
    elif terminal == "union":
        other = bt.from_arrow(draw(_table()).to_batches() or draw(_table()))
        proj = draw(st.lists(st.sampled_from(cols), min_size=1, max_size=3, unique=True))
        ds = ds.select(*proj).union(
            _project_to(bt.from_arrow(table.to_batches() or table), proj, other),
            distinct=draw(st.booleans()),
        )
    elif terminal == "select":
        proj = draw(st.lists(st.sampled_from(cols), min_size=1, max_size=4, unique=True))
        ds = ds.select(*proj)
    return ds, order_matters


def _project_to(base: bt.Dataset, proj: list[str], other: bt.Dataset) -> bt.Dataset:
    """A union RHS with exactly columns `proj` — the same base table re-projected.

    Uses the *base* table (not `other`, whose schema is fixed to `_SCHEMA` and lacks the
    derived x/y) so `proj` — which may reference derived columns — always resolves.
    """
    return base.with_columns(x=(col("v") + col("w")) * lit(1), y=col("v") - col("w")).select(*proj)


_PROP = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
)


@_PROP
@given(_query())
def test_full_optimizer_matches_no_optimizer(case: tuple[bt.Dataset, bool]) -> None:
    """FULL rule set == empty rule set == public collect(), for a random pipeline."""
    ds, order_matters = case
    logical, sources = ds._plan, ds._sources
    full = run_with_rules(logical, sources, FULL_RULES)
    none = run_with_rules(logical, sources, NO_RULES)
    collected = ds.collect()

    assert set(full.column_names) == set(none.column_names) == set(collected.column_names), (
        f"schema drift: full={full.column_names} none={none.column_names} "
        f"collect={collected.column_names}"
    )
    f_rows, n_rows, c_rows = rowset(full), rowset(none), rowset(collected)
    assert f_rows == n_rows, (
        "OPTIMIZER CHANGED THE RESULT (a rule is not result-preserving):\n"
        f"  full : {f_rows}\n  none : {n_rows}"
    )
    assert f_rows == c_rows, (
        f"public collect() disagrees with the optimized plan:\n  full: {f_rows}\n  coll: {c_rows}"
    )
    if order_matters:
        # Total-ordered pipeline: the row *order* must also be identical.
        assert ordered_rows(full) == ordered_rows(none), (
            "OPTIMIZER CHANGED ROW ORDER (a sort/top-N rule reordered results):\n"
            f"  full: {ordered_rows(full)}\n  none: {ordered_rows(none)}"
        )


@_PROP
@given(_query())
def test_optimizer_is_result_idempotent(case: tuple[bt.Dataset, bool]) -> None:
    """Re-optimizing converges to a fixpoint and never changes the result.

    A single ``logical_rewrite`` is not always a plan fixpoint — e.g. pruning a Scan can
    leave a now-identity Project that a *subsequent* sweep removes — but iterating must
    (a) reach a stable plan within a small bound (no oscillation / non-termination) and
    (b) leave the executed result identical at every step. Result invariance is the
    correctness contract; bounded convergence is the confluence contract.
    """
    ds, _ = case
    logical, sources = ds._plan, ds._sources
    base = rowset(run_with_rules(logical, sources, FULL_RULES))

    current = logical
    prev_ir = None
    converged_at = None
    for i in range(1, 6):  # generous bound; it stabilizes within 2-3 in practice
        current = Optimizer(sources=list(sources), rules=FULL_RULES).logical_rewrite(current)
        # Re-executing the (further) optimized plan must reproduce the exact rows.
        again = rowset(run_with_rules(current, sources, FULL_RULES))
        assert again == base, (
            f"re-optimizing (pass {i}) changed the result:\n  base : {base}\n  pass{i}: {again}"
        )
        ir = current.to_ir()
        if ir == prev_ir:
            converged_at = i
            break
        prev_ir = ir
    assert converged_at is not None, "optimizer did not reach a plan fixpoint within 5 passes"


@given(_query())
def test_rules_converge_within_production_cap_and_deterministically(case: tuple[bt.Dataset, bool]):
    """The full 154-rule set converges within the *production* fixpoint budget, and the
    optimizer is deterministic.

    "The rules don't interfere or hurt each other in combination" has a precise form: no
    pair may oscillate, and the combined set must reach its plan fixpoint within the
    ``fixpoint_iterations`` budget the engine actually runs in production — otherwise a
    plan is silently *under*-optimized (capped mid-convergence) once enough rules are
    added. This asserts the optimized IR at the production cap equals the IR at a far
    larger cap (fixpoint genuinely reached, not truncated), and that re-running is
    byte-identical (a confluent rule system has a unique normal form).
    """
    from batcher.config import Config, OptimizerConfig, config_context

    ds, _ = case
    logical, sources = ds._plan, ds._sources

    def _optimize_at(cap: int):
        with config_context(Config().replace(optimizer=OptimizerConfig(fixpoint_iterations=cap))):
            return Optimizer(sources=list(sources), rules=FULL_RULES).optimize(logical).ir

    at_prod = _optimize_at(8)  # the shipped default
    at_generous = _optimize_at(40)  # far more headroom; phases stop early at convergence
    assert at_prod == at_generous, (
        "the 154-rule set did not converge within the production fixpoint cap (a rule "
        "combination needs more passes than the engine runs → silent under-optimization)"
    )
    assert _optimize_at(8) == at_prod, "optimizer output is non-deterministic across runs"


# --- exhaustive: the list-function composition cross-product -------------------


@pytest.mark.parametrize("inner", ["sort", "reverse", "unique", "arg_sort", "flatten", "normalize"])
@pytest.mark.parametrize(
    "outer",
    ["sort", "reverse", "unique", "arg_sort", "normalize", "len", "min", "max", "n_unique"],
)
def test_every_list_function_pair_is_result_preserving(inner: str, outer: str) -> None:
    """Enumerate every two-deep list-function chain and pin optimized == unoptimized.

    The randomized generator above *can* build these, but it dilutes any particular pair
    to roughly one draw in thirty, so it does not reliably fire on the one that matters.
    That is not hypothetical: `arg_sort` and `flatten` shipped in the idempotent-collapse
    set, both are non-idempotent (`arg_sort` twice is the *inverse* permutation, `flatten`
    removes one nesting level per call), and both returned wrong answers. Re-introducing
    either bug leaves the random test green and turns this one red.

    Exhaustive beats random here because the family is small and closed -- there are only
    a few dozen pairs, so enumerating them costs less than hoping to hit one.
    """
    # THREE elements, not two. With two-element lists `arg_sort` is a fixed point --
    # every 2-permutation squares to itself -- so a two-column array cannot distinguish
    # `arg_sort(arg_sort(x))` from `arg_sort(x)` and the whole guard passes vacuously.
    # `arg_sort([3,1,2])` is `[1,2,0]` and applying it again gives `[2,0,1]`, which is the
    # smallest input that actually separates them. Nulls and a duplicate value are kept so
    # the null-handling and tie-breaking paths run too.
    table = pa.table(
        {
            "v": [3, 1, None, 2, 2, -5],
            "w": [2, 5, 4, 2, None, 7],
            "u": [1, 9, 6, 2, 3, 0],
        }
    )
    base = bt.from_arrow(table)
    chained = getattr(bt.array(col("v"), col("w"), col("u")).list, inner)()
    try:
        expr = getattr(chained.list, outer)()
    except Exception:
        pytest.skip(f"{outer} is not defined over the output of {inner}")
    ds = base.with_columns(r=expr)

    logical, sources = ds._plan, ds._sources
    try:
        none = run_with_rules(logical, sources, NO_RULES)
    except Exception:
        pytest.skip(f"engine rejects {outer}({inner}(...))")
    full = run_with_rules(logical, sources, FULL_RULES)
    assert full.column("r").to_pylist() == none.column("r").to_pylist(), (
        f"optimizer changed {outer}({inner}(x)):\n"
        f"  optimized:   {full.column('r').to_pylist()}\n"
        f"  unoptimized: {none.column('r').to_pylist()}"
    )


# --- exhaustive: the string-function composition cross-product -----------------

#: Unary string functions the optimizer has collapse/involution rules over. Composing
#: any two must leave the result unchanged; `extra/strings` collapses the idempotent
#: ones and `exprs/text` unwraps the involutions.
_STR_UNARY = ["upper", "lower", "reverse", "initcap", "strip", "lstrip", "rstrip"]


@pytest.mark.parametrize("inner", _STR_UNARY)
@pytest.mark.parametrize("outer", _STR_UNARY)
def test_every_string_function_pair_is_result_preserving(inner: str, outer: str) -> None:
    """Enumerate every two-deep unary string chain and pin optimized == unoptimized.

    The same guard shape as the list cross-product above, for the same reason: three
    wrong-answer bugs lived in composition rules, and reasoning about which functions are
    idempotent or involutive is exactly where I got them wrong. `upper(lower(x))` is the
    one to watch here -- it is *not* `upper(x)` in general Unicode, which is why no rule
    collapses it and why this test would go red if one were added.

    The input carries the cases the collapse rules turn on: mixed case, leading and
    trailing whitespace, a non-ASCII string (where Python and Rust case mapping may
    differ), the empty string, and a null.
    """
    table = pa.table({"s": ["AbC dEf", "  padded  ", "straße", "", None]})
    base = bt.from_arrow(table)
    chained = getattr(col("s").str, inner)()
    ds = base.with_columns(r=getattr(chained.str, outer)())

    logical, sources = ds._plan, ds._sources
    full = run_with_rules(logical, sources, FULL_RULES)
    none = run_with_rules(logical, sources, NO_RULES)
    assert full.column("r").to_pylist() == none.column("r").to_pylist(), (
        f"optimizer changed {outer}({inner}(s)):\n"
        f"  optimized:   {full.column('r').to_pylist()}\n"
        f"  unoptimized: {none.column('r').to_pylist()}"
    )


# --- exhaustive: date part over date_trunc ------------------------------------

#: Every truncation unit the engine accepts, and every date part the optimizer has a
#: lift-through rule for. The rules turn on a hand-written granularity rank table, and a
#: single wrong rank silently returns a different date field -- so the whole cross-product
#: is enumerated rather than sampled.
_TRUNC_UNITS = [
    "microsecond",
    "millisecond",
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
    "decade",
    "century",
    "millennium",
]
_DATE_PARTS = [
    "second",
    "minute",
    "hour",
    "day",
    "day_of_week",
    "day_of_year",
    "isodow",
    "week",
    "iso_year",
    "month",
    "quarter",
    "year",
    "decade",
    "century",
    "millennium",
]


@pytest.mark.parametrize("unit", _TRUNC_UNITS)
@pytest.mark.parametrize("part", _DATE_PARTS)
def test_date_part_over_trunc_is_result_preserving(part: str, unit: str) -> None:
    """`part(date_trunc(unit, t))` must mean the same optimized and unoptimized.

    The lift-through rules fire only when the truncation is at or below the part's own
    granularity, and that ordering is a table I wrote by hand. `week` is the trap: a week
    nests inside neither a month nor a quarter, so truncating to a month genuinely moves
    which week an instant falls in. `epoch` is excluded from the rules entirely because it
    reports microseconds, which every truncation changes.

    Timestamps span pre-epoch, a leap day, a year boundary, and a month end, so an
    off-by-one in the rank table shows up as a changed field rather than passing on
    mid-month data that happens to agree.
    """
    table = pa.table(
        {
            "t": pa.array(
                [
                    "1969-07-20T20:17:40",
                    "2020-02-29T23:59:59",
                    "2021-01-01T00:00:00",
                    "2021-03-31T12:34:56",
                    None,
                ]
            )
        }
    )
    ds = bt.from_arrow(table).select(ts=col("t").cast("timestamp"))
    ds = ds.with_columns(r=getattr(col("ts").dt.truncate(unit).dt, part)())

    logical, sources = ds._plan, ds._sources
    try:
        none = run_with_rules(logical, sources, NO_RULES)
    except Exception:
        pytest.skip(f"engine rejects {part}(date_trunc('{unit}', t))")
    full = run_with_rules(logical, sources, FULL_RULES)
    assert full.column("r").to_pylist() == none.column("r").to_pylist(), (
        f"optimizer changed {part}(date_trunc('{unit}', t)):\n"
        f"  optimized:   {full.column('r').to_pylist()}\n"
        f"  unoptimized: {none.column('r').to_pylist()}"
    )
