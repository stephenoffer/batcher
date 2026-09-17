"""Translations that were correct on the host backend and broken on a real device.

The two-backend design says a GPU is only *where* a translated plan runs, never *what* it
computes — but that only holds where both backends take the same path. Every case here is one
where they did not, found by replaying the whole vocabulary on a T4 fleet and comparing against
the engine:

* the **unary math functions** preferred a same-named cuDF method (`x.sqrt()`, `x.exp()`,
  `x.sin()`) behind an `is_gpu` gate. cuDF has none of them any more, so thirty-one functions
  raised `AttributeError` on every device and took the NumPy branch — the correct one — in
  every test. An `AttributeError` is not an `Unsupported`, so the query did not even decline
  gracefully: it reported the backend as broken;
* **`min_count=1`** is not implemented by cuDF, so a windowed `sum` raised there and passed
  here. The grouped aggregate had already learned this; the window had not;
* **`cumcount(ascending=False)`** is not implemented by cuDF either, which broke reading a list
  element from the end;
* **`null | true` is not `true`** on cuDF, so a Kleene-shaped guard folded to false and every
  vector distance declined on the device while running on the host;
* **neither library has a calendar-day type**, and the host backend's `astype` lands on
  `date32` anyway while the device's cannot, so a computed date came back as `timestamp[ms]`
  from a GPU and `date32` from CI.

Written against the host backend, because that is what CI has — each case pins the *shape* that
made the device diverge, so the construction cannot be reintroduced. What proves the fix is the
device replay itself; what this file prevents is the regression.
"""

from __future__ import annotations

import datetime as dt
import operator

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.backend import Unsupported
from batcher.core.gpu_plan.execute import run_chain
from batcher.plan.expr_ir.core import MathExpr
from batcher.plan.expr_ir.fn_names import MATH_FNS

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


NUMBERS = pa.table({"x": pa.array([1.0, 4.0, 0.5, None, 9.0], pa.float64())})


def _rows(table: pa.Table) -> list[tuple]:
    def canon(v):
        return float(f"{v:.12e}") if isinstance(v, float) and v == v else v

    return [tuple(canon(v) for v in r) for r in zip(*table.to_pydict().values(), strict=True)]


def _translated(ds, table: pa.Table, be) -> pa.Table:
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the translator declined a plan it is supposed to match"
    return be.to_arrow(run_chain(table, spec[1], be))


def _assert_matches_engine(ds, table: pa.Table, be) -> None:
    expected = ds.collect()
    got = _translated(ds, table, be).select(expected.column_names)
    assert _rows(got) == _rows(expected)
    assert got.schema.types == expected.schema.types


# --- the math vocabulary takes one path on both backends -------------------------------------


def test_no_math_function_is_reached_by_a_device_only_method():
    """The table is a ufunc per function, with no `is_gpu` branch left to diverge on.

    A per-backend preference here is what shipped thirty-one broken functions to a GPU fleet
    while every one of them passed on pandas — so the absence of the branch is the contract,
    not an implementation detail.
    """
    from batcher.core.gpu_plan.scalar_fns import _MATH_FNS

    assert all(isinstance(v, str) for v in _MATH_FNS.values())


#: The math functions the device tier does not translate. Each needs a host-side
#: construction neither backend dispatches to a device (a SWAR popcount, a per-row table,
#: a `libm` special function), so falling back to the CPU engine is the cheaper answer --
#: `scalar_fns` states the reason per function. Declining is the *contract* here, not a
#: gap: `.claude/rules/device-tier.md` #2 says decline rather than approximate.
DECLINED_MATH_FNS = frozenset({"bit_count", "factorial", "gamma", "lgamma"})

#: Everything else in the engine's math vocabulary, derived rather than hand-listed. This
#: used to be a list of nineteen names against a vocabulary of thirty-five, so `round`,
#: `sign`, `even` and the six reciprocal/inverse-circular functions -- the members whose
#: translations are *least* obvious, and the ones `scalar_fns` documents as traps -- were
#: the ones shipping unexercised.
TRANSLATED_MATH_FNS = tuple(sorted(MATH_FNS - DECLINED_MATH_FNS))

#: Inputs for the functions whose domain `NUMBERS` leaves. An out-of-domain argument is not
#: a harder case here, it is an unreadable one: both sides return NaN, NaN compares unequal
#: to itself, and the case fails while the two backends agree to the last bit.
_UNIT_INTERVAL = pa.table({"x": pa.array([1.0, 0.5, -0.5, None, -1.0], pa.float64())})
_AT_LEAST_ONE = pa.table({"x": pa.array([1.0, 4.0, 2.5, None, 9.0], pa.float64())})
_MATH_DOMAIN = {
    "acos": _UNIT_INTERVAL,
    "asin": _UNIT_INTERVAL,
    "atanh": _UNIT_INTERVAL,
    "acosh": _AT_LEAST_ONE,
}


def test_every_math_function_is_translated_or_declined():
    """The per-function half of the vocabulary contract.

    `test_gpu_vocabulary_contract` classifies the ``math`` *tag*, which is one entry for
    thirty-five functions -- so a function added to `MATH_FNS` with no translation is
    invisible to it. This is the check that makes such a function fail here instead.
    """
    assert set(TRANSLATED_MATH_FNS) | DECLINED_MATH_FNS == set(MATH_FNS)
    assert not set(TRANSLATED_MATH_FNS) & DECLINED_MATH_FNS


#: The inverse functions keep their SQL names as IR tags, and are spelled `arc*` on `Expr`.
_METHOD_NAME = {fn: "arc" + fn[1:] for fn in ("acos", "acosh", "asin", "asinh", "atan", "atanh")}


def _unary_math(fn: str) -> bt.Expr:
    """The `math` node for `fn`, through its `Expr` method where it has one.

    A tag can outlive its method: `rint` is still the IR the SQL `rint` lowers to, while the
    `Expr` spelling is `round(mode="half_to_even")`, which lowers to `round_even` instead.
    """
    name = _METHOD_NAME.get(fn, fn)
    if hasattr(bt.Expr, name):
        return getattr(col("x"), name)()
    return MathExpr(fn, col("x"))


@pytest.mark.parametrize("fn", TRANSLATED_MATH_FNS)
def test_a_unary_math_function_matches_the_engine(be, fn):
    table = _MATH_DOMAIN.get(fn, NUMBERS)
    ds = bt.from_arrow(table).select(out=_unary_math(fn))
    _assert_matches_engine(ds, table, be)


@pytest.mark.parametrize("fn", sorted(DECLINED_MATH_FNS))
def test_a_declined_math_function_declines_instead_of_guessing(be, fn):
    """The other half of contract #2: a decline, never a plausible wrong number.

    An `Unsupported` sends the stage to the CPU engine. Anything else -- a value, or a
    bare `AttributeError` -- is the failure this whole module was written after: the unary
    math table reported the *backend* as broken because a missing method raised the wrong
    exception type, and thirty-one functions took that path on every real device.
    """
    ds = bt.from_arrow(NUMBERS).select(out=getattr(col("x"), fn)())
    spec = gpu_plan_ops(ds._plan)
    if spec is None:
        return  # declined a step earlier, at plan match — equally a fallback to the CPU
    with pytest.raises(Unsupported, match=fn):
        run_chain(NUMBERS, spec[1], be)


# --- the window reductions avoid the keyword cuDF does not have -------------------------------


def test_no_windowed_reduction_asks_for_min_count():
    """cuDF raises `NotImplementedError` for `min_count` on every reduction that takes one, so
    the keyword that makes a windowed `sum` correct is the one that makes it impossible on a
    device. The grouped path learned this first; the window is now spelled the same way."""
    import ast
    from pathlib import Path

    from batcher.core.gpu_plan import windows

    tree = ast.parse(Path(windows.__file__).read_text())
    passed = [
        kw.arg for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords
    ]
    assert "min_count" not in passed


def test_a_windowed_sum_over_an_all_null_partition_is_null(be):
    """What `min_count` was there for, done by counting and masking instead."""
    table = pa.table(
        {
            "k": ["a", "a", "b", "b"],
            "v": pa.array([1.0, 2.0, None, None], pa.float64()),
        }
    )
    ds = bt.from_arrow(table).select("k", s=col("v").sum().over("k"))
    _assert_matches_engine(ds, table, be)


def test_a_windowed_aggregate_over_a_computed_input(be):
    """`sum(a * b) OVER (...)` is an ordinary weighted total, and a bare-column requirement
    declined the whole chain over the shape of one argument."""
    table = pa.table(
        {
            "k": ["a", "a", "b"],
            "a": pa.array([1.0, 2.0, 3.0], pa.float64()),
            "b": pa.array([10.0, 20.0, 30.0], pa.float64()),
        }
    )
    ds = bt.from_arrow(table).select("k", w=(col("a") * col("b")).sum().over("k"))
    _assert_matches_engine(ds, table, be)


@pytest.mark.parametrize("fn", ["zscore", "is_outlier", "maxabs_scale", "normalize_l1"])
def test_a_whole_column_statistic_reaches_the_device(be, fn):
    """Each of these lowers to a window over an *expression*, so each was declined whole."""
    ds = bt.from_arrow(NUMBERS).select(out=getattr(col("x"), fn)())
    _assert_matches_engine(ds, NUMBERS, be)


# --- the list vocabulary avoids the keyword and the Kleene assumption -------------------------


def test_reading_a_list_element_from_the_end(be):
    """`cumcount(ascending=False)` says this in one call and cuDF does not implement it."""
    table = pa.table({"v": pa.array([[1.0, 2.0, 3.0], [4.0], [], None], pa.list_(pa.float64()))})
    ds = bt.from_arrow(table).select(a=col("v").list.get(-1), b=col("v").list.get(-2))
    _assert_matches_engine(ds, table, be)


def test_a_vector_distance_over_a_column_carrying_a_null_list(be):
    """The length guard must not depend on `null | true` being `true` — cuDF's is null, so the
    guard folded to false and declined every distance on the device."""
    table = pa.table(
        {
            "a": pa.array([[1.0, 2.0], None, []], pa.list_(pa.float64())),
            "b": pa.array([[3.0, 4.0], None, []], pa.list_(pa.float64())),
        }
    )
    ds = bt.from_arrow(table).select(out=col("a").list.dot(col("b")))
    _assert_matches_engine(ds, table, be)


# --- three-valued logic is stated, not inherited ------------------------------------------------


#: Every combination of the three truth values, on both sides.
TRUTH = pa.table(
    {
        "a": pa.array([True, True, True, False, False, False, None, None, None], pa.bool_()),
        "b": pa.array([True, False, None, True, False, None, True, False, None], pa.bool_()),
    }
)


@pytest.mark.parametrize("fn", [operator.or_, operator.and_], ids=["or", "and"])
def test_three_valued_logic_matches_the_engine(be, fn):
    """`true OR unknown` is true and `false OR unknown` is unknown — pandas implements that on
    an Arrow column and cuDF does not, so it is computed rather than inherited."""
    ds = bt.from_arrow(TRUTH).select(out=fn(col("a"), col("b")))
    _assert_matches_engine(ds, TRUTH, be)


def test_a_filter_on_a_disjunction_keeps_the_rows_the_engine_keeps(be):
    """The consequence that costs rows rather than raising."""
    ds = bt.from_arrow(TRUTH).filter(col("a") | col("b"))
    _assert_matches_engine(ds, TRUTH, be)


def test_cut_bins_a_null_to_null(be):
    """`cut` lowers to a `CASE` whose `WHEN` is `is_null(x) or is_nan(x)`, so a non-Kleene `or`
    made it take the else arm for exactly the null rows it exists to catch — and return a
    bucket label where the engine returns null."""
    table = pa.table({"x": pa.array([1.0, -1.0, 5.0, None, float("nan")], pa.float64())})
    ds = bt.from_arrow(table).select(out=col("x").cut([0.0, 2.0]))
    _assert_matches_engine(ds, table, be)


# --- a computed date is presented as a date ----------------------------------------------------


INSTANTS = pa.table(
    {
        "ts": pa.array([dt.datetime(2024, 2, 29, 13, 45), None], pa.timestamp("us")),
        "d": pa.array([dt.date(2024, 2, 29), None], pa.date32()),
    }
)


@pytest.mark.parametrize("fn", ["date", "last_day"])
def test_a_computed_calendar_day_is_a_date(be, fn):
    """Neither library has a calendar-day type. The host backend's `astype` lands on `date32`
    anyway and the device's cannot, so this came back as `timestamp[ms]` from a GPU only."""
    ds = bt.from_arrow(INSTANTS).select(out=getattr(col("ts").dt, fn)())
    _assert_matches_engine(ds, INSTANTS, be)


def test_a_day_offset_keeps_its_input_calendar_type(be):
    """Offsetting a calendar day gives a calendar day; offsetting an instant gives an instant."""
    ds = bt.from_arrow(INSTANTS).select(
        a=col("d").dt.offset_by("1d"), b=col("ts").dt.offset_by("1d")
    )
    _assert_matches_engine(ds, INSTANTS, be)


def test_a_truncation_is_still_a_timestamp(be):
    """`date_trunc` returns an instant at midnight, not a calendar day — so the date claim must
    not spread to it."""
    ds = bt.from_arrow(INSTANTS).select(out=col("ts").dt.truncate("month"))
    _assert_matches_engine(ds, INSTANTS, be)


def test_a_later_projection_can_take_a_name_back_from_a_date(be):
    """The last projection to produce a name decides what that name is; a date read early must
    not keep casting a timestamp computed later under the same name."""
    ds = bt.from_arrow(INSTANTS).select(d=col("ts").dt.date()).select(d=col("d").dt.epoch())
    _assert_matches_engine(ds, INSTANTS, be)
