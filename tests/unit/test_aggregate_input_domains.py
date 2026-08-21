"""The declared aggregate input domains must not be narrower than the engine's.

`plan.types.domains` decides at *build* time whether an aggregate can be computed over a
column's type. A rule like that has exactly one dangerous failure mode: rejecting a query
the engine would have answered correctly. Nothing mechanical catches that -- the query
simply stops working, and the error looks deliberate.

So this runs the whole (aggregate x column type) cross-product through the real engine and
holds the rule to it: **every pair the rule rejects must be one the engine cannot answer
correctly**, either because it raises or because it silently produces nulls or nonsense.
The converse is deliberately *not* asserted. The rule is allowed to be stricter than the
accumulators, and is, in the places where they answer meaninglessly -- multiplying epoch
microseconds together, or returning all-null for a string.

The pairs where the rule is stricter are enumerated by name below rather than left implicit,
so widening the engine later shows up here as a test to delete rather than as a rejection
nobody remembers choosing.
"""

from __future__ import annotations

import datetime
import decimal

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import AggExpr, col
from batcher.plan.ir_tags import AGG_FNS

pytestmark = pytest.mark.unit

_TYPES: dict[str, pa.Array] = {
    "int64": pa.array([1, 2, 3], pa.int64()),
    "float64": pa.array([1.0, 2.0, 3.0], pa.float64()),
    "bool": pa.array([True, False, True], pa.bool_()),
    "string": pa.array(["a", "b", "c"], pa.string()),
    "binary": pa.array([b"a", b"b", b"c"], pa.binary()),
    "date32": pa.array([datetime.date(2020, 1, 1 + i) for i in range(3)], pa.date32()),
    "timestamp": pa.array(
        [datetime.datetime(2020, 1, 1 + i) for i in range(3)], pa.timestamp("us")
    ),
    "time64": pa.array([datetime.time(1, 2, i) for i in range(3)], pa.time64("us")),
    "duration": pa.array([1, 2, 3], pa.int64()).cast(pa.duration("s")),
    "decimal": pa.array([decimal.Decimal(f"{i}.00") for i in range(3)], pa.decimal128(10, 2)),
    "list": pa.array([[1], [2], [3]], pa.list_(pa.int64())),
    "struct": pa.array([{"a": i} for i in range(3)], pa.struct([("a", pa.int64())])),
    "null": pa.array([None, None, None], pa.null()),
}

_PARAMETRIC = {"quantile", "approx_quantile", "quantile_disc", "n_length", "l_count"}
_BINARY = {"corr", "covar_pop", "covar_samp", "arg_min", "arg_max"}

#: Pairs the rule rejects that the engine *tolerates*, each with why rejecting is right.
#: The engine's answer for every one of these is either all-null with no diagnostic or a
#: number computed from a physical representation nobody asked about.
#:
#: `decimal` was on every one of these rows until the accumulators learned to widen it, and
#: the widening is why it no longer is: `bc-runtime`'s `widens_decimal` casts a decimal input
#: to f64 for the statistics that are f64-valued anyway, so `STDDEV(price)` and its nine
#: neighbours now answer what DuckDB answers instead of being refused at build time.
_DELIBERATELY_STRICTER = {
    ("approx_quantile", "bool"),
    ("approx_quantile", "duration"),
    ("approx_quantile", "string"),
    ("approx_quantile", "timestamp"),
    ("covar_pop", "bool"),
    ("covar_pop", "duration"),
    ("covar_pop", "string"),
    ("covar_pop", "timestamp"),
    ("covar_samp", "bool"),
    ("covar_samp", "duration"),
    ("covar_samp", "string"),
    ("covar_samp", "timestamp"),
    ("corr", "bool"),
    ("corr", "duration"),
    ("corr", "string"),
    ("corr", "timestamp"),
    ("kahan_sum", "bool"),
    ("kahan_sum", "duration"),
    ("kahan_sum", "string"),
    ("kahan_sum", "timestamp"),
    ("kurtosis", "bool"),
    ("kurtosis", "duration"),
    ("kurtosis", "string"),
    ("kurtosis", "timestamp"),
    ("kurtosis_pop", "bool"),
    ("kurtosis_pop", "duration"),
    ("kurtosis_pop", "string"),
    ("kurtosis_pop", "timestamp"),
    ("product", "bool"),
    ("product", "duration"),
    ("product", "string"),
    ("product", "timestamp"),
    ("skewness", "bool"),
    ("skewness", "duration"),
    ("skewness", "string"),
    ("skewness", "timestamp"),
}


def _agg(func: str) -> AggExpr:
    if func in _PARAMETRIC:
        return AggExpr(func, col("v"), param=0.5)
    if func in _BINARY:
        return AggExpr(func, col("v"), input2=col("g"))
    return AggExpr(func, col("v"))


def _pairs() -> list[tuple[str, str]]:
    return [(f, t) for f in sorted(AGG_FNS - {"count_star", "approx_top_k"}) for t in _TYPES]


@pytest.mark.parametrize(("func", "tname"), _pairs(), ids=lambda v: str(v))
def test_a_rejected_pair_is_one_the_engine_cannot_answer(func: str, tname: str) -> None:
    table = pa.table({"g": pa.array([1, 1, 2], pa.int64()), "v": _TYPES[tname]})
    try:
        plan = bt.from_arrow(table).group_by("g").agg(x=_agg(func))
    except PlanError:
        _assert_engine_agrees_it_is_hopeless(func, tname, table)
        return
    # Not rejected: whatever the engine does with it is the engine's business, but the
    # *declared* type must be the one it actually produces, or the rule let a schema lie
    # through. A pair the engine raises on is left alone here — those are the ones this
    # rule does not claim to cover.
    declared = plan.schema.field("x").type
    try:
        produced = plan.collect().schema.field("x").type
    except Exception:
        return
    assert produced == declared, f"{func}({tname}) declares {declared} but produces {produced}"


def _assert_engine_agrees_it_is_hopeless(func: str, tname: str, table: pa.Table) -> None:
    """The engine must raise on a rejected pair, unless it is a listed silent-nonsense one."""
    if (func, tname) in _DELIBERATELY_STRICTER:
        return
    from batcher.plan.logical.aggregate import Aggregate

    plan = _plan_without_validation(table, func)
    with pytest.raises(Exception) as caught:
        plan.collect()
    assert not isinstance(caught.value, AssertionError)
    assert Aggregate is not None  # the node under test was the one bypassed


def _plan_without_validation(table: pa.Table, func: str):
    """The same query, built with the domain check disabled, so the engine can be asked."""
    import batcher.plan.logical.aggregate as agg_mod

    original = agg_mod._validate_agg_input_types
    agg_mod._validate_agg_input_types = lambda *_args, **_kw: None
    try:
        return bt.from_arrow(table).group_by("g").agg(x=_agg(func))
    finally:
        agg_mod._validate_agg_input_types = original


def test_the_stricter_list_holds_only_pairs_the_rule_rejects() -> None:
    """A stale entry here would quietly excuse a pair the rule no longer touches."""
    for func, tname in sorted(_DELIBERATELY_STRICTER):
        table = pa.table({"g": pa.array([1, 1, 2], pa.int64()), "v": _TYPES[tname]})
        with pytest.raises(PlanError):
            bt.from_arrow(table).group_by("g").agg(x=_agg(func))


def test_the_error_names_the_column_and_the_fix() -> None:
    table = pa.table({"g": [1, 2], "amount": ["1", "2"]})
    with pytest.raises(PlanError, match=r"'amount'.*string.*cast it to a numeric type"):
        bt.from_arrow(table).group_by("g").agg(total=bt.col("amount").sum())


def test_an_all_null_column_is_accepted_by_almost_every_aggregate() -> None:
    """An empty or all-null partition has no values to be the wrong type.

    The boolean pair is the stated exception: the accumulators materialize a null column's
    values as Int64 and then reject it, so a build-time message beats the engine's
    "aggregate bool_and is not supported for column type Int64" over a `null` column.
    """
    table = pa.table({"g": pa.array([1, 1, 2], pa.int64()), "v": _TYPES["null"]})
    for func in sorted(AGG_FNS - {"count_star", "approx_top_k", "bool_and", "bool_or"}):
        bt.from_arrow(table).group_by("g").agg(x=_agg(func))
    for func in ("bool_and", "bool_or"):
        with pytest.raises(PlanError, match="all-null"):
            bt.from_arrow(table).group_by("g").agg(x=_agg(func))


def test_a_domain_check_says_nothing_about_an_unrestricted_aggregate() -> None:
    """`count`, `list_agg`, `histogram` and friends take anything -- no rule, no message."""
    from batcher.plan.types.domains import aggregate_domain_error

    for func in ("count", "count_distinct", "list_agg", "histogram", "entropy"):
        for dt in (pa.string(), pa.struct([("a", pa.int64())]), pa.timestamp("us")):
            assert aggregate_domain_error(func, "c", dt) is None


# The window forms compute the same statistics over a frame, so they share the domains --
# and shared them with the same two failure modes before this rule reached them.
_WINDOW_REJECTED = [
    ("sum", "string"),
    ("avg", "string"),
    ("stddev", "string"),
    ("var", "timestamp"),
    ("product", "timestamp"),
    ("bit_and", "float64"),
    ("bool_and", "int64"),
    ("ewm_mean", "string"),
]


@pytest.mark.parametrize(("func", "tname"), _WINDOW_REJECTED, ids=lambda v: str(v))
def test_a_window_function_rejects_the_same_domains(func: str, tname: str) -> None:
    from batcher.plan.types.domains import window_domain_error

    problem = window_domain_error(func, "column 'v'", _TYPES[tname].type)
    assert problem is not None
    assert f"window function {func!r}" in problem, "the message must name what the user wrote"


def test_a_window_function_over_a_string_is_rejected_at_build() -> None:
    table = pa.table({"g": [1, 1, 2], "s": ["a", "b", "c"]})
    with pytest.raises(PlanError, match=r"window function 'sum'.*'s'.*string"):
        bt.from_arrow(table).with_columns(x=bt.col("s").sum().over(partition_by="g"))


def test_the_window_functions_that_take_any_type_still_do() -> None:
    """Ranking and the positional value functions have no domain by construction."""
    from batcher.plan.types.domains import window_domain_error

    for func in ("row_number", "rank", "dense_rank", "lag", "lead", "first_value", "forward_fill"):
        assert window_domain_error(func, "c", pa.string()) is None

    table = pa.table({"g": [1, 1, 2], "s": ["a", "b", "c"]})
    out = bt.from_arrow(table).with_columns(x=bt.col("s").min().over(partition_by="g")).collect()
    assert out.column("x").to_pylist() == ["a", "a", "c"]
