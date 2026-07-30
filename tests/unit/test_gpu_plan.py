"""The GPU plan translator matches the native CPU engine (verified on pandas, no GPU needed).

`gpu_plan_ops` detects a translatable chain; `run_chain` replays its RelOp/Expr IR on a
dataframe. Run against pandas here so the translation logic is CI-testable without a GPU; the
GPU path runs the identical code on cuDF. That equivalence is the whole safety argument for
the GPU backend, so these cases are its differential test — the oracle is Batcher's own CPU
engine, which is itself checked against DuckDB.

Comparison goes through Arrow rather than pandas' `equals`, because the two paths legitimately
produce different *pandas* dtypes for the same Arrow data (`int64` against `int64[pyarrow]`)
while agreeing exactly on the values. Every case where row order is part of the contract — a
sort, a window, a ranking — is compared row-for-row, since an order-independent check is
precisely what cannot see an ordering bug.

Anything the translator cannot compute exactly must raise `Unsupported` or return `None`, both
of which the dispatcher turns into a CPU-engine fallback. A wrong answer is the one outcome
that is not allowed.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import (
    DfBackend,
    Unsupported,
    gpu_join_spec,
    gpu_plan_ops,
    gpu_union_spec,
)
from batcher.core.gpu_plan.execute import run_chain, run_join, run_union

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


def _table():
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "x": rng.integers(0, 10, 300).astype("int64"),
            "y": rng.random(300),
            "z": rng.integers(0, 3, 300).astype("int64"),
        }
    )


def _nulls():
    """Nulls in a key, a measure, and a string, with an all-null group — but no `NaN`.

    `NaN` is deliberately absent: the engine orders it above every number while both dataframe
    libraries treat it as missing, so the translator declines a `NaN`-bearing reduction rather
    than approximate it. That decision has its own test below.
    """
    return pa.table(
        {
            "g": pa.array([1, 1, None, 2, 2, None, 3], type=pa.int64()),
            "v": pa.array([1.0, None, 3.0, 4.0, 5.0, None, None], type=pa.float64()),
            "s": pa.array(["a", None, "cc", "d", "a", "b", None], type=pa.string()),
        }
    )


def _signed():
    """Negative and zero values, where truncated and floored remainders disagree and where
    round-half-away-from-zero and round-half-to-even disagree."""
    return pa.table(
        {
            "a": pa.array([-7, 7, -7, 7, 0, -1], type=pa.int64()),
            "b": pa.array([3, 3, -3, -3, 5, 2], type=pa.int64()),
            "f": pa.array([-2.5, 2.5, -0.5, 0.0, 1.5, -1.5], type=pa.float64()),
        }
    )


def _temporal():
    """A timestamp column with a null, a leap day, and a year boundary."""
    return pa.table(
        {
            "t": pa.array(
                [
                    dt.datetime(2021, 3, 14, 15, 9, 26),
                    dt.datetime(2020, 2, 29),
                    None,
                    dt.datetime(1999, 12, 31, 23, 59, 59),
                ],
                type=pa.timestamp("us"),
            ),
            "s": pa.array(["hello world", "ab", None, ""], type=pa.string()),
        }
    )


def _texts():
    return pa.table(
        {
            "s": pa.array(["  Hello ", "world", None, "a-b-c", "", "MiXeD"], type=pa.string()),
            "g": pa.array([1, 1, 2, 2, 3, 3], type=pa.int64()),
        }
    )


def _canon(v):
    """`NaN` folded to a sentinel, floats compared to twelve significant digits.

    Relative, not absolute: a mergeable fold re-associates float arithmetic, which the project
    tolerates as "up to float summation order". An absolute `round(v, 9)` calls two values that
    differ in their last two bits unequal once they are large — a product that reaches 1e214
    fails on a difference of one part in 1e15.
    """
    if isinstance(v, float):
        return "__nan__" if v != v else float(f"{v:.12e}")
    return v


def _rows(table: pa.Table) -> list[tuple]:
    cols = table.to_pydict()
    return [tuple(_canon(v) for v in row) for row in zip(*cols.values(), strict=True)]


def _assert_matches(got_df, expected: pa.Table, be) -> None:
    """The translated result holds the same rows as the CPU engine's, in any order."""
    got = be.to_arrow(got_df).select(expected.column_names)
    assert sorted(_rows(got), key=repr) == sorted(_rows(expected), key=repr)


def _assert_same_order(got_df, expected: pa.Table, be) -> None:
    """...and row-for-row, for operators whose row order is part of the contract."""
    got = be.to_arrow(got_df).select(expected.column_names)
    assert _rows(got) == _rows(expected)


def _run(build, table, be):
    ds = build(bt.from_arrow(table))
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "shape should be GPU-translatable"
    return run_chain(table, spec[1], be), ds.collect()


# --- relational shapes ---------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.filter(col("y") > 0.5),
        lambda ds: ds.with_columns(w=col("y") * 2.0 + 1.0),
        lambda ds: ds.select("x", "y"),
        lambda ds: ds.filter(col("x") > 3).with_columns(w=col("y") - col("x")),
        lambda ds: ds.group_by("x", "z").agg(
            s=col("y").sum(), c=col("y").count(), m=col("y").mean()
        ),
        lambda ds: ds.select("x", "z").distinct(),
        lambda ds: ds.filter(col("y") > 0.3).group_by("x").agg(s=col("y").sum(), mx=col("y").max()),
        lambda ds: ds.limit(20, offset=5),
        # a computed group key — the optimizer leaves these in place, and rejecting them sent
        # the whole chain to the CPU engine
        lambda ds: ds.group_by(x2=col("x") + 1).agg(s=col("y").sum()),
        # a keyless aggregate, which is also the shape every distributed combine step takes
        lambda ds: ds.agg(s=col("y").sum(), n=col("y").count()),
        lambda ds: ds.group_by("x").agg(
            v=col("y").var(), sd=col("y").std(), med=col("y").median(), nd=col("y").count_distinct()
        ),
        lambda ds: ds.group_by("z").agg(q=col("y").quantile(0.25), p=col("y").product()),
        lambda ds: ds.group_by("z").agg(n=bt.count()),
        lambda ds: ds.group_by("z").agg(s=(col("y") * 2.0).sum()),
        lambda ds: ds.select(r=col("y").cast("int64")).group_by("r").agg(n=bt.count()),
    ],
)
def test_chain_matches_cpu_engine(build, be):
    got, exp = _run(build, _table(), be)
    _assert_matches(got, exp, be)


def test_deep_chain_matches_cpu_engine(be):
    """Eight operators deep — the shape a real query reaches the translator as."""
    got, exp = _run(
        lambda ds: (
            ds.filter(col("y") > 0.2)
            .with_columns(w=col("y") * 10.0)
            .filter(col("w") < 8.0)
            .group_by("x")
            .agg(s=col("w").sum(), n=bt.count())
            .filter(col("n") > 1)
            .sort("s", descending=True)
            .limit(5)
        ),
        _table(),
        be,
    )
    _assert_same_order(got, exp, be)


@pytest.mark.parametrize(
    "build",
    [
        # a null group key is a group; dropping it silently deletes rows from the answer
        lambda ds: ds.group_by("g").agg(s=col("v").sum(), c=col("v").count()),
        # the sum of an all-null group is null, not 0.0
        lambda ds: ds.group_by("g").agg(s=col("v").sum(), mn=col("v").min(), p=col("v").product()),
        lambda ds: ds.group_by("g").agg(nd=col("v").count_distinct(), md=col("v").median()),
        # a null predicate is not a match
        lambda ds: ds.filter(col("v") > 1.0),
        lambda ds: ds.filter(col("s").str.contains("a")),
        lambda ds: ds.distinct(),
    ],
)
def test_null_semantics_match_cpu_engine(build, be):
    got, exp = _run(build, _nulls(), be)
    _assert_matches(got, exp, be)


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.sort("g"),
        lambda ds: ds.sort("g", descending=True),
        lambda ds: ds.sort("v"),
        lambda ds: ds.sort("s", descending=True),
        lambda ds: ds.sort("g", "v", descending=[False, True]),
    ],
)
def test_sort_order_matches_cpu_engine(build, be):
    got, exp = _run(build, _nulls(), be)
    _assert_same_order(got, exp, be)


def test_sort_then_limit_matches_cpu_engine(be):
    got, exp = _run(lambda ds: ds.sort("y", descending=True).limit(10), _table(), be)
    _assert_same_order(got, exp, be)


# --- expressions ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        lambda: col("v").is_null(),
        lambda: col("v").is_not_null(),
        lambda: col("v").is_nan(),
        lambda: ~(col("v") > 1.0),
        lambda: col("g").cast("float64"),
        lambda: bt.coalesce(col("v"), col("g").cast("float64")),
        lambda: bt.greatest(col("v"), col("g").cast("float64")),
        lambda: bt.least(col("v"), col("g").cast("float64")),
        lambda: bt.nullif(col("g"), 1),
        lambda: bt.when(col("v") > 1.0).then(col("v")).otherwise(col("v") * -1.0),
        lambda: col("g").is_in([1, 3]),
        lambda: col("v").abs(),
        lambda: col("v").sqrt(),
        lambda: col("v").floor(),
        lambda: col("v").ceil(),
        lambda: col("v").round(1),
        lambda: col("v").pow(2.0),
        lambda: col("v").sign(),
        lambda: col("v").exp(),
        lambda: col("v").log10(),
        lambda: col("s").str.upper(),
        lambda: col("s").str.lower(),
        lambda: col("s").str.len(),
        lambda: col("s").str.contains("a"),
        lambda: col("s").str.starts_with("a"),
        lambda: col("s").str.ends_with("c"),
        lambda: col("s").str.replace("a", "Z"),
        lambda: col("s").str.substr(1, 2),
        lambda: col("g") % 2,
        lambda: col("g") // 2,
        lambda: col("g") * 3 - 1,
    ],
)
def test_expression_matches_cpu_engine(expr, be):
    got, exp = _run(lambda ds: ds.select(r=expr()), _nulls(), be)
    _assert_same_order(got, exp, be)


@pytest.mark.parametrize(
    "expr",
    [
        # `%` takes the sign of the DIVIDEND in the engine and of the divisor in Python
        lambda: col("a") % col("b"),
        lambda: col("a") % 3,
        lambda: col("a") // col("b"),
        lambda: col("a").abs(),
        # halves round AWAY FROM ZERO in the engine and to EVEN in both dataframe libraries
        lambda: col("f").round(0),
        lambda: col("f").trunc(),
        lambda: col("f").floor(),
        lambda: col("f").ceil(),
        lambda: col("f").sign(),
        lambda: col("f").cast("int64"),
    ],
)
def test_signed_arithmetic_matches_cpu_engine(expr, be):
    got, exp = _run(lambda ds: ds.select(r=expr()), _signed(), be)
    _assert_same_order(got, exp, be)


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.select(r=col("s").str.strip()),
        lambda ds: ds.select(r=col("s").str.upper()),
        lambda ds: ds.select(r=col("s").str.len()),
        # SQL `substring` is 1-based and inclusive; a 0-based slice returns a shifted window
        lambda ds: ds.select(r=col("s").str.substr(1, 3)),
        lambda ds: ds.select(r=col("s").str.substr(2, 2)),
        lambda ds: ds.filter(col("s").str.contains("o")),
        lambda ds: ds.filter(col("s") == "world"),
        lambda ds: ds.group_by("g").agg(n=col("s").count(), d=col("s").count_distinct()),
        # `distinct(subset=)` lowers to window + filter + project, so it exercises all three
        lambda ds: ds.distinct(subset=["g"]),
    ],
)
def test_string_operations_match_cpu_engine(build, be):
    got, exp = _run(build, _texts(), be)
    _assert_matches(got, exp, be)


@pytest.mark.parametrize(
    "expr",
    [
        lambda: col("t").dt.year(),
        lambda: col("t").dt.month(),
        lambda: col("t").dt.day(),
        lambda: col("t").dt.hour(),
        lambda: col("t").dt.quarter(),
        # the engine numbers the week from Sunday; both backends number it from Monday
        lambda: col("t").dt.dayofweek(),
        lambda: col("t").dt.dayofyear(),
        # ...and `week` is the ISO week, which is a calculation rather than an attribute, and
        # which fills a null timestamp with zero instead of propagating it
        lambda: col("t").dt.week(),
        lambda: col("t").dt.is_leap_year(),
        lambda: col("t").dt.days_in_month(),
        lambda: col("t").dt.epoch(),
        # `lpad`/`rpad` TRUNCATE as well as pad; `rjust`/`ljust` only ever pad
        lambda: col("s").str.lpad(6, "0"),
        lambda: col("s").str.rpad(6, "0"),
        lambda: col("s").str.repeat(2),
        # SQL `position` is 1-based and reports 0 for "not found"; `find` is 0-based and -1
        lambda: col("s").str.position("o"),
        lambda: col("s").str.right(2),
        lambda: col("s").str.initcap(),
        # the fixed-duration truncations, which are a floor
        lambda: col("t").dt.truncate("day"),
        lambda: col("t").dt.truncate("hour"),
        lambda: col("t").dt.truncate("minute"),
        lambda: col("t").dt.truncate("second"),
        lambda: col("t").dt.strftime("%Y-%m"),
    ],
)
def test_temporal_and_string_functions_match_cpu_engine(expr, be):
    got, exp = _run(lambda ds: ds.select(r=expr()), _temporal(), be)
    _assert_same_order(got, exp, be)


def test_a_date_function_can_drive_a_filter_and_a_group_key(be):
    got, exp = _run(lambda ds: ds.filter(col("t").dt.year() > 2000), _temporal(), be)
    _assert_matches(got, exp, be)
    got, exp = _run(lambda ds: ds.group_by(y=col("t").dt.year()).agg(n=bt.count()), _temporal(), be)
    _assert_matches(got, exp, be)


def test_typed_literal_survives_the_wire_form(be):
    """A date literal rides the IR as days-since-epoch; comparing the raw integer is wrong."""
    table = pa.table({"d": pa.array([dt.date(2020, 1, 1), dt.date(2021, 6, 1)], type=pa.date32())})
    got, exp = _run(lambda ds: ds.filter(col("d") > dt.date(2020, 6, 1)), table, be)
    _assert_matches(got, exp, be)


def test_nested_case_matches_cpu_engine(be):
    got, exp = _run(
        lambda ds: ds.select(
            r=bt.when(col("y") > 0.7)
            .then(bt.lit("hi"))
            .when(col("y") > 0.3)
            .then(bt.lit("mid"))
            .otherwise(bt.lit("lo"))
        ),
        _table(),
        be,
    )
    _assert_same_order(got, exp, be)


# --- window functions ----------------------------------------------------------------


@pytest.mark.parametrize(
    "function", ["row_number", "rank", "dense_rank", "percent_rank", "cume_dist", "ntile"]
)
def test_ranking_window_matches_cpu_engine(function, be):
    got, exp = _run(
        lambda ds: ds.window(
            partition_by=["x"], order_by=[("z", False), ("y", False)], functions={"r": function}
        ),
        _table(),
        be,
    )
    _assert_same_order(got, exp, be)


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.window(partition_by=["z"], order_by=[("y", True)], functions={"r": "rank"}),
        lambda ds: ds.window(
            partition_by=["z"],
            order_by=[("x", False), ("y", True)],
            functions={"a": "row_number", "b": "dense_rank"},
        ),
        # no partition at all — the whole frame is one partition
        lambda ds: ds.window(
            partition_by=[], order_by=[("y", False)], functions={"r": "row_number"}
        ),
        lambda ds: ds.window(
            partition_by=["x"], order_by=[("y", False)], functions={"r": ("first_value", "y")}
        ),
        lambda ds: ds.window(
            partition_by=["x"], order_by=[("y", False)], functions={"r": ("last_value", "y")}
        ),
        lambda ds: ds.window(
            partition_by=["x"], order_by=[("y", False)], functions={"r": ("nth_value", "y")}
        ),
        lambda ds: ds.window(
            partition_by=["x"], order_by=[("y", False)], functions={"r": ("lag", "y")}
        ),
        lambda ds: ds.window(
            partition_by=["x"],
            order_by=[("y", False)],
            functions={"r": ("sum", "y")},
            frame=(None, 0),
        ),
    ],
)
def test_window_variants_match_cpu_engine(build, be):
    got, exp = _run(build, _table(), be)
    _assert_same_order(got, exp, be)


@pytest.mark.parametrize(
    "build",
    [
        lambda ds: ds.select(r=col("v").shift(1).over(partition_by=["g"], order_by=["v"])),
        lambda ds: ds.select(r=col("v").shift(-1).over(partition_by=["g"], order_by=["v"])),
        lambda ds: ds.select(r=col("v").sum().over(partition_by=["g"])),
        lambda ds: ds.select(r=col("v").mean().over(partition_by=["g"])),
        lambda ds: ds.select(r=col("v").min().over(partition_by=["g"])),
        lambda ds: ds.select(r=col("v").max().over(partition_by=["g"])),
        lambda ds: ds.select(r=col("v").count().over(partition_by=["g"])),
        # an ORDER BY with no explicit frame makes the aggregate RUNNING, not whole-partition
        lambda ds: ds.select(r=col("v").sum().over(partition_by=["g"], order_by=["v"])),
        lambda ds: ds.select(r=col("v").count().over(partition_by=["g"], order_by=["v"])),
        lambda ds: ds.window(
            partition_by=["g"], order_by=[("v", False)], functions={"r": ("forward_fill", "s")}
        ),
        lambda ds: ds.window(
            partition_by=["g"], order_by=[("v", False)], functions={"r": ("backward_fill", "s")}
        ),
    ],
)
def test_value_and_aggregate_windows_match_cpu_engine(build, be):
    got, exp = _run(build, _nulls(), be)
    _assert_same_order(got, exp, be)


@pytest.mark.parametrize(
    "build",
    [
        # a moving window — most of what a time series is ever asked for, and a shape the
        # translator used to decline outright
        lambda ds: ds.select(r=col("v").rolling_sum(2, partition_by=["g"], order_by=["v"])),
        lambda ds: ds.select(r=col("v").rolling_mean(2, partition_by=["g"], order_by=["v"])),
        lambda ds: ds.select(r=col("v").rolling_max(3, partition_by=["g"], order_by=["v"])),
        lambda ds: ds.select(r=col("v").rolling_min(3, partition_by=["g"], order_by=["v"])),
        lambda ds: ds.select(r=col("v").rolling_count(2, partition_by=["g"], order_by=["v"])),
        # a one-row window lowers as CURRENT ROW -> CURRENT ROW, not as 0 PRECEDING
        lambda ds: ds.select(r=col("v").rolling_sum(1, partition_by=["g"], order_by=["v"])),
    ],
)
def test_rolling_frames_match_cpu_engine(build, be):
    got, exp = _run(build, _nulls(), be)
    _assert_same_order(got, exp, be)


def test_window_preserves_input_row_order(be):
    """The engine's Window keeps every input row in the order it arrived; so must this path."""
    table = pa.table({"g": [1, 2, 1, 2], "v": [10.0, 20.0, 30.0, 40.0]})
    got, exp = _run(
        lambda ds: ds.window(
            partition_by=["g"], order_by=[("v", True)], functions={"r": "row_number"}
        ),
        table,
        be,
    )
    _assert_same_order(got, exp, be)


def test_window_then_filter_matches_cpu_engine(be):
    got, exp = _run(
        lambda ds: ds.window(
            partition_by=["z"], order_by=[("y", False)], functions={"rn": "row_number"}
        ).filter(col("rn") <= 3),
        _table(),
        be,
    )
    _assert_matches(got, exp, be)


# --- joins and unions ----------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda a, b: a.join(b, on="id", how="inner"),
        lambda a, b: a.join(b, on="id", how="left"),
        lambda a, b: a.join(b, on="id", how="right"),
        lambda a, b: a.join(b, on="id", how="outer"),
        lambda a, b: a.join(b, on="id", how="semi"),
        lambda a, b: a.join(b, on="id", how="anti"),
        lambda a, b: a.join(b, on="id", how="inner").filter(col("w") > 100),
        lambda a, b: a.join(b, on="id", how="inner").group_by("w").agg(s=col("v").sum()),
        # chains pushed BELOW the join — the shape the optimizer actually produces
        lambda a, b: a.filter(col("v") > 1.0).join(b.filter(col("w") < 300), on="id"),
        lambda a, b: a.with_columns(v2=col("v") * 2.0).join(b.filter(col("w") > 100), on="id"),
    ],
)
def test_join_matches_cpu_engine(build, be):
    fact = pa.table({"id": np.array([1, 2, 3, 1, 2], "int64"), "v": np.array([1.0, 2, 3, 4, 5])})
    dim = pa.table({"id": np.array([1, 2, 3], "int64"), "w": np.array([100, 200, 300], "int64")})
    ds = build(bt.from_arrow(fact), bt.from_arrow(dim))
    spec = gpu_join_spec(ds._plan)
    assert spec is not None
    (ls, lops), (rs, rops), jir, ops = spec
    lt = fact if ls.source_id == 0 else dim
    rt = dim if rs.source_id == 1 else fact
    got = run_join(lt, rt, lops, rops, jir, ops, be)
    _assert_matches(got, ds.collect(), be)


@pytest.mark.parametrize(
    "build",
    [
        lambda a, b: a.union(b),
        lambda a, b: a.union(b).filter(col("x") > 2),
        lambda a, b: a.union(b).group_by("x").agg(s=col("y").sum()),
        lambda a, b: a.filter(col("x") > 1).union(b),
    ],
)
def test_union_matches_cpu_engine(build, be):
    a = pa.table({"x": np.array([1, 2, 3], "int64"), "y": np.array([1.0, 2, 3])})
    b = pa.table({"x": np.array([3, 4], "int64"), "y": np.array([3.0, 4])})
    ds = build(bt.from_arrow(a), bt.from_arrow(b))
    spec = gpu_union_spec(ds._plan)
    assert spec is not None
    inputs, distinct, ops = spec
    tabs = [a if s.source_id == 0 else b for s, _ in inputs]
    got = run_union(tabs, [o for _, o in inputs], distinct, ops, be)
    _assert_matches(got, ds.collect(), be)


# --- the fallback contract -----------------------------------------------------------


def test_unsupported_shapes_return_none():
    t = _table()
    ds = bt.from_arrow(t)
    # a map_batches UDF is Python-only, never lowered to the engine IR
    assert gpu_plan_ops(ds.map_batches(lambda b: b)._plan) is None
    # a join is two-source, so the linear matcher must decline it
    assert gpu_plan_ops(ds.join(bt.from_arrow(t), on="x")._plan) is None
    # a bare scan has no operators to run on the device
    assert gpu_plan_ops(ds._plan) is None


def test_unsupported_aggregate_declines():
    """An aggregate the translator cannot compute must decline the node, not guess."""
    ds = bt.from_arrow(_table())
    assert gpu_plan_ops(ds.group_by("x").agg(m=col("y").mode())._plan) is None


def test_nan_bearing_aggregate_declines(be):
    """`NaN` is the largest value to the engine and missing to both dataframe libraries.

    Reconciling that per reduction is not worth the risk, so the translator declines and the
    stage runs on the CPU engine. What must never happen is the plausible wrong number: a
    `max` that silently returns the largest *finite* value.
    """
    table = pa.table({"g": [1, 1], "v": pa.array([float("nan"), 1.0], type=pa.float64())})
    ds = bt.from_arrow(table).group_by("g").agg(m=col("v").max())
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the shape is translatable; only this data is not"
    with pytest.raises(Unsupported):
        run_chain(table, spec[1], be)


def test_nan_comparison_follows_the_engine(be):
    """`NaN > x` is True in the engine (it orders above every number) and False under IEEE."""
    table = pa.table({"v": pa.array([float("nan"), 1.0, 3.0], type=pa.float64())})
    got, exp = _run(lambda ds: ds.filter(col("v") > 2.0), table, be)
    _assert_matches(got, exp, be)
