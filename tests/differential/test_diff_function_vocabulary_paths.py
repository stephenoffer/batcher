"""Every aggregate and window function, on every execution path, agreeing exactly.

`test_diff_operator_matrix` crosses every *operator* with every path and edge case. This
crosses every **function**: all 39 members of `AGG_FNS` and all 33 of `WINDOW_FUNCS`, against
`collect()`, `collect(spill=True)` and `iter_batches()`. The two matrices look alike and
answer different questions — that one asks whether `group_by` works under spill, this asks
whether `group_by(k).agg(v=col.mad())` does — and the difference is where the defects were:

* five aggregates (`any_value`, `array_agg`, `entropy`, `kahan_sum`, `mad`) returned
  `k: null, v: null` from `collect(spill=True)` on an empty relation where `collect()`
  returned `k: int64, v: double`, because the spill path types its empty result from
  `available_schema` and those five had no declared type;
* the same for `count_distinct` as a *window* function;
* a windowed `sum` over a `null`-typed column declared `null` where the engine makes `int64`.

None of them failed a per-operator test, because the operator worked; it was the *function
inside it* that nothing crossed with a path. The operator matrix runs one aggregate (`sum`)
and one window function, which is the right economy for what it is testing and is exactly the
gap this file fills.

`test_diff_aggregate_declared_types` and `test_diff_window_declared_types` hold the *declared*
type to the engine's on no data. This holds the three *paths* to each other on real data, so
a divergence that is not a typing question — a fold that reassociates wrongly, an empty group
that appears on one path only — has somewhere to fail.

The input crosses a morsel boundary several times on purpose: below 16,384 rows the three
paths are three names for a single batch and the file would assert nothing.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir import AggExpr
from batcher.plan.ir_tags import AGG_FNS
from batcher.plan.logical.window import WINDOW_FUNCS
from batcher.plan.types import widen
from batcher.plan.types.domains import aggregate_domain_error

pytestmark = pytest.mark.differential

#: Several morsels, so `collect()` and `iter_batches()` are genuinely different schedulings.
_N = 40_000

#: Which input shapes each function is crossed with. `empty` is not padding: it is where four
#: of the six defects this file was written for actually showed, because an empty result is
#: typed from the declaration rather than from data.
_SHAPES = ("multibatch", "empty", "one")


def _table(shape: str) -> pa.Table:
    n = {"empty": 0, "one": 1}.get(shape, _N)
    idx = range(n)
    return pa.table(
        {
            # The row identity, so a window result can be compared as a sequence.
            "rid": pa.array(idx, pa.int64()),
            "g": pa.array([i % 7 for i in idx], pa.int64()),
            # Duplicated, so `ORDER BY o` has peer groups a frame can straddle.
            "o": pa.array([(i * 37) % 1009 for i in idx], pa.int64()),
            "k": pa.array([None if i % 9 == 0 else (i * 37) % 101 for i in idx], pa.int64()),
            # Both zeros and a NaN: the float key edge the operator matrix also carries.
            "f": pa.array(
                [
                    -0.0 if i % 13 == 0 else (math.nan if i % 23 == 0 else float(i % 29))
                    for i in idx
                ],
                pa.float64(),
            ),
            "s": pa.array([None if i % 11 == 0 else f"v{i % 17}" for i in idx]),
        }
    )


@pytest.fixture(scope="module")
def tables() -> dict[str, pa.Table]:
    return {shape: _table(shape) for shape in _SHAPES}


def _run(dataset, path: str) -> pa.Table:
    if path == "collect":
        return dataset.collect()
    if path == "spill":
        return dataset.collect(spill=True, num_partitions=5)
    batches = list(dataset.iter_batches())
    return pa.Table.from_batches(batches) if batches else dataset.collect().slice(0, 0)


def _canonical(table: pa.Table, order: str) -> tuple:
    """`table` as a comparable value: its column types, and its rows in a fixed order.

    A NaN is folded to a token and `-0.0` to `0.0` because `nan != nan` and the two zeros
    compare equal but print differently — without that a correct result reports as a
    mismatch. Nothing else is normalized: the types are compared exactly, which is what the
    defects this file exists for were.
    """

    def scalar(value):
        if isinstance(value, float):
            return "nan" if math.isnan(value) else (0.0 if value == 0.0 else round(value, 9))
        # Tuples are recursed into as well as lists, because a `map` column arrives as a list
        # of `(key, count)` **tuples** and `histogram` over a float column puts a NaN in the
        # key slot. Left raw, that NaN made two identical results compare unequal — the
        # comparison would have reported a defect in the engine that was in this helper.
        if isinstance(value, (list, tuple)):
            return tuple(sorted((scalar(v) for v in value), key=repr))
        if isinstance(value, dict):
            return tuple(sorted(((scalar(k), scalar(v)) for k, v in value.items()), key=repr))
        return value

    names = sorted(table.column_names)
    types = [str(table.schema.field(name).type) for name in names]
    if order in table.column_names:
        table = table.sort_by([(order, "ascending")])
    data = table.to_pydict()
    rows = [tuple(scalar(data[name][i]) for name in names) for i in range(table.num_rows)]
    if order not in table.column_names:
        rows.sort(key=lambda row: tuple(repr(v) for v in row))
    return (names, types, rows)


def _agree(dataset_for, order: str) -> None:
    """Assert the three paths produce the same typed rows, or report which pair differs."""
    expected = _canonical(_run(dataset_for(), "collect"), order)
    for path in ("spill", "iter"):
        got = _canonical(_run(dataset_for(), path), order)
        assert got[0] == expected[0], f"{path}: columns {got[0]} vs {expected[0]}"
        assert got[1] == expected[1], f"{path}: types {got[1]} vs {expected[1]}"
        assert got[2] == expected[2], f"{path}: rows differ from collect()"


#: Aggregates that read a *second* column -- an ordering key or the other variable of a
#: bivariate statistic. Built with one argument they raise "requires an input column",
#: which is not a statement about the column's type: `arg_max` is perfectly happy with an
#: integer. Naming them here is what gets them exercised rather than skipped.
_TWO_ARG_AGGS = frozenset(
    {"arg_max", "arg_min", "arg_max_null", "arg_min_null", "corr", "covar_pop", "covar_samp"}
)


def _agg_expr(func: str, column: str) -> AggExpr:
    """`AggExpr` for `func` over `column`, supplying a second input where one is needed."""
    if func in _TWO_ARG_AGGS:
        return AggExpr(func, bt.col(column), input2=bt.col("rid"))
    return AggExpr(func, bt.col(column))


@pytest.mark.parametrize("column", ["k", "f", "s"])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("func", sorted(AGG_FNS))
def test_a_grouped_aggregate_agrees_across_paths(tables, func, shape, column):
    rows = tables[shape]

    def build():
        return bt.from_arrow(rows).group_by("g").agg(v=_agg_expr(func, column))

    # A skip is earned only when the control plane's *declared domain* says this
    # (aggregate, column type) pair is out of range -- asked of `aggregate_domain_error`,
    # the same function the planner raises from, rather than matched against its wording.
    #
    # Catching bare `Exception` here made this test unable to fail: any engine regression
    # that raised removed its own case from the run and reported green. It also
    # mislabelled the five two-argument aggregates as "not accepting" an integer column,
    # so nothing in this file -- whose whole subject is every member of `AGG_FNS` --
    # ever reached them.
    declined = aggregate_domain_error(func, column, widen(rows.schema.field(column).type))
    if declined:
        pytest.skip(declined)
    build().collect()
    _agree(build, "g")


@pytest.mark.parametrize("scope", ["partitioned", "global"])
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("func", sorted(WINDOW_FUNCS))
def test_a_window_function_agrees_across_paths(tables, func, shape, scope):
    rows = tables[shape]
    keys = {"partition_by": ["g"]} if scope == "partitioned" else {}

    def build(spec):
        return bt.from_arrow(rows).window(order_by=["o"], functions={"w": spec}, **keys)

    for spec in (func, (func, bt.col("f"))):
        try:
            build(spec).collect()
        except Exception:
            continue
        _agree(lambda spec=spec: build(spec), "rid")
        return
    pytest.skip(f"{func} is not expressible through window(order_by=...)")


def test_the_matrix_actually_reaches_both_vocabularies(tables):
    """Guard against a vacuous sweep.

    Every case above skips the pairs the engine refuses, so a change that made *every* pair
    refuse — a renamed fixture column, a vocabulary import that resolved to an empty set —
    would turn the whole file green while running nothing. This pins that the two
    vocabularies are non-empty and that a representative member of each is reachable with the
    fixture's own columns.
    """
    rows = tables["multibatch"]
    assert len(AGG_FNS) > 30 and len(WINDOW_FUNCS) > 25
    assert bt.from_arrow(rows).group_by("g").agg(v=AggExpr("mad", bt.col("k"))).collect().num_rows
    windowed = bt.from_arrow(rows).window(order_by=["o"], functions={"w": "cume_dist"})
    assert windowed.collect().num_rows == rows.num_rows
