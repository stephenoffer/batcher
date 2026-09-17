"""The aggregate parameters that restore another engine's meaning, and the W0 bugfixes, vs DuckDB.

Every default here is DuckDB's, and each parameterised form is checked against the DuckDB
expression that spells the same meaning: ``coalesce(sum(x), 0)`` for ``sum(empty_value=0)``,
``arg_max_null`` for ``arg_max(ignore_nulls=False)``, a sorted ``list`` indexed at
``floor``/``ceil``/``round`` of the rank for the Polars quantile interpolations. The
competitor side of the same claim is `test_diff_agg_competitor_params.py`.

Every case runs through four schedulings of one semantics (invariant #7): ``collect()``, a
spilled collect with a forced bucket count, ``iter_batches()``, and a ``repartition(4)``
input whose partials merge. Each input carries an all-null group, a one-row group, NaN,
``-0.0``, duplicates and ties, and the ``multibatch`` shape crosses the morsel boundary so
the four paths are genuinely different.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


def _table(n: int) -> pa.Table:
    """`n` rows over five groups plus a one-row group ``solo``.

    Group ``b`` is null in every value column. ``x`` carries NaN (group ``d``), ``-0.0`` and
    duplicates; ``y`` is strictly positive (for the entropy of values, whose DuckDB oracle
    takes a logarithm); ``i`` is a nullable integer with many ties; ``k`` is a unique order key.
    """
    groups = ["a", "b", "c", "d", "e"]
    g, x, y, i, flag = [], [], [], [], []
    for r in range(n):
        grp = groups[r % 5]
        g.append(grp)
        dead = grp == "b"
        if dead or r % 7 == 0:
            x.append(None)
        elif grp == "d" and r % 3 == 0:
            x.append(float("nan"))
        elif r % 13 == 1:
            x.append(-0.0)
        else:
            x.append(float((r * 37) % 11 - 3))
        y.append(None if dead or r % 6 == 0 else float((r * 17) % 9 + 1))
        i.append(None if dead or r % 4 == 0 else (r * 7) % 5)
        flag.append(None if dead or r % 9 == 0 else r % 3 == 0)
    g.append("solo")
    x.append(2.5)
    y.append(4.0)
    i.append(3)
    flag.append(True)
    return pa.table(
        {
            "k": pa.array(range(n + 1), pa.int64()),
            "g": pa.array(g),
            "x": pa.array(x, pa.float64()),
            "y": pa.array(y, pa.float64()),
            "i": pa.array(i, pa.int64()),
            "flag": pa.array(flag, pa.bool_()),
        }
    )


INPUTS = {
    "base": _table(40),
    "empty": _table(40).slice(0, 0),
    "multibatch": _table(40_000),
}


def _stream(ds) -> pa.Table:
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches, schema=batches[0].schema)


PATHS = {
    "collect": lambda ds: ds.collect(),
    "spill": lambda ds: ds.collect(spill=True, num_partitions=4),
    "iter_batches": _stream,
    "repartitioned": lambda ds: ds.repartition(4).collect(),
}


def _modes_sql(value: str, keyed: bool) -> str:
    """DuckDB: every most-frequent non-null value, ascending, as a list (null when none)."""
    if not keyed:
        counts = f"SELECT {value} AS v, count(*) AS n FROM t WHERE {value} IS NOT NULL GROUP BY v"
        return (
            f"WITH c AS ({counts}) "
            "SELECT (SELECT list_sort(list(v)) FROM c WHERE n = (SELECT max(n) FROM c)) AS r"
        )
    counts = f"SELECT g, {value} AS v, count(*) AS n FROM t WHERE {value} IS NOT NULL GROUP BY g, v"
    return (
        f"WITH c AS ({counts}) SELECT grp.g, (SELECT list_sort(list(c.v)) FROM c "
        "WHERE c.g IS NOT DISTINCT FROM grp.g AND c.n = (SELECT max(c2.n) FROM c c2 "
        "WHERE c2.g IS NOT DISTINCT FROM grp.g)) AS r FROM (SELECT DISTINCT g FROM t) grp"
    )


def _quantile_sql(q: float) -> dict[str, str]:
    lst = "list_sort(list(x) FILTER (WHERE x IS NOT NULL))"
    pos = f"((len({lst}) - 1) * {q!r})"
    lower = f"{lst}[CAST(floor({pos}) AS BIGINT) + 1]"
    higher = f"{lst}[CAST(ceil({pos}) AS BIGINT) + 1]"
    return {
        "lower": lower,
        "higher": higher,
        "nearest": f"{lst}[CAST(round({pos}) AS BIGINT) + 1]",
        "midpoint": f"({lower} + {higher}) / 2",
    }


#: name -> (batcher aggregate, DuckDB aggregate expression). Each runs grouped by ``g`` and
#: over the whole input.
_Q = _quantile_sql(0.3)
AGGREGATES: dict[str, tuple] = {
    # Bugfix: the quantile used to be rounded to a permille, so 0.1234 answered 0.123.
    "quantile_full_precision": (col("x").quantile(0.1234), "quantile_cont(x, 0.1234)"),
    "quantile_disc_full_precision": (col("i").quantile_disc(0.6667), "quantile_disc(i, 0.6667)"),
    "quantile_default_linear": (col("x").quantile(0.3), "quantile_cont(x, 0.3)"),
    "quantile_lower": (col("x").quantile(0.3, "lower"), _Q["lower"]),
    "quantile_higher": (col("x").quantile(0.3, "higher"), _Q["higher"]),
    "quantile_nearest": (col("x").quantile(0.3, "nearest"), _Q["nearest"]),
    "quantile_midpoint": (col("x").quantile(0.3, "midpoint"), _Q["midpoint"]),
    "quantile_equiprobable": (col("x").quantile(0.3, "equiprobable"), "quantile_disc(x, 0.3)"),
    "sum_default": (col("i").sum(), "sum(i)"),
    "sum_empty_value": (col("i").sum(empty_value=0), "coalesce(sum(i), 0)"),
    "product_empty_value": (col("y").product(empty_value=1), "coalesce(product(y), 1.0)"),
    "bool_and_empty_value": (
        col("flag").bool_and(empty_value=True),
        "coalesce(bool_and(flag), true)",
    ),
    "bool_or_empty_value": (
        col("flag").bool_or(empty_value=False),
        "coalesce(bool_or(flag), false)",
    ),
    "count_distinct_default": (col("x").count_distinct(), "count(DISTINCT x)"),
    "count_distinct_nulls": (
        col("x").count_distinct(count_nulls=True),
        "count(DISTINCT x) + CAST(count(*) > count(x) AS BIGINT)",
    ),
    "max_default": (col("x").max(), "max(x)"),
    "max_nan_ignore": (
        col("x").max(nan_policy="ignore"),
        "coalesce(max(x) FILTER (WHERE NOT isnan(x)), max(x))",
    ),
    "max_nan_ignore_int": (col("i").max(nan_policy="ignore"), "max(i)"),
    "max_nan_ignore_string": (col("g").max(nan_policy="ignore"), "max(g)"),
    "skew_default": (col("y").skew(), "skewness(y)"),
    "kurtosis_bias": (col("y").kurtosis(bias=True), "kurtosis_pop(y)"),
    "kurtosis_pearson": (col("y").kurtosis(bias=True, fisher=False), "kurtosis_pop(y) + 3"),
    "entropy_default": (col("i").entropy(), "entropy(i)"),
    "entropy_base_e": (col("i").entropy(math.e), f"entropy(i) * {math.log(2.0)!r}"),
    "entropy_values": (
        col("y").entropy(base=2.0, of="values"),
        "coalesce((ln(sum(y)) - sum(y * ln(y)) / sum(y)) / ln(2), 0)",
    ),
    # Rounded: over 40,000 rows the two sums reassociate differently in the tenth digit.
    "entropy_values_unnormalized": (
        col("y").entropy(math.e, of="values", normalize=False).round(4),
        "round(coalesce(-sum(y * ln(y)), 0), 4)",
    ),
    "first_keeps_null": (col("i").first("k", ignore_nulls=False), "arg_min_null(i, k)"),
    "last_keeps_null": (col("i").last("k", ignore_nulls=False), "arg_max_null(i, k)"),
    "arg_min_keeps_null": (col("x").arg_min("k", ignore_nulls=False), "arg_min_null(x, k)"),
    "arg_max_default": (col("x").arg_max("k"), "arg_max(x, k)"),
    "var_ddof0": (col("y").var(ddof=0), "var_pop(y)"),
    "std_ddof0": (col("y").std(ddof=0), "stddev_pop(y)"),
    "std_ddof2": (
        col("y").std(ddof=2),
        "CASE WHEN count(y) > 2 THEN sqrt(var_pop(y) * count(y) / (count(y) - 2)) END",
    ),
}


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
@pytest.mark.parametrize("name", sorted(AGGREGATES))
def test_aggregate_parameter_matches_duckdb(duck, name, shape, path):
    agg, sql = AGGREGATES[name]
    table = INPUTS[shape]
    duck.register("t", table)
    ds = bt.from_arrow(table)
    grouped = PATHS[path](ds.group_by("g").agg(r=agg))
    assert_same(grouped, duck.sql(f"SELECT g, {sql} AS r FROM t GROUP BY g"))
    whole = PATHS[path](ds.agg(r=agg))
    assert_same(whole, duck.sql(f"SELECT {sql} AS r FROM t"))


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_skew_bias_is_the_population_skewness(duck, shape, path):
    # DuckDB has no population skewness, so it is spelled from the central moments. It is
    # null for a group with no variance, as `kurtosis_pop` is.
    table = INPUTS[shape]
    duck.register("t", table)
    ds = bt.from_arrow(table)
    moments = (
        "WITH c AS (SELECT g, y - avg(y) OVER (PARTITION BY g) AS d FROM t WHERE y IS NOT NULL) "
        "SELECT g, CASE WHEN avg(d * d) > 1e-12 THEN avg(d * d * d) / pow(avg(d * d), 1.5) END "
        "AS r FROM c GROUP BY g"
    )
    got = PATHS[path](
        ds.filter(col("y").is_not_null()).group_by("g").agg(r=col("y").skew(bias=True))
    )
    assert_same(got, duck.sql(moments))


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_modes_returns_every_tied_value(duck, shape, path):
    table = INPUTS[shape]
    duck.register("t", table)
    ds = bt.from_arrow(table)
    grouped = PATHS[path](ds.group_by("g").agg(r=col("i").mode(all_modes=True)))
    assert_same(grouped, duck.sql(_modes_sql("i", keyed=True)))
    whole = PATHS[path](ds.agg(r=col("i").mode(all_modes=True)))
    assert_same(whole, duck.sql(_modes_sql("i", keyed=False)))


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_array_agg_ignore_nulls(duck, shape, path):
    # Element order is arrival order, so both sides sort their lists before comparing.
    table = INPUTS[shape]
    duck.register("t", table)
    ds = bt.from_arrow(table)
    got = PATHS[path](
        ds.group_by("g")
        .agg(r=col("i").array_agg(ignore_nulls=True))
        .with_columns(r=col("r").list.sort())
    )
    want = duck.sql(
        "SELECT g, list_sort(list_filter(list(i), e -> e IS NOT NULL)) AS r FROM t GROUP BY g"
    )
    assert_same(got, want)


#: Window forms: (batcher window expression, DuckDB window call). All are ordered by the
#: unique key ``k``, so the frame -- not a tie -- decides the answer.
WINDOWS: dict[str, tuple] = {
    # Bugfix: `last_value`/`nth_value` over an ORDER BY read the whole partition, where
    # DuckDB, Spark and Batcher's own SQL use the running default frame.
    "last_value_default_frame": (bt.last_value("i"), "last_value(i)"),
    "nth_value_default_frame": (bt.nth_value("i", 3), "nth_value(i, 3)"),
    "last_value_whole_partition": (
        bt.last_value("i"),
        "last_value(i) OVER (PARTITION BY g ORDER BY k ROWS BETWEEN UNBOUNDED PRECEDING AND "
        "UNBOUNDED FOLLOWING)",
    ),
    "first_value_ignore_nulls": (
        bt.first_value("i", ignore_nulls=True),
        "first_value(i IGNORE NULLS)",
    ),
    "last_value_ignore_nulls": (
        bt.last_value("i", ignore_nulls=True),
        "last_value(i IGNORE NULLS)",
    ),
    "nth_value_ignore_nulls": (
        bt.nth_value("i", 2, ignore_nulls=True),
        "nth_value(i, 2 IGNORE NULLS)",
    ),
}


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
@pytest.mark.parametrize("name", sorted(WINDOWS))
def test_value_window_matches_duckdb(duck, name, shape, path):
    fn, sql = WINDOWS[name]
    table = INPUTS[shape]
    duck.register("t", table)
    frame = (None, None) if name.endswith("whole_partition") else None
    w = fn.over(partition_by="g", order_by="k", frame=frame)
    got = PATHS[path](bt.from_arrow(table).select("k", r=w))
    over = sql if " OVER " in sql else f"{sql} OVER (PARTITION BY g ORDER BY k)"
    assert_same(got, duck.sql(f"SELECT k, {over} AS r FROM t"))


def _peak_sql(op: str, edges: bool) -> str:
    """DuckDB spelling of Polars' three-valued peak with an explicit edge rule."""
    edge = "true" if edges else "false"
    w = "OVER (PARTITION BY g ORDER BY k)"
    side = "CASE WHEN {lag}(i IS NULL) {w} IS NULL THEN {edge} ELSE i {op} {lag}(i) {w} END"
    prev = side.format(lag="lag", w=w, edge=edge, op=op)
    nxt = side.format(lag="lead", w=w, edge=edge, op=op)
    return f"SELECT k, CASE WHEN i IS NULL THEN NULL ELSE ({prev}) AND ({nxt}) END AS r FROM t"


@pytest.mark.parametrize("path", sorted(PATHS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
@pytest.mark.parametrize(
    ("method", "op", "edges"), [("peak_max", ">", True), ("peak_min", "<", False)]
)
def test_peak_polars_policy_matches_three_valued_sql(duck, method, op, edges, shape, path):
    table = INPUTS[shape]
    duck.register("t", table)
    peak = getattr(col("i"), method)(
        partition_by=["g"], order_by=["k"], edges=edges, propagate_nulls=True
    )
    got = PATHS[path](bt.from_arrow(table).select("k", r=peak))
    assert_same(got, duck.sql(_peak_sql(op, edges)))


def test_groupby_shortcuts_carry_the_parameters(duck):
    table = INPUTS["base"]
    duck.register("t", table)
    ds = bt.from_arrow(table)
    assert_same(
        ds.group_by("g").sum("i", empty_value=0).collect(),
        duck.sql("SELECT g, coalesce(sum(i), 0) AS i FROM t GROUP BY g"),
    )
    assert_same(
        ds.group_by("g").max("x", nan_policy="ignore").collect(),
        duck.sql(
            "SELECT g, coalesce(max(x) FILTER (WHERE NOT isnan(x)), max(x)) AS x FROM t GROUP BY g"
        ),
    )
    assert_same(
        ds.group_by("g").count_distinct("x", count_nulls=True).collect(),
        duck.sql(
            "SELECT g, count(DISTINCT x) + CAST(count(*) > count(x) AS BIGINT) AS x "
            "FROM t GROUP BY g"
        ),
    )
    assert_same(
        ds.group_by("g").quantile(0.3, "x", interpolation="nearest").collect(),
        duck.sql(f"SELECT g, {_Q['nearest']} AS x FROM t GROUP BY g"),
    )
    assert_same(
        ds.group_by("g").first("i", order_by="k", ignore_nulls=False).collect(),
        duck.sql("SELECT g, arg_min_null(i, k) AS i FROM t GROUP BY g"),
    )
    assert_same(
        ds.group_by("g").kurtosis("y", bias=True).collect(),
        duck.sql("SELECT g, kurtosis_pop(y) AS y FROM t GROUP BY g"),
    )
    assert_same(
        ds.group_by("g").std("y", ddof=0).collect(),
        duck.sql("SELECT g, stddev_pop(y) AS y FROM t GROUP BY g"),
    )


def test_positional_composite_is_named_after_its_column(duck):
    table = INPUTS["base"]
    duck.register("t", table)
    got = bt.from_arrow(table).group_by("g").agg(col("i").sum(empty_value=0)).collect()
    assert got.column_names == ["g", "i"]
    assert_same(got, duck.sql("SELECT g, coalesce(sum(i), 0) AS i FROM t GROUP BY g"))


def test_default_forms_serialize_as_before():
    # A parameter at its default must build the plain aggregate, byte for byte.
    assert col("x").quantile(0.5, "linear").to_ir("q") == {
        "func": "quantile",
        "alias": "q",
        "input": {"e": "col", "name": "x"},
        "param": 0.5,
    }
    assert col("x").quantile(0.5, "nearest").to_ir("q")["interpolation"] == "nearest"
    for built, func in [
        (col("x").sum(empty_value=None), "sum"),
        (col("x").max(nan_policy="propagate"), "max"),
        (col("x").skew(bias=False), "skewness"),
        (col("x").mode(all_modes=False), "mode"),
        (col("x").arg_max("k", ignore_nulls=True), "arg_max"),
        (col("x").std(ddof=1), "stddev"),
        (col("x").entropy(2.0), "entropy"),
    ]:
        assert isinstance(built, bt.AggExpr)
        assert built.func == func


def test_parameter_validation_is_loud():
    from batcher._internal.errors import PlanError
    from batcher.plan.logical import WindowFuncSpec

    with pytest.raises(PlanError, match="interpolation"):
        col("x").quantile(0.5, "banker")
    with pytest.raises(PlanError, match="nan_policy"):
        col("x").max(nan_policy="drop")
    with pytest.raises(PlanError, match="base"):
        col("x").entropy(1.0)
    with pytest.raises(PlanError, match="of="):
        col("x").entropy(of="counts")
    with pytest.raises(PlanError, match="ignore_nulls"):
        WindowFuncSpec("lag", col("x"), "r", ignore_nulls=True)
