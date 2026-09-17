"""Each aggregate parameter against the engine whose meaning it restores: Polars, Daft, Spark.

The DuckDB side of the same parameters is `test_diff_agg_semantic_params.py`; this file is
the other half of the claim, that the non-default value really is the competitor's answer.
Polars 1.40 and Daft 0.7.25 run for real. There is no JVM here, so the Spark cases assert
the documented examples from the PySpark sources, each cited by function in
``python/pyspark/sql/functions/builtin.py`` (Spark 4.x).

Ray Data is not exercised: starting a local Ray Data pipeline cost over two minutes on this
box, so its rows are settled by codemod templates rather than by a run.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pl = pytest.importorskip("polars")

pytestmark = pytest.mark.differential

NAN = float("nan")

#: One column per concern, over three groups: ``a`` has NaN, duplicates and ties, ``b`` is
#: all null, ``c`` is a single row. ``t`` is a unique order key.
DATA = {
    "g": ["a", "a", "a", "a", "a", "a", "b", "b", "c"],
    "t": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "x": [3.0, NAN, 1.0, 3.0, -0.0, 2.0, None, None, 5.0],
    "i": [4, None, 2, 4, 2, 7, None, None, 9],
    "p": [0.5, 1.0, 2.0, 0.25, 0.5, 3.0, None, None, 1.0],
    "flag": [True, None, False, True, True, None, None, None, False],
}
SCHEMA = {
    "g": pa.string(),
    "t": pa.int64(),
    "x": pa.float64(),
    "i": pa.int64(),
    "p": pa.float64(),
    "flag": pa.bool_(),
}


def _bt() -> bt.Dataset:
    return bt.from_arrow(pa.table({k: pa.array(v, SCHEMA[k]) for k, v in DATA.items()}))


def _pl():
    return pl.DataFrame(
        DATA,
        schema={
            "g": pl.String,
            "t": pl.Int64,
            "x": pl.Float64,
            "i": pl.Int64,
            "p": pl.Float64,
            "flag": pl.Boolean,
        },
    )


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _grouped(ours: bt.Dataset, theirs) -> None:
    got = ours.sort("g").to_pydict()
    want = theirs.sort("g").to_dict(as_series=False)
    assert got.keys() == want.keys()
    for name in got:
        assert len(got[name]) == len(want[name]), name
        for a, b in zip(got[name], want[name], strict=True):
            assert _same(a, b), f"{name}: batcher {got[name]} vs polars {want[name]}"


@pytest.mark.parametrize(
    "interpolation", ["nearest", "lower", "higher", "midpoint", "linear", "equiprobable"]
)
def test_polars_quantile_interpolation(interpolation):
    # Every position class: on an element, below, on and above the half, and both ends.
    for n in (1, 2, 3, 4, 5, 6, 7, 10):
        values = [float((v * 7) % 11) for v in range(n)]
        for q in (0.0, 0.1, 1 / 6, 0.25, 0.3, 0.5, 0.625, 0.75, 0.9, 1.0):
            ours = bt.from_pydict({"v": values}).agg(r=col("v").quantile(q, interpolation))
            theirs = pl.DataFrame({"v": values}).select(
                pl.col("v").quantile(q, interpolation=interpolation)
            )
            assert _same(ours.to_pydict()["r"][0], theirs.item()), (n, q)


def test_polars_group_by_defaults():
    ours = (
        _bt()
        .group_by("g")
        .agg(
            col("x").quantile(0.3, "nearest"),
            col("i").sum(empty_value=0),
            col("p").product(empty_value=1),
            col("x").count_distinct(count_nulls=True).alias("nu"),
            col("x").max(nan_policy="ignore").alias("mx"),
            col("flag").bool_and(empty_value=True).alias("all"),
            col("flag").bool_or(empty_value=False).alias("any"),
        )
    )
    theirs = (
        _pl()
        .group_by("g")
        .agg(
            pl.col("x").quantile(0.3),
            pl.col("i").sum(),
            pl.col("p").product(),
            pl.col("x").n_unique().alias("nu"),
            pl.col("x").max().alias("mx"),
            pl.col("flag").all().alias("all"),
            pl.col("flag").any().alias("any"),
        )
    )
    _grouped(ours, theirs)


def test_polars_group_by_shortcuts():
    ds = _bt().select("g", "x", "i")
    frame = _pl().select("g", "x", "i")
    _grouped(ds.group_by("g").sum(empty_value=0), frame.group_by("g").sum())
    _grouped(ds.group_by("g").max(nan_policy="ignore"), frame.group_by("g").max())
    _grouped(ds.group_by("g").count_distinct(count_nulls=True), frame.group_by("g").n_unique())
    _grouped(
        ds.group_by("g").quantile(0.6, interpolation="nearest"),
        frame.group_by("g").quantile(0.6),
    )


def test_polars_approx_n_unique_counts_the_null():
    ours = _bt().agg(r=col("i").approx_count_distinct(count_nulls=True)).to_pydict()["r"][0]
    assert ours == _pl().select(pl.col("i").approx_n_unique()).item()


def test_polars_moments_default_to_the_biased_estimates():
    ours = _bt().filter(col("g") == "a").select("p")
    frame = _pl().filter(pl.col("g") == "a").select("p")
    got = ours.agg(
        s=col("p").skew(bias=True),
        s1=col("p").skew(),
        k=col("p").kurtosis(bias=True),
        k1=col("p").kurtosis(),
        kp=col("p").kurtosis(bias=True, fisher=False),
    ).to_pydict()
    want = frame.select(
        s=pl.col("p").skew(),
        s1=pl.col("p").skew(bias=False),
        k=pl.col("p").kurtosis(),
        k1=pl.col("p").kurtosis(bias=False),
        kp=pl.col("p").kurtosis(fisher=False),
    ).to_dict(as_series=False)
    for name in got:
        assert _same(got[name][0], want[name][0]), name


@pytest.mark.parametrize(
    "values", [[1.0, 2.0, 3.0], [0.5, 0.25, 0.25], [None, 1.0, 1.0], [0.0, 1.0], [-1.0, 2.0]]
)
@pytest.mark.parametrize(("base", "normalize"), [(math.e, True), (2.0, True), (10.0, False)])
def test_polars_entropy_reads_the_values_as_probabilities(values, base, normalize):
    ours = bt.from_pydict({"p": values}).with_columns(p=col("p").cast("float64"))
    got = ours.agg(r=col("p").entropy(base, of="values", normalize=normalize)).to_pydict()["r"][0]
    want = (
        pl.DataFrame({"p": values}, schema={"p": pl.Float64})
        .select(pl.col("p").entropy(base=base, normalize=normalize))
        .item()
    )
    assert _same(got, want)


def test_polars_mode_returns_every_tied_value():
    got = _bt().group_by("g").agg(r=col("i").mode(all_modes=True)).sort("g").to_pydict()
    want = _pl().drop_nulls("i").group_by("g").agg(r=pl.col("i").mode().sort()).sort("g")
    # Polars keeps a null as a candidate mode and drops an all-null group from this frame;
    # Batcher does not count nulls (DuckDB) and answers null for that group.
    assert got["r"] == [[2, 4], None, [9]]
    assert want.to_dict(as_series=False)["r"] == [[2, 4], [9]]


def test_polars_first_last_and_max_by_keep_a_null():
    # In group `a` the first row, the last row and the max/min-key rows all hold a null
    # value; group `b` has none. A unique order key, so no tie decides anything.
    data = {
        "g": ["a", "a", "a", "a", "b", "b"],
        "t": [0, 1, 2, 3, 4, 5],
        "v": [None, 2, 3, None, 5, 6],
        "key": [1.0, 9.0, 5.0, 0.5, 2.0, 1.0],
    }
    frame = bt.from_pydict(data).with_columns(v=col("v").cast("int64"))
    got = (
        frame.group_by("g")
        .agg(
            f=col("v").first("t", ignore_nulls=False),
            l=col("v").last("t", ignore_nulls=False),
            mx=col("v").arg_max("key", ignore_nulls=False),
            mn=col("v").arg_min("key", ignore_nulls=False),
            f_skip=col("v").first("t"),
        )
        .sort("g")
        .to_pydict()
    )
    want = (
        pl.DataFrame(data, schema={"g": pl.String, "t": pl.Int64, "v": pl.Int64, "key": pl.Float64})
        .sort("t")
        .group_by("g")
        .agg(
            f=pl.col("v").first(),
            l=pl.col("v").last(),
            mx=pl.col("v").max_by("key"),
            mn=pl.col("v").min_by("key"),
            f_skip=pl.col("v").drop_nulls().first(),
        )
        .sort("g")
        .to_dict(as_series=False)
    )
    assert got == want
    assert got["f"] == [None, 5]


@pytest.mark.parametrize(
    "values",
    [[1, 2, 1], [2, 1, 2], [5], [1, None, 1], [None, 1, None], [2, None, 1, 0, 1], [3, 2, None]],
)
def test_polars_peaks(values):
    ours = bt.from_pydict({"t": list(range(len(values))), "x": values}).with_columns(
        x=col("x").cast("int64")
    )
    got = (
        ours.with_columns(
            mx=col("x").peak_max(order_by=["t"], edges=True, propagate_nulls=True),
            mn=col("x").peak_min(order_by=["t"], propagate_nulls=True),
        )
        .sort("t")
        .select("mx", "mn")
        .to_pydict()
    )
    want = (
        pl.DataFrame({"x": values}, schema={"x": pl.Int64})
        .select(mx=pl.col("x").peak_max(), mn=pl.col("x").peak_min())
        .to_dict(as_series=False)
    )
    assert got == want


@pytest.mark.parametrize("ddof", [0, 1, 2])
def test_polars_std_ddof(ddof):
    got = _bt().group_by("g").agg(s=col("p").std(ddof=ddof), v=col("p").var(ddof=ddof))
    want = _pl().group_by("g").agg(s=pl.col("p").std(ddof=ddof), v=pl.col("p").var(ddof=ddof))
    _grouped(got, want)


def test_daft_skew_is_the_population_skewness():
    daft = pytest.importorskip("daft")
    values = [1.0, 2.0, 10.0, 4.0]
    got = bt.from_pydict({"x": values}).agg(r=col("x").skew(bias=True)).to_pydict()["r"][0]
    want = daft.from_pydict({"x": values}).agg(daft.col("x").skew()).to_pydict()["x"][0]
    assert math.isclose(got, want, rel_tol=1e-12)
    grouped = bt.from_pydict({"g": ["a"] * 4, "x": values}).group_by("g").skew(bias=True)
    assert math.isclose(grouped.to_pydict()["x"][0], want, rel_tol=1e-12)


# --- Spark: documented examples ------------------------------------------------------------


def test_spark_skewness_and_kurtosis_examples():
    # builtin.py::skewness -> 0.7071067811865..., builtin.py::kurtosis -> -1.5, on [1, 1, 2].
    got = bt.from_pydict({"c": [1, 1, 2]}).agg(
        s=col("c").skew(bias=True), k=col("c").kurtosis(bias=True)
    )
    s, k = got.to_pydict()["s"][0], got.to_pydict()["k"][0]
    assert str(s).startswith("0.7071067811865")
    assert k == pytest.approx(-1.5)


def test_spark_max_by_and_min_by_examples():
    # builtin.py::max_by / min_by, Examples 1-3.
    courses = bt.from_pydict(
        {
            "course": ["Java", "dotNET", "dotNET", "Java"],
            "year": [2012, 2012, 2013, 2013],
            "earnings": [20000, 5000, 48000, 30000],
        }
    )
    got = courses.group_by("course").agg(
        mx=col("year").arg_max("earnings", ignore_nulls=False),
        mn=col("year").arg_min("earnings", ignore_nulls=False),
    )
    assert got.sort("course").to_pydict() == {
        "course": ["Java", "dotNET"],
        "mx": [2013, 2013],
        "mn": [2012, 2012],
    }
    depts = bt.from_pydict(
        {
            "department": ["Consult", "Finance", "Finance", "Consult"],
            "name": ["Eva", "Frank", "George", "Henry"],
            "years_in_dept": [6, 5, 9, 7],
        }
    )
    got = depts.group_by("department").agg(
        mx=col("name").arg_max("years_in_dept", ignore_nulls=False),
        mn=col("name").arg_min("years_in_dept", ignore_nulls=False),
    )
    assert got.sort("department").to_pydict() == {
        "department": ["Consult", "Finance"],
        "mx": ["Henry", "George"],
        "mn": ["Eva", "Frank"],
    }


def test_spark_first_and_last_examples():
    # builtin.py::first / last: rows (Alice, 2), (Bob, 5), (Alice, NULL) ordered by age, so
    # the null sorts first ascending and last descending.
    people = bt.from_pydict(
        {
            "name": ["Alice", "Bob", "Alice"],
            "age": [2, 5, None],
            "asc": [1, 2, 0],  # orderBy(age): the null first
            "desc": [1, 0, 2],  # orderBy(age.desc()): the null last
        }
    )
    first = people.group_by("name").agg(
        keep=col("age").first("asc", ignore_nulls=False), skip=col("age").first("asc")
    )
    assert first.sort("name").to_pydict() == {
        "name": ["Alice", "Bob"],
        "keep": [None, 5],
        "skip": [2, 5],
    }
    last = people.group_by("name").agg(
        keep=col("age").last("desc", ignore_nulls=False), skip=col("age").last("desc")
    )
    assert last.sort("name").to_pydict() == {
        "name": ["Alice", "Bob"],
        "keep": [None, 5],
        "skip": [2, 5],
    }


def test_spark_collect_list_and_array_agg_skip_nulls():
    # builtin.py::array_agg, third example: [[1], [None], [2]] -> sorted [1, 2].
    got = bt.from_pydict({"c": [1, None, 2]}).agg(
        r=col("c").array_agg(ignore_nulls=True), kept=col("c").array_agg()
    )
    out = got.with_columns(r=col("r").list.sort()).to_pydict()
    assert out["r"] == [[1, 2]]
    assert sorted(out["kept"][0], key=lambda v: (v is None, v)) == [1, 2, None]
    fn = bt.from_pydict({"c": [1, None, 2]}).agg(r=bt.array_agg("c", ignore_nulls=True))
    assert sorted(fn.to_pydict()["r"][0]) == [1, 2]


def test_spark_nth_value_example():
    # builtin.py::nth_value: Window.partitionBy("c1").orderBy("c2").
    df = bt.from_pydict({"c1": ["a", "a", "a", "b", "b"], "c2": [1, 2, 3, 8, 2]})
    w1 = bt.nth_value("c2", 1).over(partition_by="c1", order_by="c2")
    w2 = bt.nth_value("c2", 2).over(partition_by="c1", order_by="c2")
    got = df.with_columns(n1=w1, n2=w2).sort("c1", "c2").to_pydict()
    assert got["n1"] == [1, 1, 1, 2, 2]
    assert got["n2"] == [None, 2, 2, None, 8]


def test_spark_first_value_and_last_value_examples():
    # builtin.py::first_value / last_value, over the rows in the order they were created.
    first = bt.from_pydict({"a": [None, "a", "a", "b", "b"], "b": [1, 2, 3, 8, 2], "t": range(5)})
    whole = (None, None)
    got = first.select(
        fa=bt.first_value("a").over(order_by="t", frame=whole),
        fb=bt.first_value("b").over(order_by="t", frame=whole),
        ia=bt.first_value("a", ignore_nulls=True).over(order_by="t", frame=whole),
    ).to_pydict()
    assert (got["fa"][0], got["fb"][0], got["ia"][0]) == (None, 1, "a")
    last = bt.from_pydict({"a": ["a", "a", "a", "b", None], "b": [1, 2, 3, 8, 2], "t": range(5)})
    got = last.select(
        la=bt.last_value("a").over(order_by="t", frame=whole),
        lb=bt.last_value("b").over(order_by="t", frame=whole),
        ia=bt.last_value("a", ignore_nulls=True).over(order_by="t", frame=whole),
    ).to_pydict()
    assert (got["la"][0], got["lb"][0], got["ia"][0]) == (None, 2, "b")
