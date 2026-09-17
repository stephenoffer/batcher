"""Each scalar and list parameter against the engine whose meaning it restores: Polars, Daft, Spark.

The DuckDB side of the same parameters is `test_diff_scalar_semantic_params.py` and
`test_diff_list_semantic_params.py`; this file is the other half of the claim, that the
non-default value really is the competitor's answer. Polars 1.40 and Daft 0.7.25 run for
real. There is no JVM here, so the Spark cases assert the documented examples in
``python/pyspark/sql/functions/builtin.py`` (Spark 4.x), each cited by function name.

A projection preserves row order, so rows are compared positionally.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


def _same(got: list, want: list) -> None:
    """Positional equality where NaN equals NaN and floats agree to 1e-12."""
    assert len(got) == len(want), (got, want)
    for g, w in zip(got, want, strict=True):
        if isinstance(g, list) and isinstance(w, list):
            _same(g, w)
        elif isinstance(g, float) and isinstance(w, float):
            assert (math.isnan(g) and math.isnan(w)) or g == pytest.approx(w, rel=1e-12), (g, w)
        else:
            assert g == w, (got, want)


# --- Polars 1.40 ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pl():
    return pytest.importorskip("polars")


def test_polars_round_defaults_to_half_to_even(pl):
    xs = [-2.5, 2.5, 0.5, -0.5, 1.5, 2.675, None]
    want = pl.Series(xs).round(0).to_list()
    want2 = pl.Series(xs).round(2, mode="half_to_even").to_list()
    got = (
        bt.from_pydict({"x": xs})
        .select(r=col("x").round(mode="half_to_even"), r2=col("x").round(2, mode="half_to_even"))
        .to_pydict()
    )
    _same(got["r"], want)
    _same(got["r2"], want2)


def test_polars_boolean_xor(pl):
    a, b = [True, False, None, True], [True, True, True, None]
    want = pl.DataFrame({"a": a, "b": b}).select(pl.col("a").xor(pl.col("b"))).to_series()
    got = bt.from_pydict({"a": a, "b": b}).select(r=col("a") ^ col("b")).to_pydict()["r"]
    assert got == want.to_list()


@pytest.mark.parametrize("method", ["cum_sum", "cum_prod", "cum_min", "cum_max"])
@pytest.mark.parametrize("reverse", [False, True])
def test_polars_running_aggregates_propagate_nulls(pl, method, reverse):
    xs = [3, None, -2, 5, None, 1]
    want = getattr(pl.Series(xs), method)(reverse=reverse).to_list()
    # Polars' implicit row order is the one `with_row_index` makes explicit; a running value
    # requires an order, so the port numbers the rows and orders by that number.
    running = getattr(col("x"), method)(reverse=reverse, propagate_nulls=True)
    got = (
        bt.from_pydict({"x": xs})
        .with_row_index("_row")
        .select("_row", r=running.over(order_by="_row"))
        .collect()
        .sort_by("_row")
        .column("r")
        .to_pylist()
    )
    assert got == want


@pytest.mark.parametrize("method", ["average", "min", "max", "dense", "ordinal"])
def test_polars_rank_methods_with_null_rows_null(pl, method):
    xs = [3.0, 1.0, 3.0, None, 2.0, 3.0]
    want = pl.Series(xs).rank(method=method).to_list()
    ds = bt.from_pydict({"i": list(range(6)), "x": xs})
    expr = col("x").rank(method, propagate_nulls=True)
    got = ds.with_columns(r=expr).sort("i").to_pydict()["r"]
    if method == "ordinal":
        # Ties are numbered arbitrarily; the multiset of ranks within a tie is what agrees.
        assert sorted(g for g in got if g is not None) == sorted(w for w in want if w is not None)
        assert [g is None for g in got] == [w is None for w in want]
    else:
        assert got == want


@pytest.mark.parametrize(
    ("values", "nulls_equal"), [([1, None], True), ([1, None], False), ([1], True), ([None], True)]
)
def test_polars_is_in_nulls_equal(pl, values, nulls_equal):
    xs = [1, 2, None]
    want = pl.Series(xs).is_in(values, nulls_equal=nulls_equal).to_list()
    kept = values if nulls_equal else [v for v in values if v is not None]
    got = (
        bt.from_pydict({"x": xs})
        .select(r=col("x").is_in(kept, nulls_equal=nulls_equal))
        .to_pydict()["r"]
    )
    # Polars' default ignores a None in the list, which is the list without it under SQL IN.
    assert got == want


def test_polars_rolling_std_poisons_only_nan_windows(pl):
    xs = [1.0, float("nan"), 3.0, 4.0, 6.0, 10.0]
    want = pl.Series(xs).rolling_std(2, min_samples=1).to_list()
    want_var = pl.Series(xs).rolling_var(3, min_samples=1, ddof=0).to_list()
    got = (
        bt.from_pydict({"i": list(range(6)), "x": xs})
        .select(
            s=col("x").rolling_std(2, order_by="i"), v=col("x").rolling_var(3, ddof=0, order_by="i")
        )
        .to_pydict()
    )
    _same(got["s"], want)
    _same(got["v"], want_var)


def test_polars_bare_strings_are_columns(pl):
    data = {"a": [None, 1.0, 2.0], "b": [2.0, None, -1.0]}
    frame = pl.DataFrame(data)
    want = frame.select(
        c=pl.coalesce("a", "b"), t=pl.arctan2("a", "b"), g=pl.max_horizontal("a", "b")
    ).to_dict(as_series=False)
    got = (
        bt.from_pydict(data)
        .select(c=bt.coalesce("a", "b"), t=bt.arctan2("a", "b"), g=bt.greatest("a", "b"))
        .to_pydict()
    )
    assert got["c"] == want["c"] and got["g"] == want["g"]
    _same(got["t"], want["t"])


def test_polars_arg_extremes_top_k_and_max_by(pl):
    data = {
        "g": [1, 1, 1, 1, 2, 2, 3],
        "x": [4, None, 9, 9, -3, 7, None],
        "t": [1, 2, 3, 0, 5, 4, 6],
    }
    frame = pl.DataFrame(data)
    want = (
        frame.group_by("g", maintain_order=True)
        .agg(
            imax=pl.col("x").arg_max(),
            imin=pl.col("x").arg_min(),
            top=pl.col("x").top_k(2).sort(descending=True),
            by=pl.col("x").max_by("t"),
        )
        .sort("g")
        .to_dict(as_series=False)
    )
    # Polars numbers a group's rows in frame order; Batcher needs that order named, so the
    # frame is numbered at the source and the positions are taken along it.
    got = (
        bt.from_pydict(data)
        .with_row_index("_row")
        .group_by("g")
        .agg(
            imax=col("x").arg_max(order_by="_row"),
            imin=col("x").arg_min(order_by="_row"),
            top=col("x").top_k(2),
            by=col("x").max_by("t"),
        )
        .sort("g")
        .to_pydict()
    )
    assert got["imax"] == want["imax"] and got["imin"] == want["imin"]
    # Polars pads a group with fewer than k non-null values with its nulls (group 3 is
    # `[None]`); Batcher's `top_k` returns the largest *values*, as DuckDB's `max(x, k)`
    # does, so the groups agree wherever k values exist.
    assert got["top"][:2] == want["top"][:2] and want["top"][2] == [None] and got["top"][2] == []
    # Polars `max_by` returns the value at the max even when it is null (group 3's only row);
    # Batcher's `max_by` skips null values, which is the aggregates wave's to reconcile.
    assert got["by"][:2] == want["by"][:2]


_LISTS = [[3, None, 1, 1], [], None, [None], [5, -5, None, 5]]


def test_polars_list_sort_unique_n_unique_sum(pl):
    s = pl.Series(_LISTS, dtype=pl.List(pl.Int64))
    want = {
        "sort": s.list.sort().to_list(),
        "sort_desc": s.list.sort(descending=True, nulls_last=True).to_list(),
        "unique": s.list.unique(maintain_order=True).to_list(),
        "n_unique": s.list.n_unique().to_list(),
        "sum": s.list.sum().to_list(),
    }
    got = (
        bt.from_arrow(pa.table({"l": pa.array(_LISTS, pa.list_(pa.int64()))}))
        .select(
            sort=col("l").list.sort(nulls_last=False),
            sort_desc=col("l").list.sort(descending=True),
            unique=col("l").list.unique(drop_nulls=False),
            n_unique=col("l").list.n_unique(count_nulls=True),
            sum=col("l").list.sum(empty_as_zero=True),
        )
        .to_pydict()
    )
    assert got == want


def test_polars_list_diff_and_concat(pl):
    rows = [[1, 3, 6], [], None, [None, 1, 2]]
    want_diff = pl.Series(rows, dtype=pl.List(pl.Int64)).list.diff().to_list()
    a, b = [[1], None, [2], None], [[2], [3], None, None]
    frame = pl.DataFrame({"a": a, "b": b}, schema={"a": pl.List(pl.Int64), "b": pl.List(pl.Int64)})
    want_concat = frame.select(pl.col("a").list.concat("b")).to_series().to_list()
    got_diff = bt.from_arrow(pa.table({"l": pa.array(rows, pa.list_(pa.int64()))})).select(
        d=col("l").list.diff()
    )
    assert got_diff.collect().schema.field("d").type == pa.list_(pa.int64())
    assert got_diff.to_pydict()["d"] == want_diff
    table = pa.table(
        {"a": pa.array(a, pa.list_(pa.int64())), "b": pa.array(b, pa.list_(pa.int64()))}
    )
    got_concat = (
        bt.from_arrow(table)
        .select(c=col("a").list.concat(col("b"), propagate_nulls=True))
        .to_pydict()["c"]
    )
    assert got_concat == want_concat


# --- Daft 0.7.25 ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def daft():
    return pytest.importorskip("daft")


def _u64_as_i64(v: int | None) -> int | None:
    return None if v is None else (v - 2**64 if v >= 2**63 else v)


@pytest.mark.parametrize("seed", [0, 7])
def test_daft_xxhash3(daft, seed):
    ints = [1, -1, 0, None, 2**62]
    strs = ["ABC", "", "a longer string past the sixteen-byte xxh3 bucket", None, "é"]
    floats = [1.5, -0.0, 0.0, float("nan"), None]
    frame = daft.from_pydict({"i": ints, "s": strs, "f": floats})
    want = frame.select(
        daft.col("i").hash(seed=seed),
        daft.col("s").hash(seed=seed).alias("s"),
        daft.col("f").hash(seed=seed).alias("f"),
    ).to_pydict()
    got = (
        bt.from_pydict({"i": ints, "s": strs, "f": floats})
        .select(
            i=col("i").hash(seed, algorithm="xxhash3"),
            s=col("s").hash(seed, algorithm="xxhash3"),
            f=col("f").hash(seed, algorithm="xxhash3"),
        )
        .to_pydict()
    )
    for name in ("i", "s", "f"):
        assert got[name] == [_u64_as_i64(v) for v in want[name]], name


def test_daft_bin_is_twos_complement(daft):
    xs = [-1, 5, -8, 0, 2**62, None]
    want = daft.from_pydict({"x": xs}).select(daft.functions.bin(daft.col("x"))).to_pydict()["x"]
    got = (
        bt.from_pydict({"x": xs})
        .select(b=col("x").to_base(2, twos_complement=True))
        .to_pydict()["b"]
    )
    assert got == want


def test_daft_list_join_null_field(daft):
    rows = [["a", None, "b"], [], None, [None]]
    want = daft.from_pydict({"l": rows}).select(daft.col("l").list_join("-")).to_pydict()["l"]
    got = (
        bt.from_pydict({"l": rows})
        .select(j=col("l").list.join("-", null_replacement=""))
        .to_pydict()["j"]
    )
    assert got == want


def test_daft_jaccard_similarity_over_embeddings(daft):
    a = [[1.0, 2.0, 3.0], [1.0, float("nan"), -0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    b = [[3.0, 1.0, 5.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 2.0, 7.0]]
    embed = pa.list_(pa.float64(), 3)
    table = pa.table({"a": pa.array(a, embed), "b": pa.array(b, embed)})
    want = (
        daft.from_arrow(table)
        .select(daft.col("a").jaccard_similarity(daft.col("b")))
        .to_pydict()["a"]
    )
    got = (
        bt.from_arrow(table)
        .select(j=col("a").list.jaccard(col("b"), mode="nonzero"))
        .to_pydict()["j"]
    )
    _same(got, want)


# --- Spark (documented examples) -------------------------------------------------------------


def test_spark_bround_documented_examples():
    """`builtin.py::bround`: `bround(2.5)` is 2.0 and `bround(2.1267, 2)` is 2.13."""
    got = (
        bt.from_pydict({"x": [2.5], "y": [2.1267]})
        .select(a=col("x").round(mode="half_to_even"), b=col("y").round(2, mode="half_to_even"))
        .to_pydict()
    )
    assert got == {"a": [2.0], "b": [2.13]}


def test_spark_sort_array_documented_examples():
    """`builtin.py::sort_array`: `[2, 1, None, 3]` ascending is `[NULL, 1, 2, 3]` and
    descending is `[3, 2, 1, NULL]`; `[None, None, None]` stays three nulls."""
    table = pa.table(
        {"d": pa.array([[2, 1, None, 3], [None, None, None], []], pa.list_(pa.int64()))}
    )
    got = (
        bt.from_arrow(table)
        .select(asc=col("d").list.sort(nulls_last=False), desc=col("d").list.sort(descending=True))
        .to_pydict()
    )
    assert got["asc"] == [[None, 1, 2, 3], [None, None, None], []]
    assert got["desc"][0] == [3, 2, 1, None]


def test_spark_array_contains_position_flatten_documented_examples():
    """`builtin.py::array_contains`: a null array is NULL, `["a", None, "c"]` contains "a".
    `::array_position`: absent is 0, an empty array is 0, `[None, "b", "a"]` finds "a" at 3.
    `::flatten`: `[None, [4, 5]]` is NULL."""
    words = pa.table(
        {"d": pa.array([["c", "b", "a"], [], None, [None, "b", "a"], ["a", None, "c"]])}
    )
    got = (
        bt.from_arrow(words)
        .select(
            has_a=col("d").list.contains("a", propagate_nulls=True),
            pos_a=col("d").list.position("a", zero_if_absent=True),
            pos_d=col("d").list.position("d", zero_if_absent=True),
        )
        .to_pydict()
    )
    assert got["has_a"] == [True, False, None, True, True]
    assert got["pos_a"] == [3, 0, None, 3, 1]
    assert got["pos_d"] == [0, 0, None, 0, 0]
    nested = pa.table({"d": pa.array([[None, [4, 5]], [[1, 2, 3], [4, 5], [6]]])})
    flat = bt.from_arrow(nested).select(f=col("d").list.flatten(propagate_nulls=True))
    assert flat.to_pydict()["f"] == [None, [1, 2, 3, 4, 5, 6]]


def test_spark_bin_documented_example():
    """`builtin.py::bin`: `bin(0..9)` is `0, 1, 10, 11, 100, 101, 110, 111, 1000, 1001`."""
    got = bt.range(10).select(b=col("value").to_base(2, twos_complement=True)).to_pydict()["b"]
    assert got == ["0", "1", "10", "11", "100", "101", "110", "111", "1000", "1001"]
