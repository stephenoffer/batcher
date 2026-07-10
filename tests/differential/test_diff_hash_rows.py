"""`hash_rows` — a typed, deterministic row digest, checked structurally against DuckDB.

A hash's *value* is Batcher's own contract, not DuckDB's, so there is nothing to compare
digit-for-digit (the golden values are pinned in the Rust unit tests instead). What *is*
oracle-able is the relational behaviour a hash is used for, and that is what a wrong
implementation breaks:

* **grouping by the hash must group exactly as grouping by the values does** — the
  property that makes it a legal bucketing key. A hash that collided, or that treated
  ``-0.0`` and ``0.0`` differently, would split or merge groups here.
* **rows that compare equal hash equally, rows that differ do not** (absent collision),
  and null is a *positional value*, so ``(1, NULL)`` and ``(NULL, 1)`` are distinct.

The float cases are the ones the previous `concat_ws(cast(x, 'string'))` idiom got
wrong-ish: it hashed the *rendering* of a float, so `-0.0` and `0.0` — equal to every
comparison in the engine — landed in different buckets.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import col, hash_rows

pytestmark = pytest.mark.differential


def _hash(table: pa.Table, *columns: str, seed: int = 0) -> list[int]:
    ds = bt.from_arrow(table)
    return ds.select(h=hash_rows(*(col(c) for c in columns), seed=seed)).to_pydict()["h"]


def test_grouping_by_the_hash_matches_grouping_by_the_values(duck):
    """The defining property of a bucketing key: same partition of the rows."""
    t = pa.table(
        {
            "a": pa.array([1, 1, 2, 2, 3, None, None], type=pa.int64()),
            "b": pa.array(["x", "x", "y", "y", "x", "x", None], type=pa.string()),
        }
    )
    duck.register("t", t)
    ds = bt.from_arrow(t)

    by_hash = (
        ds.with_columns(h=hash_rows(col("a"), col("b")))
        .group_by("h")
        .agg(n=bt.count())
        .to_pydict()["n"]
    )
    by_values = duck.sql("SELECT count(*) AS n FROM t GROUP BY a, b").to_arrow_table()
    assert sorted(by_hash) == sorted(by_values.column("n").to_pylist())


def test_the_hash_has_as_many_distinct_values_as_the_key(duck):
    t = pa.table({"a": pa.array(list(range(500)) * 2, type=pa.int64())})
    duck.register("t", t)
    ds = bt.from_arrow(t)
    distinct_hashes = ds.select(h=hash_rows(col("a"))).distinct().count()
    distinct_keys = duck.sql("SELECT count(DISTINCT a) FROM t").fetchone()[0]
    assert distinct_hashes == distinct_keys


def test_equal_rows_hash_equally_and_different_rows_do_not():
    t = pa.table({"a": pa.array([1, 1, 2], type=pa.int64())})
    got = _hash(t, "a")
    assert got[0] == got[1]
    assert got[0] != got[2]


def test_the_seed_changes_the_digest():
    t = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
    assert _hash(t, "a", seed=0) != _hash(t, "a", seed=1)


def test_column_order_is_part_of_the_digest():
    t = pa.table({"a": pa.array([1], type=pa.int64()), "b": pa.array([2], type=pa.int64())})
    assert _hash(t, "a", "b") != _hash(t, "b", "a")


def test_nulls_are_positional_values():
    """`(1, NULL)` and `(NULL, 1)` must differ, and neither may collide with `(1, 1)`."""
    t = pa.table(
        {
            "a": pa.array([1, None, 1], type=pa.int64()),
            "b": pa.array([None, 1, 1], type=pa.int64()),
        }
    )
    got = _hash(t, "a", "b")
    assert len(set(got)) == 3


def test_negative_zero_hashes_like_zero():
    """They compare equal in the engine, so they must bucket together."""
    t = pa.table({"x": pa.array([0.0, -0.0], type=pa.float64())})
    got = _hash(t, "x")
    assert got[0] == got[1]


def test_all_nans_hash_alike():
    t = pa.table({"x": pa.array([math.nan, -math.nan], type=pa.float64())})
    got = _hash(t, "x")
    assert got[0] == got[1]


def test_the_digest_does_not_depend_on_how_a_float_renders():
    """`1.0` and `1` are different *values* (float vs int) but a float column's digest
    must come from its bits, not from `'1.0'` vs `'1'`."""
    a = _hash(pa.table({"x": pa.array([1.5], type=pa.float64())}), "x")
    b = _hash(pa.table({"x": pa.array([1.5], type=pa.float32())}), "x")
    assert a == b, "widening a float32 to float64 must not change its digest"


@pytest.mark.parametrize(
    "column",
    [
        pa.array([1, 2, 3], type=pa.int64()),
        pa.array([1, 2, 3], type=pa.int32()),
        pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        pa.array(["a", "b", "c"], type=pa.string()),
        pa.array([True, False, True], type=pa.bool_()),
    ],
)
def test_every_common_type_hashes(column):
    got = _hash(pa.table({"x": column}), "x")
    assert len(got) == 3
    assert all(isinstance(v, int) for v in got)


def test_hashes_spread_across_buckets():
    """A structured (e.g. affine) hash would pile rows into a few buckets."""
    t = pa.table({"a": pa.array(list(range(8192)), type=pa.int64())})
    ds = bt.from_arrow(t)
    counts = (
        ds.with_columns(b=hash_rows(col("a"), seed=3).abs() % 8)
        .group_by("b")
        .agg(n=bt.count())
        .to_pydict()["n"]
    )
    assert len(counts) == 8
    assert all(850 < n < 1200 for n in counts), counts


def test_expr_hash_is_the_one_argument_spelling():
    t = pa.table({"a": pa.array([1, 2], type=pa.int64())})
    ds = bt.from_arrow(t)
    assert (
        ds.select(h=col("a").hash(seed=4)).to_pydict()["h"]
        == ds.select(h=hash_rows(col("a"), seed=4)).to_pydict()["h"]
    )


def test_hash_rows_requires_an_argument():
    with pytest.raises(PlanError, match="at least one expression"):
        hash_rows()


def test_the_hash_is_partition_independent():
    """Same rows, different batch boundaries — same digests."""
    rows = pa.table({"a": pa.array(list(range(300)), type=pa.int64())})
    one = _hash(rows, "a")
    many = bt.from_arrow(rows.to_batches(max_chunksize=17))
    assert many.select(h=hash_rows(col("a"))).to_pydict()["h"] == one


def test_projection_pushdown_keeps_the_hashed_columns():
    """The optimizer must see through `hash_rows` to the columns it reads, or it prunes
    the very columns the digest depends on."""
    from batcher.plan.expr_ir import referenced_columns

    assert referenced_columns(hash_rows(col("a"), col("b"))) == {"a", "b"}

    t = pa.table({"a": pa.array([1, 2]), "b": pa.array([3, 4]), "c": pa.array([5, 6])})
    ds = bt.from_arrow(t)
    got = ds.select(h=hash_rows(col("a"), col("b"))).to_pydict()["h"]
    assert len(set(got)) == 2


def test_list_columns_hash_by_their_elements():
    """An embedding or a packed training sequence is a list column — and has no `Utf8`
    cast, so a textual fallback would fail outright on exactly the columns an ML
    pipeline carries."""
    t = pa.table({"v": pa.array([[1, 2], [1, 2], [2, 1], [1], None], type=pa.list_(pa.int64()))})
    got = _hash(t, "v")
    assert got[0] == got[1], "equal lists hash equally"
    assert got[0] != got[2], "element order matters"
    assert got[0] != got[3], "length matters"
    assert got[3] != got[4], "a null list is not a short one"


def test_a_fixed_size_list_column_can_be_split_on():
    """`train_test_split` defaults to hashing every column; a packed sequence column
    (`FixedSizeList<Int64>[n]`) must not crash it."""
    values = pa.array(list(range(8)), type=pa.int64())
    t = pa.table({"tokens": pa.FixedSizeListArray.from_arrays(values, 4)})
    ds = bt.from_arrow(t)
    train, test = ds.ml.train_test_split(0.5, seed=1)
    assert train.count() + test.count() == 2
