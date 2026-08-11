"""The common-supertype lattice matches DuckDB, and the advertised schema matches it too.

Every operation that has to hold two differently-typed values in one column — a set
operation's branches, ``coalesce``, ``greatest``/``least``, a comparison — asks the same
question: what single type holds both? Batcher answers it in `bc_expr::common_supertype`,
mirrored by `batcher.plan.types.promote` so `Dataset.schema` predicts what the engine
produces.

These cases pin that answer against DuckDB for the type pairs a real ingest actually
produces and that the lattice used to decline: an all-null column, two decimals of
differing scale, two timestamps of differing resolution, a date against a timestamp, a
boolean against an integer. Each raised in the engine while the control plane had already
advertised a type for the result, so every case here asserts **both** the values and the
advertised type — an engine that agrees with DuckDB while `Dataset.schema` says something
else is still a bug.
"""

from __future__ import annotations

from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col


def _register(duck, name: str, table: pa.Table) -> bt.Dataset:
    """Register `table` with the oracle under `name` and return it as a `Dataset`."""
    duck.register(name, table)
    return bt.from_arrow(table)


@pytest.fixture
def nulls_and_ints(duck):
    """An all-null column beside a typed one — what an empty day partition reads as."""
    left = pa.table({"k": pa.array([1, 2], pa.int64()), "v": pa.array([None, None], pa.null())})
    right = pa.table({"k": pa.array([3, 4], pa.int64()), "v": pa.array([7, 8], pa.int64())})
    return _register(duck, "l", left), _register(duck, "r", right)


def test_union_of_a_null_column_with_an_int_column(duck, nulls_and_ints):
    left, right = nulls_and_ints
    ds = left.union(right)
    assert ds.schema.field("v").type == pa.int64()
    assert_same(ds.collect(), duck.sql("SELECT * FROM l UNION ALL SELECT * FROM r"))


def test_coalesce_over_a_null_column_falls_through_to_the_typed_one(duck, nulls_and_ints):
    left, _ = nulls_and_ints
    ds = left.select(bt.coalesce(col("v"), col("k")).alias("c"))
    assert ds.schema.field("c").type == pa.int64()
    assert_same(ds.collect(), duck.sql("SELECT COALESCE(v, k) AS c FROM l"))


def test_comparing_a_null_column_against_a_typed_one_yields_null(duck, nulls_and_ints):
    left, _ = nulls_and_ints
    ds = left.select((col("v") == col("k")).alias("eq"))
    assert_same(ds.collect(), duck.sql("SELECT v = k AS eq FROM l"))


def test_greatest_over_a_null_column_ignores_it(duck, nulls_and_ints):
    left, _ = nulls_and_ints
    ds = left.select(bt.greatest(col("v"), col("k")).alias("g"))
    assert_same(ds.collect(), duck.sql("SELECT GREATEST(v, k) AS g FROM l"))


@pytest.fixture
def decimals(duck):
    """Two money columns whose scale drifted between one write and the next."""
    coarse = pa.table({"amt": pa.array([Decimal("1.50"), Decimal("2.25")], pa.decimal128(10, 2))})
    fine = pa.table({"amt": pa.array([Decimal("3.1416"), Decimal("1.5000")], pa.decimal128(12, 4))})
    return _register(duck, "coarse", coarse), _register(duck, "fine", fine)


def test_union_of_decimals_with_differing_scale_keeps_the_finer_scale(duck, decimals):
    coarse, fine = decimals
    ds = coarse.union(fine)
    assert ds.schema.field("amt").type == pa.decimal128(12, 4)
    assert_same(ds.collect(), duck.sql("SELECT * FROM coarse UNION ALL SELECT * FROM fine"))


@pytest.fixture
def side_by_side_decimals(duck):
    """The same money values at two scales, in one relation, so a comparison pairs them.

    `1.50` and `1.5000` are the same number written two ways. Comparing them used to
    raise `Invalid comparison operation` because the kernels demand identical types.
    """
    table = pa.table(
        {
            "coarse": pa.array([Decimal("1.50"), Decimal("2.25")], pa.decimal128(10, 2)),
            "fine": pa.array([Decimal("1.5000"), Decimal("9.9999")], pa.decimal128(12, 4)),
        }
    )
    return _register(duck, "amounts", table)


def test_comparing_decimals_of_differing_scale(duck, side_by_side_decimals):
    ds = side_by_side_decimals.select((col("coarse") == col("fine")).alias("same"))
    assert_same(ds.collect(), duck.sql("SELECT coarse = fine AS same FROM amounts"))


def test_greatest_over_decimals_of_differing_scale(duck, side_by_side_decimals):
    ds = side_by_side_decimals.select(bt.greatest(col("coarse"), col("fine")).alias("g"))
    assert_same(ds.collect(), duck.sql("SELECT GREATEST(coarse, fine) AS g FROM amounts"))


@pytest.fixture
def timestamps(duck):
    """The same instants written at millisecond and at microsecond resolution."""
    ms = pa.table({"ts": pa.array([1_000, 2_000], pa.timestamp("ms")), "src": pa.array(["a", "b"])})
    us = pa.table(
        {"ts": pa.array([2_000_000, 4_000_000], pa.timestamp("us")), "src": pa.array(["c", "d"])}
    )
    return _register(duck, "ms", ms), _register(duck, "us", us)


def test_union_of_timestamps_with_differing_resolution(duck, timestamps):
    ms, us = timestamps
    ds = ms.union(us)
    assert ds.schema.field("ts").type == pa.timestamp("us")
    assert_same(ds.collect(), duck.sql("SELECT * FROM ms UNION ALL SELECT * FROM us"))


def test_comparing_timestamps_of_differing_resolution(duck):
    """Two seconds in millis is the same instant as two seconds in micros."""
    table = pa.table(
        {
            "coarse": pa.array([1_000, 2_000], pa.timestamp("ms")),
            "fine": pa.array([1_000_000, 9_000_000], pa.timestamp("us")),
        }
    )
    ds = _register(duck, "instants", table)
    assert_same(
        ds.select((col("coarse") == col("fine")).alias("same")).collect(),
        duck.sql("SELECT coarse = fine AS same FROM instants"),
    )


def test_union_of_a_date_with_a_timestamp(duck):
    dates = _register(duck, "dates", pa.table({"d": pa.array([0, 1], pa.date32())}))
    stamps = _register(
        duck, "stamps", pa.table({"d": pa.array([86_400_000_000 * 2], pa.timestamp("us"))})
    )
    ds = dates.union(stamps)
    assert ds.schema.field("d").type == pa.timestamp("us")
    assert_same(ds.collect(), duck.sql("SELECT * FROM dates UNION ALL SELECT * FROM stamps"))


def test_union_of_a_boolean_with_an_integer(duck):
    flags = _register(duck, "flags", pa.table({"v": pa.array([True, False], pa.bool_())}))
    counts = _register(duck, "counts", pa.table({"v": pa.array([7, 0], pa.int64())}))
    ds = flags.union(counts)
    assert ds.schema.field("v").type == pa.int64()
    assert_same(ds.collect(), duck.sql("SELECT * FROM flags UNION ALL SELECT * FROM counts"))


def test_string_concat_does_not_promote_its_operands(duck):
    """`||` renders each operand on its own terms, so the lattice must not reach it.

    This is the boundary of the lattice, and it is a boundary the lattice crossed once:
    teaching it that a boolean widens into a number for a UNION also made
    ``bool || double`` render the boolean as ``'1.0'`` where DuckDB renders ``'true'``.
    Concatenation casts each side to text itself and never wanted a common numeric type.
    """
    table = pa.table(
        {
            "i": pa.array([1, 2], pa.int64()),
            "f": pa.array([1.5, 2.5], pa.float64()),
            "b": pa.array([True, False], pa.bool_()),
        }
    )
    ds = _register(duck, "mixed", table)
    assert_same(
        ds.select(
            bt.concat_str(col("i").cast("string"), col("f").cast("string")).alias("int_float"),
            bt.concat_str(col("f").cast("string"), col("b").cast("string")).alias("float_bool"),
        ).collect(),
        duck.sql(
            "SELECT CAST(i AS VARCHAR) || CAST(f AS VARCHAR) AS int_float, "
            "CAST(f AS VARCHAR) || CAST(b AS VARCHAR) AS float_bool FROM mixed"
        ),
    )


def test_a_union_with_no_common_type_still_raises(duck):
    """Declining is the point of the `None` arm — a lenient cast would null a branch."""
    ints = _register(duck, "i", pa.table({"v": pa.array([1], pa.int64())}))
    strs = _register(duck, "s", pa.table({"v": pa.array(["x"], pa.string())}))
    with pytest.raises(Exception, match=r"(?i)incompatible|type"):
        ints.union(strs).collect()


def test_joining_on_decimals_of_differing_scale(duck, decimals):
    """`1.50` and `1.5000` are the same number, so the join must match them.

    A join compares its keys through a byte-wise row encoder that needs the two sides to
    have the identical Arrow type, which two sources almost never do. Widening both keys
    to the pair's common supertype is what lets the encoder run; it cannot change a value,
    so no match is gained or lost.
    """
    coarse, fine = decimals
    joined = coarse.join(fine, left_on="amt", right_on="amt", how="inner")
    # A join emits one key column, named after the left key, so the oracle selects one too.
    assert_same(
        joined.collect(),
        duck.sql("SELECT c.amt FROM coarse c JOIN fine f ON c.amt = f.amt"),
    )


def test_joining_on_timestamps_of_differing_resolution(duck, timestamps):
    """Two seconds in millis is the same instant as two seconds in micros."""
    ms, us = timestamps
    joined = ms.join(us, left_on="ts", right_on="ts", how="inner")
    assert_same(
        joined.collect(),
        duck.sql("SELECT m.ts, m.src, u.src AS src_right FROM ms m JOIN us u ON m.ts = u.ts"),
    )


def test_joining_on_an_int_key_against_a_float_key(duck):
    """The int/float mix is the promotion users hit most, and a join has to take it too."""
    ints = _register(duck, "ik", pa.table({"k": pa.array([1, 2, 3], pa.int64()), "l": [*"abc"]}))
    floats = _register(
        duck, "fk", pa.table({"k": pa.array([2.0, 3.0, 4.5], pa.float64()), "r": [*"xyz"]})
    )
    joined = ints.join(floats, left_on="k", right_on="k", how="inner")
    assert_same(
        joined.collect(),
        duck.sql("SELECT i.k, i.l, f.r FROM ik i JOIN fk f ON i.k = f.k"),
    )


def test_a_left_join_on_an_all_null_key_keeps_every_left_row(duck):
    """An all-null key matches nothing, and the outer side must still come through."""
    nulls = _register(duck, "nk", pa.table({"k": pa.array([None, None], pa.null()), "l": [*"ab"]}))
    typed = _register(duck, "tk", pa.table({"k": pa.array([1, 2], pa.int64()), "r": [*"xy"]}))
    joined = nulls.join(typed, left_on="k", right_on="k", how="left")
    assert_same(
        joined.collect(),
        duck.sql("SELECT n.k, n.l, t.r FROM nk n LEFT JOIN tk t ON n.k = t.k"),
    )


def test_a_join_key_pair_with_no_common_type_still_raises(duck):
    """Coercion widens; it never invents. An int key against a string key is a bug."""
    ints = _register(duck, "a", pa.table({"k": pa.array([1], pa.int64())}))
    strs = _register(duck, "b", pa.table({"k": pa.array(["x"], pa.string())}))
    with pytest.raises(Exception, match="join key type mismatch"):
        ints.join(strs, on="k")
