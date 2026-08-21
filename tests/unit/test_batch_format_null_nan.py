"""NaN and null through a `batch_format` round-trip: what survives, and what is announced.

`pyarrow` distinguishes a null from a NaN; NumPy and pandas each have one representation for
both, so a `fn` that returns its input *unchanged* can still change the data. The two lose it
in opposite directions -- NumPy resolves the pair to NaN, pandas to null -- which is why the
matrix below is asserted rather than described: it is the only place the divergence is stated
as a fact that fails when it moves.

Announcing the loss is the part this module gates, and two announcements were missing. A
nullable *float* column widened to NaN silently on the NumPy path while a nullable int or
bool beside it warned. And a nullable *int* column changed type -- `int64` in, `double` back
-- on the pandas path with nothing said, which stayed quiet because the pandas *values* are
fine: a null becomes NaN in the frame and `from_pandas` reads it back as a null.

Nothing here changes what the conversions compute. Every assertion about data passes before
the fix as well as after; only the four announcement tests move.
"""

from __future__ import annotations

import math
import warnings

import pyarrow as pa
import pytest

from batcher.ml.batch_format import result_to_arrowable, to_format


def _round_trip(column: pa.Array, fmt: str) -> list:
    """`column` handed to a `fn` as `fmt` and returned unchanged, back as Python values."""
    batch = pa.RecordBatch.from_arrays([column], names=["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = result_to_arrowable(to_format(batch, fmt), fmt)
    if isinstance(back, dict):
        back = pa.RecordBatch.from_pydict({k: pa.array(v) for k, v in back.items()})
    return back.column(0).to_pylist()


def _shape(values: list) -> list[str]:
    """Values as `null` / `NaN` / a number, so the two missing-value kinds stay distinct."""
    return [
        "null" if v is None else "NaN" if isinstance(v, float) and math.isnan(v) else repr(v)
        for v in values
    ]


NAN = float("nan")
BOTH = pa.array([NAN, None, 3.0], type=pa.float64())


def test_pyarrow_round_trip_keeps_nan_and_null_distinct():
    """The lossless format, and the oracle the other two are measured against."""
    assert _shape(_round_trip(BOTH, "pyarrow")) == ["NaN", "null", "3.0"]


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        # NumPy has only NaN, so the null is resolved to it.
        ("numpy", ["NaN", "NaN", "3.0"]),
        # `from_pandas` reads every NaN in a float column as a null, so the pair goes the
        # other way. Round-tripping a null is therefore *correct* here and a NaN is not.
        ("pandas", ["null", "null", "3.0"]),
    ],
)
def test_numpy_and_pandas_conflate_nan_and_null_in_opposite_directions(fmt, expected):
    """A known limitation, asserted so it cannot change direction unnoticed.

    Neither is a defect that can be fixed inside the format: a NumPy `float64` array and a
    pandas `float64` column each hold one missing-value representation, so the conversion back
    to Arrow has to choose. What must not happen is the choice moving silently.
    """
    assert _shape(_round_trip(BOTH, fmt)) == expected


@pytest.mark.parametrize(
    ("arrow_type", "present"),
    [(pa.int64(), 1), (pa.bool_(), True), (pa.float64(), 1.0)],
    ids=["int64", "bool", "float64"],
)
def test_every_nullable_numeric_widening_to_numpy_is_announced(arrow_type, present):
    """All three widen a null to NaN, so all three say so.

    `float64` is the regression: its nulls became NaN with nothing said, while an int or bool
    column doing the identical thing warned. It is also the likeliest of the three to be the
    label column the warning exists for.
    """
    column = pa.array([present, None, present], type=arrow_type)
    batch = pa.RecordBatch.from_arrays([column], names=["x"])
    with pytest.warns(UserWarning, match="NaN"):
        to_format(batch, "numpy")


@pytest.mark.parametrize(
    "column",
    [
        pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        pa.array([1.0, NAN, 3.0], type=pa.float64()),
        pa.array([1, 2, 3], type=pa.int64()),
    ],
    ids=["float-no-nulls", "float-nan-but-no-nulls", "int-no-nulls"],
)
def test_a_column_that_loses_nothing_is_not_announced(column):
    """The announcement tracks the loss, not the type -- otherwise it is noise on every batch.

    A float column carrying a NaN and no nulls is the case that pins this: nothing is widened,
    so warning would fire on ordinary ML data and be filtered out, taking the real one with it.
    """
    batch = pa.RecordBatch.from_arrays([column], names=["x"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        to_format(batch, "numpy")
    assert [str(w.message) for w in caught] == []


@pytest.mark.parametrize("fmt", ["numpy", "pandas"])
def test_a_nullable_integer_widening_to_float_is_announced(fmt):
    """An identity `fn` returns `double` where the engine had `int64`, so both formats say so.

    The value survives on the pandas path -- a null becomes NaN in the frame and reads back as
    a null -- which is exactly what made this one quiet: nothing looks wrong in the data. The
    *type* is what moves, and `pandas` announced nothing until this test.
    """
    batch = pa.RecordBatch.from_arrays([pa.array([1, None, 3], type=pa.int64())], names=["x"])
    with pytest.warns(UserWarning, match="int64"):
        to_format(batch, fmt)


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [("pyarrow", "int64"), ("polars", "int64"), ("numpy", "double"), ("pandas", "double")],
)
def test_which_formats_preserve_a_nullable_integer_type(fmt, expected):
    """The escape hatch the warnings name, asserted so it stays true.

    A warning that recommends `batch_format='pyarrow'` is only useful while that format
    actually keeps the type, so the recommendation and the behaviour are pinned together.
    """
    if fmt == "polars":
        pytest.importorskip("polars")
    column = pa.array([1, None, 3], type=pa.int64())
    batch = pa.RecordBatch.from_arrays([column], names=["x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = result_to_arrowable(to_format(batch, fmt), fmt)
    if isinstance(back, dict):
        back = pa.RecordBatch.from_pydict({k: pa.array(v) for k, v in back.items()})
    assert str(back.column(0).type) == expected


def test_a_nullable_integer_is_not_announced_when_it_has_no_nulls():
    """No null, no widening, no warning -- on both lossy formats."""
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3], type=pa.int64())], names=["x"])
    for fmt in ("numpy", "pandas"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            to_format(batch, fmt)
        assert [str(w.message) for w in caught] == [], fmt


def test_restore_null_typed_columns_puts_back_only_what_the_reference_declares():
    """The unit of the empty-column restore: a `null` column with a reference type, and only that.

    An invented column has no entry in the reference, and a column that came back with a real
    type is not second-guessed -- so neither moves.
    """
    from batcher.interop.formats import restore_null_typed_columns

    reference = pa.schema(
        [pa.field("s", pa.string()), pa.field("i", pa.int64()), pa.field("n", pa.null())]
    )
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([None, None], type=pa.null()),  # lost its string type
            pa.array([1, 2], type=pa.int64()),  # intact, leave alone
            pa.array([None, None], type=pa.null()),  # null in the reference too
            pa.array([None, None], type=pa.null()),  # invented by the fn
        ],
        names=["s", "i", "n", "invented"],
    )
    out = restore_null_typed_columns(batch, reference)
    assert dict(zip(out.schema.names, (str(t) for t in out.schema.types), strict=True)) == {
        "s": "string",
        "i": "int64",
        "n": "null",
        "invented": "null",
    }
    assert out.column("s").to_pylist() == [None, None]


def test_restore_null_typed_columns_is_a_no_op_without_a_null_column():
    """The common case returns the very same object, so the check costs a schema scan."""
    from batcher.interop.formats import restore_null_typed_columns

    batch = pa.RecordBatch.from_arrays([pa.array(["a"], type=pa.string())], names=["s"])
    assert restore_null_typed_columns(batch, batch.schema) is batch
