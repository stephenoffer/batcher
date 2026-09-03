"""`batch_format` chooses what the `fn` speaks, never what the query returns.

`map_batches` hands each batch to a user function in the framework the caller named, and the
documented contract is that this is a *presentation* choice. The query's schema is fixed
before the `fn` runs -- `LogicalPlan.available_schema` answers `Dataset.schema` from static
analysis, on no rows -- so a format that changes a column's type on the way back makes the
engine's declared schema disagree with what it delivers.

Two round-trips did exactly that, in opposite directions and for unrelated reasons:

- NumPy and pandas carry a string column as `object`, and an `object` column holding no
  non-null value says nothing about what it held. Arrow infers `null`, so a `string` column
  returned `null`-typed -- but only on the batches with no rows in them. Repaired by
  `interop.formats.restore_null_typed_columns`, and covered by
  `test_map_batches_empty_column_type.py`.
- Polars lays every variable-length type out at 64-bit offsets, so `string` came back
  `large_string`, `binary` as `large_binary` and `list` as `large_list` -- on *every* batch,
  not only the empty ones. `Dataset.schema` said `string`, `collect()` returned
  `large_string`, and a Parquet file written from that plan was `large_string` on disk.
  Repaired by `interop.formats.restore_widened_columns`.

Neither was caught, and the reason is worth recording: the empty-column test enumerated
`["pyarrow", "numpy", "pandas"]` by hand while `interop.formats.FORMATS` has six entries, so
the format with the defect was simply not in the list. This file derives its cases from
`FORMATS` instead, and classifies rather than skips -- a seventh format has to be placed in
one of the two groups below before these tests will run at all.

The assertions are of the declared schema against the delivered one, which is the property
that actually matters and is stronger than comparing formats to each other: if every format
were wrong in the same way, a cross-format comparison would still pass.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.interop.formats import FORMATS

pytestmark = pytest.mark.integration

#: Formats able to carry a non-numeric column to the `fn` and back.
_CARRIES_EVERYTHING = frozenset({"pyarrow", "numpy", "pandas", "polars"})

#: Tensor formats, which cannot represent a string or binary column at all. `map_batches`
#: warns and drops those columns by design, so the fidelity claim applies to what they *can*
#: carry -- which is why they are classified here rather than skipped.
_NUMERIC_ONLY = frozenset({"torch", "jax"})

#: One column per variable-length layout polars re-lays, plus numeric controls no round-trip
#: has ever moved -- so a test failing on every column is distinguishable from one failing
#: only on the widened family. A dictionary column is deliberately *not* here; it diverges
#: for an unrelated and still-open reason, pinned on its own at the end of this file.
_TABLE = pa.table(
    {
        "s": pa.array(["a", "b", "c"], pa.string()),
        "bn": pa.array([b"x", b"y", b"z"], pa.binary()),
        "l": pa.array([[1], [2, 3], []], pa.list_(pa.int64())),
        "i": pa.array([1, 2, 3], pa.int64()),
        "f": pa.array([1.5, 2.5, 3.5], pa.float64()),
    }
)


def _types(schema: pa.Schema) -> dict[str, str]:
    return {name: str(t) for name, t in zip(schema.names, schema.types, strict=True)}


def _available(fmt: str) -> None:
    pytest.importorskip({"pyarrow": "pyarrow", "numpy": "numpy"}.get(fmt, fmt))


def test_every_format_is_classified():
    """A seventh `batch_format` must be placed before the tests below can cover it.

    This is the assertion the hand-written `["pyarrow", "numpy", "pandas"]` list did not
    make, and its absence is the whole reason the polars defect survived.
    """
    assert set(FORMATS) == _CARRIES_EVERYTHING | _NUMERIC_ONLY
    assert not (_CARRIES_EVERYTHING & _NUMERIC_ONLY)


@pytest.mark.parametrize("fmt", sorted(_CARRIES_EVERYTHING))
@pytest.mark.parametrize("rows", ["full", "empty"])
def test_an_identity_function_returns_the_schema_it_was_declared(fmt, rows):
    """The contract, on both a populated batch and one the filter emptied.

    Holding the result against `Dataset.schema` rather than against another format is what
    makes this a statement about the engine's promise instead of about consensus between
    round-trips.
    """
    _available(fmt)
    ds = bt.from_arrow(_TABLE)
    if rows == "empty":
        ds = ds.filter(bt.col("i") > 100)
    ds = ds.map_batches(lambda b: b, batch_format=fmt)

    declared = _types(ds.schema)
    got = ds.collect()
    assert _types(got.schema) == declared, (
        f"batch_format={fmt!r} changed the query's own schema on a {rows} batch"
    )
    assert declared == _types(_TABLE.schema), "the declared schema drifted from the input"


@pytest.mark.parametrize("fmt", sorted(_NUMERIC_ONLY))
def test_a_tensor_format_keeps_the_numeric_columns_it_can_carry(fmt):
    """The applicable half of the contract for torch/jax, which cannot hold strings.

    Selecting the numeric columns explicitly is what the warning tells a user to do, so this
    is the supported shape rather than a workaround invented for the test.
    """
    _available(fmt)
    ds = (
        bt.from_arrow(_TABLE)
        .select("i", "f")
        .map_batches(lambda b: b, batch_format=fmt, output_columns=["i", "f"])
    )
    assert _types(ds.collect().schema) == {"i": "int64", "f": "double"}


@pytest.mark.parametrize("fmt", sorted(_CARRIES_EVERYTHING))
def test_the_written_file_carries_the_declared_type(fmt, tmp_path):
    """The consequence that made this worth fixing rather than tolerating.

    A type that only differs in memory could be argued as cosmetic. One that reaches Parquet
    is a schema on disk that disagrees with the schema the engine published for it, and every
    later reader inherits that.

    The comparison is against the *same write without a `fn`*, not against the input schema.
    Parquet renames a list's element field (`item` becomes `element`) on its own, with or
    without a UDF, so holding the file to the in-memory schema would fail on a convention
    that has nothing to do with what is being tested. Differencing the two writes isolates
    exactly the `fn`'s contribution.
    """
    _available(fmt)
    pq = pytest.importorskip("pyarrow.parquet")

    def written(ds, name):
        out = tmp_path / f"{name}.parquet"
        ds.write.parquet(str(out))
        assert out.exists(), "the write produced no parquet file"
        return _types(pq.read_schema(out))

    plain = written(bt.from_arrow(_TABLE), f"plain-{fmt}")
    mapped = written(
        bt.from_arrow(_TABLE).map_batches(lambda b: b, batch_format=fmt), f"mapped-{fmt}"
    )
    assert mapped == plain, f"batch_format={fmt!r} changed the types written to disk"


def test_a_genuine_retype_by_the_function_is_not_reverted():
    """The negative control: the repair must be narrow, not a blanket schema restore.

    A `fn` that really does change a column's type is doing what `map_batches` is for. If the
    repair reverted that too, it would be silently discarding the user's work rather than
    correcting a round-trip artifact -- and every assertion above would still pass.
    """
    out = (
        bt.from_arrow(_TABLE)
        .map_batches(
            lambda b: pa.RecordBatch.from_arrays([b.column("i").cast(pa.float64())], names=["i"]),
            batch_format="pyarrow",
            output_columns=["i"],
        )
        .collect()
    )
    assert _types(out.schema) == {"i": "double"}, "the fn's own retype was undone"


# --- a second divergence, not fixed: dictionary encoding across the UDF boundary ----------


@pytest.mark.xfail(
    strict=True,
    reason="map_batches hands the fn a dictionary column the plan declares as string, and "
    "returns it encoded; a plain collect() decodes it. Pinned rather than fixed because "
    "where to decode is a design call with a real cost either way.",
)
def test_a_dictionary_column_matches_its_declared_type_across_a_udf():
    """Declared `string`, delivered `dictionary<values=string, indices=int32>`.

    Measured on every `batch_format`, so it is not a framework round-trip artifact like the
    widening above -- the `fn` is *handed* the dictionary in the first place, while
    `Dataset.schema` and every non-UDF path say `string`:

        no map_batches        declared string   actual string
        map_batches pyarrow   declared string   actual dictionary<...>
        map_batches polars    declared string   actual dictionary<...>
        select / filter       declared string   actual string

    This is the CPU counterpart of the defect `.claude/rules/device-tier.md` cites as the
    reason to derive type rules rather than restate them -- "how the device came to keep a
    `dictionary` column encoded where the engine decodes it".

    Left open deliberately. Decoding before the `fn` runs makes it see what the plan
    declares and costs the dictionary's compression on every UDF batch; decoding the result
    keeps the `fn` efficient but needs the plan's declared schema at a call site that
    currently only has the input batch's. Both are defensible and neither is a repair to
    make quietly in passing. `strict=True` so that whoever chooses has to retire this.
    """
    table = pa.table({"d": pa.array(["a", "b", "a"]).dictionary_encode(), "i": pa.array([1, 2, 3])})
    ds = bt.from_arrow(table).map_batches(lambda b: b, batch_format="pyarrow")
    assert _types(ds.collect().schema) == _types(ds.schema)
