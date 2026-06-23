"""`try_cast` (DuckDB ``TRY_CAST``) — unconvertible values become NULL, not an error.

Locks in parity with DuckDB's ``TRY_CAST``: strict ``cast`` errors on a bad value,
while ``try_cast`` yields NULL. The safe-ingest spelling for dirty source columns.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col


def test_try_cast_string_to_int_nulls_bad_values(duck):
    """A dirty string column → int64: unparseable values become NULL (vs DuckDB).

    Note: float-like strings (e.g. ``"3.5"``) are a known DuckDB↔Arrow divergence —
    DuckDB parses them as a number and rounds (``"3.5"`` → 4), Arrow returns NULL.
    Batcher follows Arrow here; this test exercises integer and junk strings only,
    which is the safe-ingest contract `try_cast` is for.
    """
    from conftest import assert_same

    t = pa.table({"s": ["10", "-5", "abc", "", "42", None, "999"]})
    duck.register("tc", t)
    out = bt.from_arrow(t).select(si=col("s").try_cast("int64")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(s AS BIGINT) si FROM tc"))


def test_try_cast_string_to_float(duck):
    """String → float64 with junk values nulled (vs DuckDB ``TRY_CAST``)."""
    from conftest import assert_same

    t = pa.table({"s": ["1.5", "nope", "-2.25", None, "1e3"]})
    duck.register("tcf", t)
    out = bt.from_arrow(t).select(f=col("s").try_cast("float64")).collect()
    assert_same(out, duck.sql("SELECT TRY_CAST(s AS DOUBLE) f FROM tcf"))


def test_try_cast_dataset_strict_false(duck):
    """`Dataset.cast(strict=False)` applies ``TRY_CAST`` to every named column."""
    from conftest import assert_same

    t = pa.table({"a": ["1", "x", "3"], "b": ["9", "8", "bad"]})
    duck.register("tcd", t)
    out = bt.from_arrow(t).cast({"a": "int64", "b": "int64"}, strict=False).collect()
    assert_same(
        out,
        duck.sql("SELECT TRY_CAST(a AS BIGINT) a, TRY_CAST(b AS BIGINT) b FROM tcd"),
    )


def test_strict_cast_still_errors_on_bad_value():
    """Strict `cast` (the default) errors on an unconvertible value rather than
    silently nulling — the behavior that distinguishes it from `try_cast`."""
    t = pa.table({"s": ["10", "abc"]})
    with pytest.raises(Exception):  # noqa: B017 — engine raises on the invalid cast
        bt.from_arrow(t).select(si=col("s").cast("int64")).collect()
