"""Dictionary-encoded inputs decode at the FFI and behave like plain columns.

Arrow `Dictionary` columns (categoricals, low-cardinality strings from Parquet) used
to pass through the boundary unhandled. The FFI now decodes them to their value type
(then widens narrow numerics), so every operator sees a plain column — and the result
must be identical to feeding the same data un-encoded, and match DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


def _plain():
    return pa.table(
        {
            "k": ["a", "b", "a", "c", "b", "a"],
            "n": [1, 2, 3, 4, 5, 6],
        }
    )


def _dict_encoded():
    t = _plain()
    return t.set_column(0, "k", t.column("k").dictionary_encode()).set_column(
        1, "n", pa.chunked_array([pa.array([1, 2, 3, 4, 5, 6], pa.int32()).dictionary_encode()])
    )


def _norm(t):
    # None-safe, type-tolerant sort key: rows may carry NULLs alongside strings/ints,
    # and `None < str` / `int < str` raise under the default comparison. Sort each cell by
    # (is-null, type-name, repr) so a multiset of rows orders deterministically regardless
    # of NULLs or mixed cell types.
    rows = [tuple(r.values()) for r in t.to_pylist()]
    key = lambda row: tuple((v is None, type(v).__name__, repr(v)) for v in row)  # noqa: E731
    return sorted(rows, key=key)


def test_dictionary_group_by_equals_plain():
    plain = bt.from_arrow(_plain()).group_by("k").agg(s=col("n").sum()).collect()
    encoded = bt.from_arrow(_dict_encoded()).group_by("k").agg(s=col("n").sum()).collect()
    assert _norm(encoded) == _norm(plain)


def test_dictionary_filter_select_equals_plain():
    plain = bt.from_arrow(_plain()).filter(col("k") == "a").select("k", "n").collect()
    encoded = bt.from_arrow(_dict_encoded()).filter(col("k") == "a").select("k", "n").collect()
    assert _norm(encoded) == _norm(plain)


def test_dictionary_decodes_to_value_type():
    # The dict<int32> column decodes to the widened int64 (not a dictionary type).
    out = bt.from_arrow(_dict_encoded()).select("n").collect()
    assert out.schema.field("n").type == pa.int64()
    out_k = bt.from_arrow(_dict_encoded()).select("k").collect()
    assert out_k.schema.field("k").type == pa.string()


def _dict_with_null_value():
    # A dictionary whose VALUES array carries a null, referenced by several rows — the shape
    # Parquet emits for a low-cardinality string column that contains NULLs. The null lives
    # once in the value list but decodes to a null in every row that references it.
    keys = pa.array([0, 1, 0, 1, 2], type=pa.int32())
    values = pa.array(["a", None, "c"])
    d = pa.DictionaryArray.from_arrays(keys, values)
    return pa.table({"k": d, "n": [1, 2, 3, 4, 5]})


def test_dictionary_with_null_value_group_by(duck):
    # Regression: the FFI data-loss guard (meant only for UInt64->Int64 overflow) summed
    # physical null counts across the dictionary's buffers, so decoding a null dictionary
    # *value* — which replicates to N null rows — was mis-flagged as data loss and the whole
    # batch was rejected with a bogus "value exceeds the Int64 range" error. It must instead
    # decode and group like the plain column, matching DuckDB (NULL forms one group).
    t = _dict_with_null_value()
    plain = t.set_column(0, "k", t.column("k").combine_chunks().dictionary_decode())
    duck.register("t", plain)

    out = bt.from_arrow(t).group_by("k").agg(s=col("n").sum()).collect()
    expected = duck.sql("SELECT k, SUM(n) AS s FROM t GROUP BY k")
    assert_same(out, expected)


def test_dictionary_with_null_value_equals_plain():
    t = _dict_with_null_value()
    plain = t.set_column(0, "k", t.column("k").combine_chunks().dictionary_decode())
    enc = bt.from_arrow(t).select("k", "n").collect()
    dec = bt.from_arrow(plain).select("k", "n").collect()
    assert _norm(enc) == _norm(dec)
