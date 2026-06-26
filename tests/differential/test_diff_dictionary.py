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
    return sorted(tuple(r.values()) for r in t.to_pylist())


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
