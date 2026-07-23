"""`.list` set-op (intersect/difference/union) edge parity with DuckDB.

Covers a corner the wave-1 fix (signed-zero folding) did not: two list columns whose
element types differ in numeric width (`List<Int64>` vs `List<Float64>`). The engine
concatenated the two children before comparing them, which errored on a type mismatch
("cannot concatenate arrays of different data types") where DuckDB coerces to the wider
numeric type. The oracle is DuckDB `list_intersect`.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def _same_list(got: list, exp: list) -> None:
    assert len(got) == len(exp), f"{got} vs {exp}"
    for g, e in zip(got, exp, strict=True):
        if g is None or e is None:
            assert g is e, f"{got} vs {exp}"
            continue
        assert [float(x) for x in g] == [float(y) for y in e], f"{got} vs {exp}"


def test_intersect_coerces_mismatched_numeric_element_types():
    # List<Int64> ∩ List<Float64> must promote, not error. DuckDB coerces to the wider
    # numeric type: list_intersect([1,2,3],[2.0,3.0,4.0]) == [2.0, 3.0].
    ds = bt.from_pydict({"a": [[1, 2, 3]], "b": [[2.0, 3.0, 4.0]]})
    got = ds.select(r=col("a").list.intersect(col("b"))).collect().to_pydict()["r"]
    _same_list(got, [[2.0, 3.0]])


def test_union_and_difference_coerce_mismatched_numeric_element_types():
    ds = bt.from_pydict({"a": [[1, 2]], "b": [[2.0, 3.0]]})
    out = (
        ds.select(
            u=col("a").list.union(col("b")),
            d=col("a").list.difference(col("b")),
        )
        .collect()
        .to_pydict()
    )
    _same_list(out["u"], [[1.0, 2.0, 3.0]])  # left distinct ++ right-only
    _same_list(out["d"], [[1.0]])  # 1 is not in b once widened
