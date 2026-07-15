"""Schema-inference completeness: `Dataset.schema` (declared) == `collect().schema`.

Continues the B18/B19/B67 sweep into the list/date/struct/map accessors whose
result type `infer_type` did not previously derive — so `Dataset.schema` fell back
to a zero-row execution that reports ``null`` for a column whose real type is
concrete. Every case below fails (declared ``null`` != actual) without the
`plan/types/infer.py` additions.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt

_TBL = pa.table(
    {
        "li": pa.array([[1, 2], [3], None], pa.list_(pa.int64())),
        "lf": pa.array([[1.0, 2.0], [3.0], None], pa.list_(pa.float64())),
        "ts": pa.array(
            [dt.datetime(2020, 1, 1), dt.datetime(2021, 6, 1), None], pa.timestamp("us")
        ),
        "d": pa.array([dt.date(2020, 1, 1), dt.date(2021, 6, 1), None], pa.date32()),
        "st": pa.array(
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, None],
            pa.struct([("a", pa.int64()), ("b", pa.string())]),
        ),
        "mp": pa.array([[("k", 1)], [("j", 2)], None], pa.map_(pa.string(), pa.int64())),
    }
)

C = bt.col


def _cases():
    return [
        # list — count/index → int64
        ("list.len", C("li").list.len(), "li"),
        ("list.n_unique", C("li").list.n_unique(), "li"),
        ("list.arg_max", C("li").list.arg_max(), "li"),
        ("list.arg_min", C("li").list.arg_min(), "li"),
        # list — element/list-preserving
        ("list.reverse", C("li").list.reverse(), "li"),
        ("list.sort", C("li").list.sort(), "li"),
        ("list.unique", C("li").list.unique(), "li"),
        ("list.get", C("li").list.get(0), "li"),
        ("list.first", C("li").list.first(), "li"),
        ("list.contains", C("li").list.contains(1), "li"),
        ("list.position", C("li").list.position(1), "li"),
        ("list.slice", C("li").list.slice(0, 1), "li"),
        ("list.dot", C("lf").list.dot(C("lf")), "lf"),
        ("list.l2_distance", C("lf").list.l2_distance(C("lf")), "lf"),
        ("list.intersect", C("li").list.intersect(C("li")), "li"),
        ("list.difference", C("li").list.difference(C("li")), "li"),
        # date
        ("dt.truncate_ts", C("ts").dt.truncate("day"), "ts"),
        ("dt.truncate_date", C("d").dt.truncate("month"), "d"),
        ("dt.strftime", C("ts").dt.strftime("%Y"), "ts"),
        ("dt.offset_by", C("ts").dt.offset_by("1mo"), "ts"),
        ("dt.convert_timezone", C("ts").dt.convert_timezone("UTC", "America/New_York"), "ts"),
        # struct / map
        ("struct.field_int", C("st").struct.field("a"), "st"),
        ("struct.field_str", C("st").struct.field("b"), "st"),
        ("map.keys", C("mp").map.keys(), "mp"),
        ("map.values", C("mp").map.values(), "mp"),
        ("map.get", C("mp").map.get("k"), "mp"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("name,expr,incol", _cases(), ids=[c[0] for c in _cases()])
def test_declared_schema_matches_actual(name: str, expr, incol: str) -> None:
    ds = bt.from_arrow(_TBL).select(bt.col(incol).alias("keep"), expr.alias("v"))
    declared = ds.schema.field("v").type
    actual = ds.collect().schema.field("v").type
    assert declared == actual, f"{name}: declared {declared} != actual {actual}"
    # The whole point of the fix: the declared type is concrete, not the null
    # zero-row fallback.
    assert not pa.types.is_null(declared), f"{name}: schema still falls back to null"
