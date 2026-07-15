"""Regression: `Dataset.schema` must not lie about a derived column's dtype.

`Dataset.schema` answers without scanning by inferring each output column's Arrow
type (`plan.types.infer_type`). When inference is wrong (or silently absent and the
zero-row fallback misreports), the declared schema disagrees with the type the
engine actually produces — a B18/B19-class bug. Each case below asserts the
*declared* schema equals the *executed* schema for a function whose output type is
certain. Failing cases were real defects:

- true division declared ``null`` (actually ``float64``);
- ``str.reverse``/``translate``/``unhex`` and ``str.regexp_extract_all`` declared
  ``null`` (actually ``string`` / ``list<string>``);
- every ``dt`` accessor (``dayname``/``year``/``last_day``/…) declared ``null``;
- ``str.to_datetime`` (``Strptime``) declared ``null`` (actually ``timestamp[us]``);
- ``nullif`` declared ``null`` (actually its left operand's type).
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt


def _table() -> object:
    return bt.from_arrow(
        pa.table(
            {
                "i": pa.array([1, 2, 3], pa.int64()),
                "i32": pa.array([1, 2, 3], pa.int32()),
                "f": pa.array([1.5, 2.5, 3.5], pa.float64()),
                "s": pa.array(["ab", "cd", "ef"]),
                "ts": pa.array([dt.datetime(2021, 3, 14, 5, 6, 7)] * 3, pa.timestamp("us")),
            }
        )
    )


def _cases() -> list[tuple[str, object]]:
    c = bt.col
    return [
        ("div_int", c("i") / c("i")),
        ("div_int_float", c("i") / c("f")),
        ("div_narrow", c("i32") / c("i32")),
        ("str_reverse", c("s").str.reverse()),
        ("str_translate", c("s").str.translate("a", "b")),
        ("str_unhex", c("s").str.unhex()),
        ("str_regexp_extract_all", c("s").str.regexp_extract_all("(.)")),
        ("str_to_datetime", c("s").str.to_datetime("%Y")),
        ("dt_dayname", c("ts").dt.dayname()),
        ("dt_monthname", c("ts").dt.monthname()),
        ("dt_year", c("ts").dt.year()),
        ("dt_epoch", c("ts").dt.epoch()),
        ("dt_days_in_month", c("ts").dt.days_in_month()),
        ("dt_iso_year", c("ts").dt.iso_year()),
        ("dt_is_leap_year", c("ts").dt.is_leap_year()),
        ("dt_last_day", c("ts").dt.last_day()),
        ("nullif_int", bt.nullif(c("i"), bt.lit(2))),
        ("nullif_float_left", bt.nullif(c("f"), c("i"))),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("name,expr", _cases(), ids=[c[0] for c in _cases()])
def test_declared_schema_matches_execution(name: str, expr: object) -> None:
    ds = _table().select(o=expr)
    declared = ds.schema.field("o").type
    actual = ds.collect().schema.field("o").type
    assert declared == actual, f"{name}: declared {declared} != actual {actual}"
