"""`Dataset.upsample`: a regular time grid per group, against Polars and hand-built answers.

Polars is the oracle for the grid itself on data where every observation sits on the grid.
Where they part ways on purpose (an observation between grid points, which Polars' left join
onto the grid drops and Batcher keeps) the expected rows are written out by hand.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pl = pytest.importorskip("polars")

D = dt.datetime


def _rows(table: pa.Table, keys: list[str]) -> list[tuple]:
    rows = [
        tuple(r.values())
        for r in table.select(keys + [c for c in table.column_names if c not in keys]).to_pylist()
    ]
    return sorted(rows, key=lambda r: tuple((v is None, str(v)) for v in r))


def _readings() -> pa.Table:
    return pa.table(
        {
            "g": ["a", "a", "b", "b", "b"],
            "t": pa.array(
                [
                    D(2024, 1, 1, 0),
                    D(2024, 1, 1, 3),
                    D(2024, 1, 1, 1),
                    D(2024, 1, 1, 2),
                    D(2024, 1, 1, 5),
                ],
                pa.timestamp("us"),
            ),
            "v": pa.array([1.0, 4.0, 10.0, 20.0, 50.0]),
        }
    )


@pytest.mark.parametrize("fill", [None, "forward", "backward"])
def test_grid_matches_polars_upsample(fill: str | None) -> None:
    out = bt.from_arrow(_readings()).upsample("t", "1h", by="g", fill=fill).collect()
    frame = pl.from_arrow(_readings()).sort("g", "t").upsample("t", every="1h", group_by="g")
    frame = frame.with_columns(pl.col("g").forward_fill())
    if fill is not None:
        frame = frame.with_columns(pl.col("v").fill_null(strategy=fill).over("g"))
    assert out.column_names == ["g", "t", "v"]
    assert _rows(out, ["g", "t"]) == _rows(frame.to_arrow(), ["g", "t"])


def test_indicator_marks_only_inserted_rows() -> None:
    out = bt.from_arrow(_readings()).upsample("t", "1h", by="g", fill="forward", indicator="ins")
    got = {(r["g"], r["t"].hour): (r["v"], r["ins"]) for r in out.collect().to_pylist()}
    assert got[("a", 0)] == (1.0, False)
    assert got[("a", 1)] == (1.0, True)
    assert got[("b", 3)] == (20.0, True)
    assert got[("b", 5)] == (50.0, False)
    assert sum(ins for _, ins in got.values()) == 4


def test_off_grid_null_time_and_null_key_rows_are_kept() -> None:
    """Rows Polars would drop or reject stay, and a null `by` key is a group of its own."""
    table = pa.table(
        {
            "g": ["a", "a", "a", None, None, "c"],
            "t": [
                D(2024, 1, 1, 0),
                D(2024, 1, 1, 1, 30),
                D(2024, 1, 1, 2),
                D(2024, 1, 1, 0),
                D(2024, 1, 1, 1),
                None,
            ],
            "v": [1, 2, 3, 4, 5, 6],
        }
    )
    out = bt.from_arrow(table).upsample("t", "1h", by="g", indicator="ins").collect()
    expected = pa.table(
        {
            "g": ["a", "a", "a", "a", None, None, "c"],
            "t": [
                D(2024, 1, 1, 0),
                D(2024, 1, 1, 1),
                D(2024, 1, 1, 1, 30),
                D(2024, 1, 1, 2),
                D(2024, 1, 1, 0),
                D(2024, 1, 1, 1),
                None,
            ],
            "v": [1, None, 2, 3, 4, 5, 6],
            "ins": [False, True, False, False, False, False, False],
        }
    )
    assert _rows(out, ["g", "t"]) == _rows(expected, ["g", "t"])


def test_duplicate_times_are_kept_and_not_reinserted() -> None:
    table = pa.table({"t": [D(2024, 1, 1, 0), D(2024, 1, 1, 0), D(2024, 1, 1, 2)], "v": [1, 2, 3]})
    out = bt.from_arrow(table).upsample("t", "1h", indicator="ins").collect()
    assert sorted(out.to_pydict()["v"], key=lambda v: (v is None, v)) == [1, 2, 3, None]


@pytest.mark.parametrize(
    ("ttype", "values", "every", "expected"),
    [
        (pa.date32(), [dt.date(2024, 1, 1), dt.date(2024, 1, 4)], "1d", 4),
        (pa.timestamp("ns", tz="UTC"), [0, 3_000_000_000], "1s", 4),
        (pa.timestamp("ms"), [0, 90_000], "30s", 4),
        (pa.timestamp("s"), [0, 7200], "1h", 3),
    ],
)
def test_the_column_type_and_unit_are_preserved(ttype, values, every, expected) -> None:
    table = pa.table({"t": pa.array(values, ttype), "v": pa.array([1, 2])})
    out = bt.from_arrow(table).upsample("t", every).collect()
    assert out.schema.field("t").type == ttype
    assert out.num_rows == expected


def test_an_empty_input_upsamples_to_nothing() -> None:
    ds = bt.from_arrow(_readings()).filter(bt.col("v") > 1_000)
    assert ds.upsample("t", "1h", by="g").collect().num_rows == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"time_col": "v", "every": "1h"}, "date or timestamp"),
        ({"time_col": "t", "every": "1mo"}, "calendar unit"),
        ({"time_col": "t", "every": "1h", "fill": "linear"}, "fill"),
        ({"time_col": "t", "every": "1h", "indicator": "v"}, "existing column"),
        ({"time_col": "t", "every": "1h", "by": "t"}, "cannot also be"),
        ({"time_col": "nope", "every": "1h"}, "unknown column"),
    ],
)
def test_bad_arguments_are_plan_errors(kwargs: dict, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        bt.from_arrow(_readings()).upsample(**kwargs)


def test_a_step_finer_than_a_date_is_refused() -> None:
    table = pa.table({"d": pa.array([dt.date(2024, 1, 1)]), "v": [1]})
    with pytest.raises(PlanError, match="whole number"):
        bt.from_arrow(table).upsample("d", "12h")
