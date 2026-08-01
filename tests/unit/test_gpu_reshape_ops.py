"""The operators and temporal expressions the translator used to decline, against the engine.

`unnest`, `unpivot`, `row_id`, `offset_by`, `window_start` and a computed window key each used
to send a whole plan to the CPU engine. For `unnest` that meant every document or array
pipeline — the shapes with the most rows to move and the most to gain from a device.

The cases here are chosen around the two places the dataframe libraries and the engine disagree
rather than around the happy path, because the happy path was never the risk:

* `explode` implements the **outer** form on both libraries, so the default SQL semantics need
  the invented rows filtered back out — and a row invented for an empty list has to be told
  apart from a row carrying a list's own null element, which by then looks identical;
* the element index is 0-based *within its list*, and a row that has no element has no position
  either, where a zero would read as "the first element".
"""

from __future__ import annotations

import contextlib
import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.core.gpu_plan import DfBackend, gpu_plan_ops
from batcher.core.gpu_plan.execute import run_chain

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def be():
    import pandas as pd

    return DfBackend(pd)


def _lists() -> pa.Table:
    """A two-element list, an empty one, a null one, and one carrying its own null element.

    The last two are the pair the whole operator turns on: after an explode they are the same
    row, and they need opposite answers.
    """
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4], pa.int64()),
            "xs": pa.array([[10, 20], [], None, [30, None]], pa.list_(pa.int64())),
            "tag": pa.array(["a", "b", "c", "d"], pa.string()),
        }
    )


def _wide() -> pa.Table:
    return pa.table(
        {
            "k": pa.array(["x", "y"], pa.string()),
            "a": pa.array([1, None], pa.int64()),
            "b": pa.array([3, 4], pa.int64()),
            "ignored": pa.array([9, 9], pa.int64()),
        }
    )


def _translated(ds, table: pa.Table, be) -> pa.Table:
    spec = gpu_plan_ops(ds._plan)
    assert spec is not None, "the translator declined a plan it is supposed to match"
    return be.to_arrow(run_chain(table, spec[1], be))


def _assert_matches_engine(ds, table: pa.Table, be, *, ordered: bool = True) -> None:
    expected = ds.collect()
    got = _translated(ds, table, be).select(expected.column_names)
    if ordered:
        assert got.to_pydict() == expected.to_pydict()
    else:
        key = repr
        rows = lambda t: sorted(zip(*t.to_pydict().values(), strict=True), key=key)  # noqa: E731
        assert rows(got) == rows(expected)
    assert got.schema.types == expected.schema.types


# --- unnest ---------------------------------------------------------------------------------


def test_unnest_drops_the_rows_with_no_elements_by_default(be):
    """SQL's `UNNEST` contributes nothing for an empty or null list, where `explode` invents a
    row. Getting this backwards is invisible row loss, not an error."""
    table = _lists()
    _assert_matches_engine(bt.from_arrow(table).explode("xs"), table, be)


def test_unnest_keeps_a_lists_own_null_element(be):
    """The element of `[30, None]` that is null is a row the engine keeps.

    Filtering the invented rows by "the element is null" would drop it, which is why the
    emptiness marker is taken before the explode rather than inferred after it. This is the
    case that caught the first implementation.
    """
    table = _lists()
    got = _translated(bt.from_arrow(table).explode("xs"), table, be)
    assert got.to_pydict()["xs"] == [10, 20, 30, None]


def test_unnest_outer_keeps_the_rows_with_no_elements(be):
    table = _lists()
    _assert_matches_engine(bt.from_arrow(table).explode("xs", outer=True), table, be)


def test_unnest_numbers_each_element_within_its_own_list(be):
    table = _lists()
    ds = bt.from_arrow(table).explode("xs", alias="x", index="pos")
    _assert_matches_engine(ds, table, be)
    assert _translated(ds, table, be).to_pydict()["pos"] == [0, 1, 0, 1]


def test_unnest_gives_an_outer_row_no_position(be):
    """A row kept only by `outer` has no element, so it has no position either — null, not 0,
    which would read as the first element of a list that has none."""
    table = _lists()
    ds = bt.from_arrow(table).explode("xs", outer=True, index="pos")
    _assert_matches_engine(ds, table, be)
    assert _translated(ds, table, be).to_pydict()["pos"] == [0, 1, None, None, 0, 1]


def test_unnest_keeps_the_exploded_column_in_place(be):
    """The element column stays where the list column was; only the index is appended."""
    table = _lists()
    ds = bt.from_arrow(table).explode("xs", alias="x", index="pos")
    assert _translated(ds, table, be).column_names == ["id", "x", "tag", "pos"]


def test_unnest_composes_with_the_rest_of_the_chain(be):
    table = _lists()
    ds = bt.from_arrow(table).explode("xs").filter(col("xs") > 15).group_by("tag").agg(n=bt.count())
    _assert_matches_engine(ds, table, be, ordered=False)


# --- unpivot --------------------------------------------------------------------------------


def test_unpivot_matches_the_engine(be):
    table = _wide()
    ds = bt.from_arrow(table).unpivot(index=["k"], on=["a", "b"])
    _assert_matches_engine(ds, table, be, ordered=False)


def test_unpivot_honors_the_output_names(be):
    table = _wide()
    ds = bt.from_arrow(table).unpivot(index=["k"], on=["a", "b"], variable_name="m", value_name="v")
    _assert_matches_engine(ds, table, be, ordered=False)


def test_unpivot_without_an_index_column(be):
    table = _wide()
    ds = bt.from_arrow(table).unpivot(index=[], on=["a", "b"])
    _assert_matches_engine(ds, table, be, ordered=False)


def test_unpivot_drops_the_columns_it_carries_neither_way(be):
    """`ignored` is in neither `index` nor `on`, and the operator's contract is that it goes."""
    table = _wide()
    ds = bt.from_arrow(table).unpivot(index=["k"], on=["a", "b"])
    assert "ignored" not in _translated(ds, table, be).column_names


# --- offset_by ------------------------------------------------------------------------------


def _calendar_sweep() -> pa.Table:
    """Every month-end and mid-month across leap years, century turns and the epoch."""
    instants = []
    for year in (1900, 1969, 1970, 2000, 2023, 2024, 2100):
        for month in range(1, 13):
            for day in (1, 15, 28, 29, 30, 31):
                # A day the month does not have is simply not a case.
                with contextlib.suppress(ValueError):
                    instants.append(dt.datetime(year, month, day, 13, 45, 30, 123_456))
    instants.append(None)
    return pa.table({"ts": pa.array(instants, pa.timestamp("us"))})


def _instants() -> pa.Table:
    return pa.table(
        {
            "ts": pa.array(
                [dt.datetime(2024, 2, 29, 13, 45), dt.datetime(1969, 7, 20, 20, 17), None],
                pa.timestamp("us"),
            ),
            "d": pa.array([dt.date(2024, 2, 29), dt.date(1969, 7, 20), None], pa.date32()),
        }
    )


@pytest.mark.parametrize("offset", ["7d", "3h", "-1w", "90m"])
def test_offset_by_an_exact_amount_matches_the_engine(offset, be):
    table = _instants()
    ds = bt.from_arrow(table).select(r=col("ts").dt.offset_by(offset))
    _assert_matches_engine(ds, table, be)


@pytest.mark.parametrize("offset", ["7d", "-1w"])
def test_offset_by_preserves_a_date(offset, be):
    """A DATE shifted by whole days is still a DATE — a timestamp here is a column a fan-out
    could not concatenate with a shard the CPU engine produced."""
    table = _instants()
    ds = bt.from_arrow(table).select(r=col("d").dt.offset_by(offset))
    _assert_matches_engine(ds, table, be)
    assert _translated(ds, table, be).schema.field("r").type == pa.date32()


@pytest.mark.parametrize("offset", ["1mo", "-1mo", "13mo", "1y", "-3y", "-25mo", "1mo3d"])
def test_offset_by_a_calendar_month_matches_the_engine(offset, be):
    """A month is a construction from (year, month, day), not a distance, so it is built with
    `days_from_civil` and has to agree with chrono across the whole calendar."""
    table = _calendar_sweep()
    ds = bt.from_arrow(table).select(r=col("ts").dt.offset_by(offset))
    _assert_matches_engine(ds, table, be)


def test_offset_by_a_month_clamps_to_the_end_of_the_target_month(be):
    """January 31 plus one month is the end of February, never March 3.

    Clamping is the rule every calendar library follows and the easiest thing here to get
    wrong, because the naive construction — add one to the month and keep the day — produces a
    date that looks reasonable and is off by two or three days for one month in twelve.
    """
    table = pa.table(
        {
            "ts": pa.array(
                [dt.datetime(2024, 1, 31), dt.datetime(2023, 1, 31), dt.datetime(2024, 3, 31)],
                pa.timestamp("us"),
            )
        }
    )
    ds = bt.from_arrow(table).select(r=col("ts").dt.offset_by("1mo"))
    _assert_matches_engine(ds, table, be)
    assert _translated(ds, table, be).to_pydict()["r"] == [
        dt.datetime(2024, 2, 29),  # leap year
        dt.datetime(2023, 2, 28),  # common year
        dt.datetime(2024, 4, 30),  # a 31-day month into a 30-day one
    ]


def test_offset_by_a_month_keeps_the_time_of_day(be):
    """A month shift moves the date and nothing else."""
    table = _instants()
    ds = bt.from_arrow(table).select(r=col("ts").dt.offset_by("1mo"))
    _assert_matches_engine(ds, table, be)


def test_offset_by_a_sub_day_amount_on_a_date_is_declined(be):
    """The engine errors rather than inventing a time of day, so the fallback has to reach it."""
    from batcher.core.gpu_plan.backend import Unsupported

    table = _instants()
    ds = bt.from_arrow(table).select(r=col("d").dt.offset_by("3h"))
    with pytest.raises(Unsupported):
        _translated(ds, table, be)


# --- window_start ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", ["30s", "15m", "1h", "1d"])
def test_window_start_matches_the_engine(width, be):
    """The tumbling-window bucket key a streaming aggregate groups by.

    The pre-epoch instant is the case worth having: the offset from the origin is negative
    there, and flooring puts it in the window that contains it where truncating would put it
    in the one after.
    """
    table = _instants()
    ds = bt.from_arrow(table).select(r=bt.window(col("ts"), width))
    _assert_matches_engine(ds, table, be)


# --- the epoch constructors -------------------------------------------------------------------


def _epoch_counts() -> pa.Table:
    """Ordinary epoch values, a null, and one far past what an instant can hold.

    The last is the case the guard exists for: the engine scales *checked* and reports null,
    where an unchecked multiply gives a plausible date in the wrong millennium.
    """
    return pa.table({"n": pa.array([1_700_000_000, 0, -86_400, None, 2**62], pa.int64())})


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_from_epoch_matches_the_engine(unit, be):
    """Compared on Arrow rather than through Python, because a valid instant here can sit
    outside `datetime`'s range — the conversion would fail on both sides and prove nothing."""
    table = _epoch_counts()
    ds = bt.from_arrow(table).select(r=bt.from_epoch(col("n"), unit))
    expected = ds.collect()
    assert _translated(ds, table, be).select(expected.column_names).equals(expected)


def test_from_epoch_nulls_a_count_too_large_to_be_an_instant(be):
    """Arrow's multiply is checked and evaluates the slots under a null mask too, so the
    out-of-range rows are neutralized *before* the scaling rather than masked after it.
    Masking first still raises on the value it was told to ignore, which on a device is a
    silent fallback of the whole query rather than one null row."""
    table = _epoch_counts()
    ds = bt.from_arrow(table).select(r=bt.from_epoch(col("n"), "s"))
    assert _translated(ds, table, be).to_pydict()["r"][-1] is None


def test_from_unix_date_matches_the_engine(be):
    """A day count is already a DATE's own representation, so it narrows rather than scales —
    the widest Date32 is a year in the millions, which no microsecond count could hold."""
    table = pa.table({"n": pa.array([19_000, 0, -1, None, 2**40], pa.int64())})
    ds = bt.from_arrow(table).select(r=bt.from_unix_date(col("n")))
    expected = ds.collect()
    assert _translated(ds, table, be).select(expected.column_names).equals(expected)


def test_make_date_is_declined(be):
    """`make_date` validates, and `days_from_civil` is a mapping rather than a validator.

    February 30 is null in the engine; a day count computed for it would be March 1 or 2 —
    a real date, silently, where the engine reports that there is none.
    """
    from batcher.core.gpu_plan.backend import Unsupported

    table = pa.table({"y": pa.array([2024], pa.int64()), "m": pa.array([2], pa.int64())})
    ds = bt.from_arrow(table).select(r=bt.make_date(col("y"), col("m"), bt.lit(30)))
    with pytest.raises(Unsupported):
        _translated(ds, table, be)


# --- row_id ---------------------------------------------------------------------------------


def test_row_id_matches_the_engine(be):
    table = _lists()
    _assert_matches_engine(bt.from_arrow(table).with_row_index("n"), table, be)


def test_row_id_honors_an_offset(be):
    table = _lists()
    _assert_matches_engine(bt.from_arrow(table).with_row_index("n", offset=10), table, be)


def test_row_id_prepends_its_column(be):
    """Polars' `with_row_index` puts the index first, and so does the plan's own
    `available_columns` — appending it would put every other column in the wrong place, which
    a comparison by column name would never notice."""
    table = _lists()
    ds = bt.from_arrow(table).with_row_index("n")
    assert _translated(ds, table, be).column_names[0] == "n"


# --- the list length ------------------------------------------------------------------------


def test_list_len_matches_the_engine(be):
    """The one list reduction both libraries spell the same way."""
    table = _lists()
    ds = bt.from_arrow(table).select(r=col("xs").list.len())
    _assert_matches_engine(ds, table, be)


def test_a_list_reduction_runs_through_the_element_view(be):
    """`sum` over a list is on cuDF's accessor and not on the host backend's, so it is built
    from `explode` plus `groupby` instead — the two primitives both libraries do have.

    The vocabulary and its edge cases live in `test_gpu_list_vocabulary`; this is the reshape
    module's own check that the construction composes with the operators around it.
    """
    table = _lists()
    ds = bt.from_arrow(table).select(r=col("xs").list.sum())
    _assert_matches_engine(ds, table, be)


def test_a_list_to_list_function_is_still_declined(be):
    """Reassembling a list result is the one thing the element view cannot do without
    materializing a Python object per row, which is a hot-path tuple touch."""
    from batcher.core.gpu_plan.backend import Unsupported

    table = _lists()
    ds = bt.from_arrow(table).select(r=col("xs").list.sort())
    with pytest.raises(Unsupported):
        _translated(ds, table, be)


# --- computed window keys -------------------------------------------------------------------


def _series() -> pa.Table:
    return pa.table(
        {
            "ts": pa.array(
                [
                    dt.datetime(2024, 1, 5),
                    dt.datetime(2024, 1, 20),
                    dt.datetime(2024, 2, 3),
                    dt.datetime(2024, 2, 20),
                    dt.datetime(2024, 3, 1),
                ],
                pa.timestamp("us"),
            ),
            "g": pa.array(["a", "a", "b", "b", "a"], pa.string()),
            "v": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], pa.float64()),
        }
    )


def test_a_computed_partition_key_runs_on_the_device(be):
    """`PARTITION BY date_trunc('month', ts)` is an ordinary way to write a monthly ranking.

    The translator used to require every partition key to be a plain column, so the shape of
    one key dropped the whole chain to the CPU engine — and it does the opposite for a group
    key, which it has always materialized.
    """
    table = _series()
    ds = bt.from_arrow(table).with_columns(r=col("v").sum().over(col("ts").dt.truncate("month")))
    _assert_matches_engine(ds, table, be)


def test_a_computed_order_key_runs_on_the_device(be):
    table = _series()
    ds = bt.from_arrow(table).with_columns(
        r=col("v").cum_sum().over("g", order_by=col("ts").dt.truncate("month"))
    )
    _assert_matches_engine(ds, table, be)


def test_a_materialized_window_key_is_not_an_output_column(be):
    """The window operator adds one column per function and nothing else, so the private
    column a computed key is evaluated into has to be dropped again."""
    table = _series()
    ds = bt.from_arrow(table).with_columns(r=col("v").sum().over(col("ts").dt.truncate("month")))
    assert _translated(ds, table, be).column_names == ["ts", "g", "v", "r"]


# --- str.reverse ----------------------------------------------------------------------------


def test_str_reverse_matches_the_engine(be):
    """Neither library exposes a `.str.reverse` the other also has; a negative-step slice is
    the spelling they share."""
    table = pa.table({"s": pa.array(["abc", "de", None, "x/y/z", ""], pa.string())})
    ds = bt.from_arrow(table).select(r=col("s").str.reverse())
    _assert_matches_engine(ds, table, be)
