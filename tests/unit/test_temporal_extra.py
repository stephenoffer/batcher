"""Plan-shape unit tests for the `temporal_extra` rules.

Two halves. The first asserts each rule *fires* and produces the intended shape (a
`date_trunc`/`strftime` comparison becomes a bare-column range; a nested truncation
collapses; a temporal literal folds). The second — the one that matters — asserts the
rules **do not fire** on the shapes where a rewrite would be a wrong answer: a
non-monotone extraction (`month`/`day`/`quarter` recur every year, so they are not a
contiguous range), a timezone conversion (DST nulls out ambiguous local times), an
unaligned literal, a non-fixed-width format, and a calendar-month offset.
"""

from __future__ import annotations

import datetime as dt
import json

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.temporal_extra import (
    DATE_TRUNC_RANGE_RULES,
    STRFTIME_RANGE_RULES,
    rewrite_date_cast_filter,
    rewrite_date_trunc_filter,
    rewrite_strftime_filter,
)
from batcher.plan.logical import Filter

_EPOCH = dt.date(1970, 1, 1)


def _days(year: int, month: int, day: int) -> int:
    return (dt.date(year, month, day) - _EPOCH).days


def _dates() -> bt.Dataset:
    return bt.from_pydict(
        {"d": [dt.date(2021, 3, 5), dt.date(2024, 2, 29)], "v": [1, 2]},
    )


def _ts() -> bt.Dataset:
    return bt.from_pydict(
        {"t": [dt.datetime(2021, 3, 5, 4, 5), dt.datetime(2022, 1, 1)], "v": [1, 2]},
    )


def _rewritten(ds: bt.Dataset):
    return Optimizer(sources=ds._sources).logical_rewrite(ds._plan)


def _pred(node: Filter) -> dict:
    return node.predicate.to_ir()


# --- registration ---------------------------------------------------------------


def test_fifteen_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    expected = {f"date_trunc_{op}_to_range" for op in ("lt", "le", "gt", "ge")}
    expected |= {f"strftime_{op}_to_range" for op in ("eq", "lt", "le", "gt", "ge")}
    expected |= {
        "date_trunc_idempotent",
        "date_trunc_nested_to_coarser",
        "drop_date_to_timestamp_cast_in_comparison",
        "fold_date_func_of_literal",
        "fold_date_offset_of_literal",
        "fold_temporal_literal_comparison",
    }
    assert expected <= names
    assert len(expected) == 15
    assert len(DATE_TRUNC_RANGE_RULES) == 4
    assert len(STRFTIME_RANGE_RULES) == 5


# --- date_trunc inequalities → bounds --------------------------------------------


def test_date_trunc_ge_becomes_lower_bound():
    node = _ts().filter(col("t").dt.truncate("month") >= lit(dt.datetime(2021, 3, 1)))._plan
    p = _pred(rewrite_date_trunc_filter(node, "ge"))
    assert p["op"] == "ge" and p["left"]["name"] == "t"
    assert "date_trunc" not in json.dumps(p)


def test_date_trunc_le_uses_next_unit():
    # trunc(month, t) <= 2021-03-01  ⟺  t < 2021-04-01
    node = _ts().filter(col("t").dt.truncate("month") <= lit(dt.datetime(2021, 3, 1)))._plan
    p = _pred(rewrite_date_trunc_filter(node, "le"))
    assert p["op"] == "lt"
    assert p["right"]["value"]["timestamp"] == _days(2021, 4, 1) * 86_400_000_000


def test_date_trunc_gt_uses_next_unit():
    node = _ts().filter(col("t").dt.truncate("year") > lit(dt.datetime(2021, 1, 1)))._plan
    p = _pred(rewrite_date_trunc_filter(node, "gt"))
    assert p["op"] == "ge"
    assert p["right"]["value"]["timestamp"] == _days(2022, 1, 1) * 86_400_000_000


def test_date_trunc_lt_keeps_bound():
    node = _ts().filter(col("t").dt.truncate("day") < lit(dt.datetime(2021, 3, 5)))._plan
    p = _pred(rewrite_date_trunc_filter(node, "lt"))
    assert p["op"] == "lt"
    assert p["right"]["value"]["timestamp"] == _days(2021, 3, 5) * 86_400_000_000


def test_date_trunc_literal_on_left_is_mirrored():
    # `L < trunc(t)` is `trunc(t) > L` → the `gt` rule, not the `lt` one.
    node = _ts().filter(lit(dt.datetime(2021, 1, 1)) < col("t").dt.truncate("year"))._plan
    assert rewrite_date_trunc_filter(node, "lt") is None
    p = _pred(rewrite_date_trunc_filter(node, "gt"))
    assert p["op"] == "ge"


def test_date_trunc_unaligned_literal_does_not_fire():
    # 2021-03-05 is not a month boundary — the truncation cannot equal it, and no plain
    # bound on `t` reproduces the comparison. Leave it to the engine.
    node = _ts().filter(col("t").dt.truncate("month") >= lit(dt.datetime(2021, 3, 5)))._plan
    assert rewrite_date_trunc_filter(node, "ge") is None


def test_date_trunc_subday_unit_on_date_literal_does_not_fire():
    # `date + timedelta(hours=1)` silently drops the hour, so the upper bound would
    # collapse onto the lower one. The rule must refuse.
    node = _dates().filter(col("d").dt.truncate("hour") <= lit(dt.date(2021, 3, 5)))._plan
    assert rewrite_date_trunc_filter(node, "le") is None


def test_date_trunc_eq_is_left_to_the_existing_rule():
    node = _ts().filter(col("t").dt.truncate("month") == lit(dt.datetime(2021, 3, 1)))._plan
    for op in ("lt", "le", "gt", "ge"):
        assert rewrite_date_trunc_filter(node, op) is None


# --- strftime comparisons → ranges -----------------------------------------------


def test_strftime_year_month_eq_becomes_half_open_range():
    node = _dates().filter(col("d").dt.strftime("%Y-%m") == lit("2021-03"))._plan
    p = _pred(rewrite_strftime_filter(node, "eq"))
    assert p["op"] == "and"
    assert p["left"]["op"] == "ge" and p["left"]["right"]["value"]["date"] == _days(2021, 3, 1)
    assert p["right"]["op"] == "lt" and p["right"]["right"]["value"]["date"] == _days(2021, 4, 1)
    assert "strftime" not in json.dumps(p)


def test_strftime_full_date_eq_is_one_day():
    node = _dates().filter(col("d").dt.strftime("%Y-%m-%d") == lit("2024-02-29"))._plan
    p = _pred(rewrite_strftime_filter(node, "eq"))
    assert p["left"]["right"]["value"]["date"] == _days(2024, 2, 29)
    assert p["right"]["right"]["value"]["date"] == _days(2024, 3, 1)


def test_strftime_year_inequalities_are_single_bounds():
    ds = _dates()
    node = ds.filter(col("d").dt.strftime("%Y") >= lit("2022"))._plan
    p = _pred(rewrite_strftime_filter(node, "ge"))
    assert p["op"] == "ge" and p["right"]["value"]["date"] == _days(2022, 1, 1)

    node = ds.filter(col("d").dt.strftime("%Y") <= lit("2022"))._plan
    p = _pred(rewrite_strftime_filter(node, "le"))
    assert p["op"] == "lt" and p["right"]["value"]["date"] == _days(2023, 1, 1)

    node = ds.filter(col("d").dt.strftime("%Y") > lit("2022"))._plan
    p = _pred(rewrite_strftime_filter(node, "gt"))
    assert p["op"] == "ge" and p["right"]["value"]["date"] == _days(2023, 1, 1)

    node = ds.filter(col("d").dt.strftime("%Y") < lit("2022"))._plan
    p = _pred(rewrite_strftime_filter(node, "lt"))
    assert p["op"] == "lt" and p["right"]["value"]["date"] == _days(2022, 1, 1)


def test_strftime_timestamp_column_emits_timestamp_literals():
    node = _ts().filter(col("t").dt.strftime("%Y-%m-%d") == lit("2021-03-05"))._plan
    p = _pred(rewrite_strftime_filter(node, "eq"))
    assert "timestamp" in p["left"]["right"]["value"]


def test_strftime_non_fixed_width_format_does_not_fire():
    # `%d-%m-%Y` is day-first: its string order is not chronological order.
    node = _dates().filter(col("d").dt.strftime("%d-%m-%Y") == lit("05-03-2021"))._plan
    assert rewrite_strftime_filter(node, "eq") is None


def test_strftime_malformed_literal_does_not_fire():
    # chrono zero-pads; `'2021-3'` is not a string any row can produce.
    node = _dates().filter(col("d").dt.strftime("%Y-%m") == lit("2021-3"))._plan
    assert rewrite_strftime_filter(node, "eq") is None


def test_strftime_impossible_date_does_not_fire():
    node = _dates().filter(col("d").dt.strftime("%Y-%m-%d") == lit("2021-02-30"))._plan
    assert rewrite_strftime_filter(node, "eq") is None


def test_strftime_on_non_temporal_column_does_not_fire():
    node = bt.from_pydict({"v": [1, 2]}).filter(col("v").dt.strftime("%Y") == lit("2021"))._plan
    assert rewrite_strftime_filter(node, "eq") is None


# --- the non-monotone extractions must NEVER become a range ----------------------


def test_month_predicate_is_not_range_sargable():
    # `month(d) = 3` matches March of *every* year in the data: a union of intervals,
    # never one contiguous range. No rule may touch it.
    ds = _dates().filter(col("d").dt.month() == 3)
    ir = json.dumps(_rewritten(ds).to_ir())
    assert '"month"' in ir


def test_day_and_quarter_and_week_predicates_are_not_range_sargable():
    for expr in (
        col("d").dt.day() == 5,
        col("d").dt.quarter() == 1,
        col("d").dt.week() == 9,
        col("d").dt.dayofweek() == 5,
    ):
        ir = json.dumps(_rewritten(_dates().filter(expr)).to_ir())
        assert '"date"' in ir  # the DateFunc node tag survives — no range was invented
        assert '"and"' not in ir  # …and no half-open range was fabricated


def test_convert_timezone_is_never_dropped():
    # Even a same-zone conversion is not the identity: the engine nulls out DST-ambiguous
    # and nonexistent local times. Dropping it would resurrect those rows' values.
    ds = _ts().filter(col("t").dt.convert_timezone("UTC", "UTC") > lit(dt.datetime(2021, 1, 1)))
    ir = json.dumps(_rewritten(ds).to_ir())
    assert "convert_timezone" in ir


# --- date_trunc collapsing -------------------------------------------------------


def test_date_trunc_idempotent_collapses():
    ds = _ts().select(col("t").dt.truncate("day").dt.truncate("day").alias("r"))
    ir = json.dumps(_rewritten(ds).to_ir())
    assert ir.count("date_trunc") == 1
    assert '"unit": "day"' in ir


def test_date_trunc_nested_keeps_the_coarser_unit():
    # Both nesting orders collapse to the coarser unit: the year floor absorbs the day
    # floor, and the day floor of a year start is that same year start.
    for outer, inner, kept in (("year", "day", "year"), ("day", "year", "year")):
        ds = _ts().select(col("t").dt.truncate(inner).dt.truncate(outer).alias("r"))
        ir = json.dumps(_rewritten(ds).to_ir())
        assert ir.count("date_trunc") == 1
        assert f'"unit": "{kept}"' in ir


# --- constant folding ------------------------------------------------------------


def test_fold_date_func_of_literal():
    ds = _dates().select(lit(dt.date(2021, 3, 5)).dt.quarter().alias("q"))
    assert _rewritten(ds).items[0].expr.to_ir() == {"e": "lit", "value": {"int": 1}}


def test_fold_date_func_leaves_the_engine_specific_ones():
    # `dayname` is chrono `%A` in the engine and locale-dependent in Python: never folded.
    ds = _dates().select(lit(dt.date(2021, 3, 5)).dt.dayname().alias("n"))
    assert "dayname" in json.dumps(_rewritten(ds).to_ir())


def test_fold_date_offset_of_literal():
    ds = _dates().select(lit(dt.date(2021, 3, 5)).dt.offset_by("3d").alias("r"))
    assert _rewritten(ds).items[0].expr.to_ir()["value"]["date"] == _days(2021, 3, 8)


def test_fold_date_offset_leaves_calendar_months():
    # Month arithmetic clamps to the end of the target month — the engine's rule, not
    # one to re-derive at plan time.
    ds = _dates().select(lit(dt.date(2021, 1, 31)).dt.offset_by("1mo").alias("r"))
    assert "date_offset" in json.dumps(_rewritten(ds).to_ir())


def test_fold_temporal_literal_comparison():
    ds = _dates().filter(lit(dt.date(2021, 3, 5)) < lit(dt.date(2021, 3, 6)))
    ir = json.dumps(_rewritten(ds).to_ir())
    assert "filter" not in ir  # a TRUE predicate is then pruned by `prune_true_filter`


def test_fold_temporal_comparison_refuses_mixed_kinds():
    # date vs datetime: the engine rebases one onto the other; don't guess how.
    ds = (
        _dates()
        .filter(col("v") > lit(1))
        .filter(lit(dt.date(2021, 3, 5)) < lit(dt.datetime(2021, 3, 5, 1)))
    )
    ir = json.dumps(_rewritten(ds).to_ir())
    assert '"date"' in ir and '"timestamp"' in ir


# --- date → timestamp cast in a comparison ---------------------------------------


def test_date_to_timestamp_cast_is_dropped():
    node = _dates().filter(col("d").cast("timestamp") >= lit(dt.datetime(2021, 3, 5)))._plan
    p = _pred(rewrite_date_cast_filter(node))
    assert p == {
        "e": "binary",
        "op": "ge",
        "left": {"e": "col", "name": "d"},
        "right": {"e": "lit", "value": {"date": _days(2021, 3, 5)}},
    }


def test_date_cast_literal_on_left_is_dropped_without_mirroring():
    node = _dates().filter(lit(dt.datetime(2021, 3, 5)) <= col("d").cast("timestamp"))._plan
    p = _pred(rewrite_date_cast_filter(node))
    assert p["op"] == "le"
    assert p["left"]["value"]["date"] == _days(2021, 3, 5)
    assert p["right"]["name"] == "d"


def test_date_cast_with_non_midnight_literal_does_not_fire():
    node = _dates().filter(col("d").cast("timestamp") >= lit(dt.datetime(2021, 3, 5, 12)))._plan
    assert rewrite_date_cast_filter(node) is None


def test_date_cast_of_timestamp_column_does_not_fire():
    # The cast is a no-op there (that is `drop_self_cast_in_filter`'s job), and the
    # date-midnight isomorphism this rule relies on does not apply.
    node = _ts().filter(col("t").cast("timestamp") >= lit(dt.datetime(2021, 3, 5)))._plan
    assert rewrite_date_cast_filter(node) is None


# --- idempotence / termination through the real optimizer ------------------------


def test_rules_are_idempotent():
    node = _ts().filter(col("t").dt.truncate("month") >= lit(dt.datetime(2021, 3, 1)))._plan
    once = rewrite_date_trunc_filter(node, "ge")
    assert rewrite_date_trunc_filter(once, "ge") is None

    node = _dates().filter(col("d").dt.strftime("%Y") == lit("2021"))._plan
    once = rewrite_strftime_filter(node, "eq")
    assert rewrite_strftime_filter(once, "eq") is None


def test_full_optimizer_terminates_and_drops_the_extraction():
    ds = _dates().filter(
        (col("d").dt.strftime("%Y-%m") == lit("2021-03"))
        & (col("d").cast("timestamp") > lit(dt.datetime(2020, 1, 1)))
    )
    ir = json.dumps(_rewritten(ds).to_ir())
    assert "strftime" not in ir and "cast" not in ir
