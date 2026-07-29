"""Plan-shape unit tests for the `temporal_sargable` rules.

`year(col)`/`decade(col)` comparisons rewrite to half-open ranges on the raw column;
these tests assert each rule fires and yields the intended shape, is idempotent, and
does *not* fire on the non-contiguous extractions (month/quarter/…) or on ambiguous
inputs (non-integer literal, non-temporal column).
"""

from __future__ import annotations

import datetime as dt
import json

import batcher as bt
from batcher import col, lit
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.temporal_sargable import (
    TEMPORAL_SARGABLE_RULES,
    rewrite_temporal_filter,
)
from batcher.plan.logical import Filter


def _dates():
    return bt.from_pydict(
        {"d": [dt.date(2020, 6, 1), dt.date(2021, 1, 1), dt.date(2022, 3, 3)], "v": [1, 2, 3]}
    )


def _ts():
    return bt.from_pydict(
        {"t": [dt.datetime(2021, 1, 1, 12, 0), dt.datetime(2022, 6, 1, 0, 0)], "v": [1, 2]}
    )


def _pred(node):
    return node.predicate.to_ir()


# --- registration --------------------------------------------------------------


def test_one_rule_per_bucket_and_operator_is_registered():
    # Four contiguous calendar-year buckets against six comparisons. `century` and
    # `millennium` are 1-based (century 20 is 1901-2000); `iso_year` is absent because its
    # boundaries are Mondays rather than 1 January, so it is not calendar-year aligned.
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    buckets = ("year", "decade", "century", "millennium")
    operators = ("eq", "ne", "lt", "le", "gt", "ge")
    expected = {f"{fn}_{op}_to_range" for fn in buckets for op in operators}
    assert expected <= names
    assert len(TEMPORAL_SARGABLE_RULES) == len(buckets) * len(operators)


# --- year: one shape assertion per operator ------------------------------------


def test_year_eq_becomes_half_open_range():
    node = _dates().filter(col("d").dt.year() == 2021)._plan
    out = rewrite_temporal_filter(node, "year", "eq")
    assert isinstance(out, Filter)
    p = _pred(out)
    assert p["op"] == "and"
    assert p["left"]["op"] == "ge" and p["left"]["left"]["name"] == "d"
    assert p["left"]["right"]["value"]["date"] == (dt.date(2021, 1, 1) - dt.date(1970, 1, 1)).days
    assert p["right"]["op"] == "lt"
    assert p["right"]["right"]["value"]["date"] == (dt.date(2022, 1, 1) - dt.date(1970, 1, 1)).days
    assert "year" not in json.dumps(p)


def test_year_lt_becomes_single_lower_bound():
    node = _dates().filter(col("d").dt.year() < 2021)._plan
    p = _pred(rewrite_temporal_filter(node, "year", "lt"))
    assert p["op"] == "lt"
    assert p["right"]["value"]["date"] == (dt.date(2021, 1, 1) - dt.date(1970, 1, 1)).days


def test_year_le_uses_next_year_start():
    node = _dates().filter(col("d").dt.year() <= 2021)._plan
    p = _pred(rewrite_temporal_filter(node, "year", "le"))
    assert p["op"] == "lt"
    assert p["right"]["value"]["date"] == (dt.date(2022, 1, 1) - dt.date(1970, 1, 1)).days


def test_year_gt_uses_next_year_start():
    node = _dates().filter(col("d").dt.year() > 2021)._plan
    p = _pred(rewrite_temporal_filter(node, "year", "gt"))
    assert p["op"] == "ge"
    assert p["right"]["value"]["date"] == (dt.date(2022, 1, 1) - dt.date(1970, 1, 1)).days


def test_year_ge_uses_year_start():
    node = _dates().filter(col("d").dt.year() >= 2021)._plan
    p = _pred(rewrite_temporal_filter(node, "year", "ge"))
    assert p["op"] == "ge"
    assert p["right"]["value"]["date"] == (dt.date(2021, 1, 1) - dt.date(1970, 1, 1)).days


def test_literal_on_left_is_mirrored():
    # `2021 < year(d)` is `year(d) > 2021` → handled by the `gt` rule.
    node = _dates().filter(lit(2021) < col("d").dt.year())._plan
    assert rewrite_temporal_filter(node, "year", "lt") is None
    p = _pred(rewrite_temporal_filter(node, "year", "gt"))
    assert p["op"] == "ge"
    assert p["right"]["value"]["date"] == (dt.date(2022, 1, 1) - dt.date(1970, 1, 1)).days


# --- timestamp columns emit datetime literals ----------------------------------


def test_timestamp_column_emits_timestamp_literal():
    node = _ts().filter(col("t").dt.year() == 2021)._plan
    p = _pred(rewrite_temporal_filter(node, "year", "eq"))
    assert "timestamp" in p["left"]["right"]["value"]
    assert "timestamp" in p["right"]["right"]["value"]


# --- decade --------------------------------------------------------------------


def test_decade_eq_spans_ten_years():
    node = _dates().filter(col("d").dt.decade() == 202)._plan
    p = _pred(rewrite_temporal_filter(node, "decade", "eq"))
    assert p["left"]["right"]["value"]["date"] == (dt.date(2020, 1, 1) - dt.date(1970, 1, 1)).days
    assert p["right"]["right"]["value"]["date"] == (dt.date(2030, 1, 1) - dt.date(1970, 1, 1)).days


def test_decade_ge_uses_decade_start():
    node = _dates().filter(col("d").dt.decade() >= 202)._plan
    p = _pred(rewrite_temporal_filter(node, "decade", "ge"))
    assert p["op"] == "ge"
    assert p["right"]["value"]["date"] == (dt.date(2020, 1, 1) - dt.date(1970, 1, 1)).days


# --- idempotence ---------------------------------------------------------------


def test_idempotent_year_eq():
    node = _dates().filter(col("d").dt.year() == 2021)._plan
    once = rewrite_temporal_filter(node, "year", "eq")
    assert rewrite_temporal_filter(once, "year", "eq") is None


def test_full_optimizer_terminates_and_drops_year():
    # The rewrite has no `DateFunc` left to match, so the NORMALIZE fixpoint
    # terminates (the call returns) and the extraction is gone from the IR.
    node = _dates().filter(col("d").dt.year() == 2021)._plan
    ir = Optimizer().optimize(node).ir
    text = json.dumps(ir)
    assert "year" not in text
    assert '"ge"' in text and '"lt"' in text


# --- does-not-fire when unsafe -------------------------------------------------


def test_month_not_range_sargable():
    # month recurs every year → not a contiguous range; every year-rule leaves it.
    node = _dates().filter(col("d").dt.month() == 6)._plan
    for op in ("eq", "lt", "le", "gt", "ge"):
        assert rewrite_temporal_filter(node, "year", op) is None


def test_quarter_not_range_sargable():
    node = _dates().filter(col("d").dt.quarter() == 2)._plan
    assert rewrite_temporal_filter(node, "year", "eq") is None


def test_wrong_operator_does_not_fire():
    # The `eq` rule must ignore a `>=` comparison (and vice versa).
    node = _dates().filter(col("d").dt.year() >= 2021)._plan
    assert rewrite_temporal_filter(node, "year", "eq") is None


def test_ne_left_untouched():
    # `!=` maps to a disjunction that prunes nothing; deliberately not rewritten.
    node = _dates().filter(col("d").dt.year() != 2021)._plan
    for op in ("eq", "lt", "le", "gt", "ge"):
        assert rewrite_temporal_filter(node, "year", op) is None


def test_non_temporal_column_does_not_fire():
    # `year` over an integer column has no date/timestamp schema type → skip.
    node = bt.from_pydict({"v": [1, 2]}).filter(col("v").dt.year() == 2021)._plan
    assert rewrite_temporal_filter(node, "year", "eq") is None


def test_out_of_range_year_left_untouched():
    # `year(d) <= 9999` would need a Jan-1-10000 upper bound, which `datetime.date`
    # cannot represent → the rule must leave the predicate untouched.
    node = _dates().filter(col("d").dt.year() <= 9999)._plan
    out = rewrite_temporal_filter(node, "year", "le")
    assert out is None
