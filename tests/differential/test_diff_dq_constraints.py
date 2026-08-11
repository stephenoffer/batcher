"""The expanded `ds.dq` constraint vocabulary, checked against DuckDB.

Every constraint here lowers to a filter, so the oracle for "which rows survive `drop()`"
is the equivalent DuckDB ``WHERE``. That is the property worth pinning: a constraint whose
predicate drifts from its SQL meaning is wrong in a way no unit test on the report can see,
because the report would drift with it.
"""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher._internal.errors import DataQualityError, PlanError


def _rows():
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "qty": pa.array([3, 0, -1, None, 7], pa.int64()),
            "code": pa.array(["US", "FRA", "DE", None, ""], pa.string()),
            "note": pa.array(["ok", "N/A", "TODO fix", "fine", None], pa.string()),
            "start": pa.array([1, 5, 2, None, 9], pa.int64()),
            "end": pa.array([3, 2, 2, 4, None], pa.int64()),
        }
    )


def test_positive_matches_sql(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.positive("qty").drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE qty IS NULL OR qty > 0"))


def test_positive_non_strict_admits_zero(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.positive("qty", strict=False).drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE qty IS NULL OR qty >= 0"))


def test_rejected_values_matches_sql(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.rejected_values("note", ["N/A", "unknown"]).drop().collect()
    assert_same(
        out, duck.sql("SELECT * FROM t WHERE note IS NULL OR note NOT IN ('N/A', 'unknown')")
    )


def test_not_matches_matches_sql(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.not_matches("note", r"TODO").drop().collect()
    assert_same(
        out, duck.sql("SELECT * FROM t WHERE note IS NULL OR NOT regexp_matches(note, 'TODO')")
    )


def test_str_length_between_matches_sql(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.str_length_between("code", 2, 2).drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE code IS NULL OR length(code) = 2"))


def test_str_length_open_upper_bound(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.str_length_between("code", 2).drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE code IS NULL OR length(code) >= 2"))


def test_not_empty_matches_sql(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.not_empty("code").drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE code IS NULL OR trim(code) <> ''"))


def test_not_empty_without_strip_keeps_whitespace(duck):
    t = pa.table({"s": pa.array(["a", "", "  "], pa.string())})
    duck.register("t", t)
    out = bt.from_arrow(t).dq.not_empty("s", strip=False).drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE s <> ''"))


def test_compare_columns_matches_sql(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.compare_columns("start", "<=", "end").drop().collect()
    assert_same(
        out,
        duck.sql('SELECT * FROM t WHERE "start" IS NULL OR "end" IS NULL OR "start" <= "end"'),
    )


def test_in_range_closed_left_matches_sql(duck):
    t = _rows()
    duck.register("t", t)
    out = bt.from_arrow(t).dq.in_range("qty", 0, 7, closed="left").drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE qty IS NULL OR (qty >= 0 AND qty < 7)"))


def test_is_finite_rejects_nan_and_inf(duck):
    t = pa.table({"x": pa.array([1.0, float("nan"), float("inf"), None], pa.float64())})
    duck.register("t", t)
    out = bt.from_arrow(t).dq.is_finite("x").drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE x IS NULL OR isfinite(x)"))


def test_matches_format_email(duck):
    t = pa.table({"e": ["a@x.io", "a+tag@sub.example.com", "nope", "a@b", None]})
    duck.register("t", t)
    out = bt.from_arrow(t).dq.matches_format("e", "email").drop().collect()
    assert_same(
        out,
        duck.sql(
            "SELECT * FROM t WHERE e IS NULL OR "
            r"regexp_matches(e, '^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$')"
        ),
    )


def test_matches_format_uuid_and_ipv4():
    t = pa.table(
        {
            "u": ["3f2504e0-4f89-11d3-9a0c-0305e82c3301", "nope"],
            "ip": ["192.168.0.1", "999.1.1.1"],
        }
    )
    ds = bt.from_arrow(t)
    assert ds.dq.matches_format("u", "uuid").validate().violations["is_uuid(u)"] == 1
    assert ds.dq.matches_format("ip", "ipv4").validate().violations["is_ipv4(ip)"] == 1


def test_matches_format_rejects_unknown_name():
    with pytest.raises(PlanError, match="unknown format"):
        bt.from_pydict({"a": ["x"]}).dq.matches_format("a", "phone")


def test_references_is_a_constraint_the_chain_can_drop(duck):
    orders = pa.table({"oid": [1, 2, 3, 4], "cid": pa.array([10, 20, 99, None], pa.int64())})
    customers = pa.table({"cid": pa.array([10, 20], pa.int64())})
    duck.register("o", orders)
    duck.register("c", customers)
    out = bt.from_arrow(orders).dq.references("cid", to=bt.from_arrow(customers)).drop().collect()
    assert_same(out, duck.sql("SELECT * FROM o WHERE cid IS NULL OR cid IN (SELECT cid FROM c)"))


def test_references_counts_orphans_like_foreign_key():
    orders = bt.from_pydict({"cid": [10, 20, 99, None]})
    customers = bt.from_pydict({"cid": [10, 20]})
    report = orders.dq.references("cid", to=customers).validate()
    assert report.violations["references(cid)"] == 1
    assert orders.dq.foreign_key("cid", references=customers).count() == 1


def test_references_does_not_duplicate_rows_on_repeated_reference_keys():
    orders = bt.from_pydict({"cid": [10, 10, 20]})
    customers = bt.from_pydict({"cid": [10, 10, 10, 20]})
    clean, bad = orders.dq.references("cid", to=customers).quarantine()
    assert clean.count() == 3
    assert bad.count() == 0


def test_references_composite_key(duck):
    left = pa.table({"a": [1, 1, 2], "b": ["x", "y", "x"]})
    right = pa.table({"a": [1], "b": ["x"]})
    duck.register("l", left)
    duck.register("r", right)
    out = bt.from_arrow(left).dq.references(["a", "b"], to=bt.from_arrow(right)).drop().collect()
    assert_same(out, duck.sql("SELECT * FROM l WHERE (a, b) IN (SELECT a, b FROM r)"))


def test_where_scopes_a_constraint(duck):
    t = pa.table({"country": ["US", "FR", "US"], "state": ["CA", None, None]})
    duck.register("t", t)
    out = bt.from_arrow(t).dq.where(bt.col("country") == "US").not_null("state").drop().collect()
    assert_same(out, duck.sql("SELECT * FROM t WHERE NOT (country = 'US') OR state IS NOT NULL"))


def test_where_null_predicate_leaves_the_row_out_of_scope():
    t = bt.from_pydict({"flag": [True, None, False], "x": [None, None, None]})
    kept = t.dq.where(bt.col("flag")).not_null("x").drop().to_pydict()
    assert kept["flag"] == [None, False]


def test_not_in_future_flags_only_the_future():
    ds = bt.from_pydict({"ts": [dt.datetime(2020, 1, 1), dt.datetime(2999, 1, 1), None]})
    assert ds.dq.not_in_future("ts").validate().violations["not_in_future(ts)"] == 1
    assert ds.dq.not_in_future("ts").drop().count() == 2


def test_fresh_within_fails_on_stale_and_passes_on_new():
    stale = bt.from_pydict({"ts": [dt.datetime(2020, 1, 1)]})
    assert not stale.dq.fresh_within("ts", "1d").validate().ok
    fresh = bt.from_pydict({"ts": [dt.datetime.now() - dt.timedelta(seconds=5)]})
    assert fresh.dq.fresh_within("ts", "1d").validate().ok


def test_freshness_reads_the_clock_in_the_column_s_own_frame():
    """A row written a moment ago is fresh under a one-minute bound, in any timezone.

    Reading the wall clock as UTC while the column is a naive local timestamp offsets every
    age by the local UTC offset — silently, and only outside UTC. The measured age of a
    just-written row is the assertion that catches it.
    """
    just_now = bt.from_pydict({"ts": [dt.datetime.now()]})
    report = just_now.dq.fresh_within("ts", "1m").validate()
    assert report.ok, report.violations
    age = report.results[0].value
    assert 0 <= age < 60, f"newest row measured {age}s old"


def test_not_in_future_reads_the_clock_in_the_column_s_own_frame():
    just_now = bt.from_pydict({"ts": [dt.datetime.now() - dt.timedelta(seconds=1)]})
    assert just_now.dq.not_in_future("ts").validate().ok


def test_relation_level_constraints_measure_the_whole_table():
    ds = bt.from_pydict({"x": [1.0, 2.0, 3.0, None]})
    report = (
        ds.dq.row_count_between(4, 4)
        .mean_between("x", 1.9, 2.1)
        .null_rate_below("x", 0.3)
        .distinct_count_between("x", 3, 3)
        .validate()
    )
    assert report.ok, report.violations
    assert report.result("mean_between(x, 1.9, 2.1)").value == pytest.approx(2.0)


def test_relation_level_failure_is_reported_not_silently_ignored():
    ds = bt.from_pydict({"x": [1.0, 1.0, 1.0]})
    report = ds.dq.stddev_between("x", 0.5, None).validate()
    assert not report.ok
    assert report.violations["stddev_between(x, 0.5, None)"] == 1


def test_relation_level_constraint_refuses_to_drop_rows():
    ds = bt.from_pydict({"x": [1, 2]})
    with pytest.raises(PlanError, match="no violating row"):
        ds.dq.row_count_between(10).drop()
    with pytest.raises(PlanError, match="no violating row"):
        ds.dq.row_count_between(10).quarantine()


def test_schema_constraints_answer_without_executing():
    ds = bt.from_pydict({"id": [1], "name": ["a"]})
    ok = ds.dq.has_columns("id", "name").column_types({"id": "int64"}).validate()
    assert ok.ok
    bad = ds.dq.has_columns("id", "missing").validate()
    assert not bad.ok
    assert "missing" in bad.result("has_columns(id, missing)").detail


def test_no_unexpected_columns_catches_a_widened_schema():
    ds = bt.from_pydict({"id": [1], "leaked": ["secret"]})
    report = ds.dq.no_unexpected_columns("id").validate()
    assert not report.ok
    assert "leaked" in report.result("no_unexpected_columns(1 allowed)").detail


def test_broken_schema_contract_blocks_the_row_split():
    ds = bt.from_pydict({"id": [1]})
    with pytest.raises(DataQualityError, match="schema contract"):
        ds.dq.has_columns("nope").drop()


def test_column_types_reports_the_mismatch():
    ds = bt.from_pydict({"id": [1]})
    report = ds.dq.column_types({"id": "string"}).validate()
    assert not report.ok
    assert "int64" in report.result("column_types(id)").detail
