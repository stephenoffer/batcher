"""Tolerance, severity, annotation, and reuse — the policy half of `ds.dq`.

These are the behaviours where "the check ran" and "the check was enforced" come apart, and
each one has a wrong answer that looks right: a tolerated constraint that silently stops
removing rows, a warning that quietly gates the pipeline anyway, an annotation that names
the constraints a row passed. Every assertion below pins one of those apart.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import DataQualityError


def _mixed():
    return bt.from_pydict({"id": [1, 2, 3, 4], "x": [5, -1, 7, -2]})


def test_mostly_moves_the_pass_line_not_the_rows():
    ds = _mixed()
    report = ds.dq.positive("x", mostly=0.5).validate()
    assert report.ok  # half the rows pass, and half is the bar
    assert report.violations["positive(x)"] == 2  # still counted
    assert ds.dq.positive("x", mostly=0.5).drop().count() == 2  # still dropped
    assert ds.dq.positive("x", mostly=0.5).fail().count() == 4  # and never raises


def test_mostly_below_the_pass_rate_still_fails():
    report = _mixed().dq.positive("x", mostly=0.9).validate()
    assert not report.ok
    assert report.result("positive(x)").pass_rate == 0.5


def test_mostly_of_one_is_the_default_strictness():
    assert not _mixed().dq.positive("x").validate().ok
    assert not _mixed().dq.positive("x", mostly=1.0).validate().ok


def test_warn_severity_reports_without_enforcing():
    ds = _mixed()
    report = ds.dq.positive("x", severity="warn").validate()
    assert report.ok
    assert report.violations["positive(x)"] == 2
    assert [r.name for r in report.warnings] == ["positive(x)"]
    assert report.failed == ()
    # A warning removes no row and raises nothing.
    assert ds.dq.positive("x", severity="warn").drop().count() == 4
    assert ds.dq.positive("x", severity="warn").fail().count() == 4
    clean, bad = ds.dq.positive("x", severity="warn").quarantine()
    assert (clean.count(), bad.count()) == (4, 0)


def test_error_and_warn_compose_in_one_chain():
    ds = _mixed()
    chain = ds.dq.not_null("id").positive("x", severity="warn")
    assert chain.validate().ok
    assert chain.drop().count() == 4


def test_fail_names_only_the_blocking_constraints():
    ds = _mixed()
    with pytest.raises(DataQualityError) as err:
        ds.dq.positive("x").not_null("id", severity="warn").fail()
    assert "positive(x)" in str(err.value)
    assert "not_null" not in str(err.value)


def test_annotate_names_what_each_row_failed():
    ds = bt.from_pydict({"id": [1, None], "x": [5, -1]})
    out = ds.dq.not_null("id").positive("x").annotate().to_pydict()
    assert out["dq_failed"] == ["", "not_null(id),positive(x)"]
    assert out["id"] == [1, None]


def test_annotate_keeps_every_row_and_adds_one_column():
    ds = _mixed()
    out = ds.dq.positive("x").annotate("why")
    assert out.count() == 4
    assert out.schema.names == ["id", "x", "why"]


def test_annotate_includes_warnings():
    ds = _mixed()
    out = ds.dq.positive("x", severity="warn").annotate().to_pydict()
    assert out["dq_failed"] == ["", "positive(x)", "", "positive(x)"]


def test_annotate_refuses_to_overwrite_a_column():
    ds = bt.from_pydict({"dq_failed": ["a"], "x": [1]})
    with pytest.raises(Exception, match="overwrite"):
        ds.dq.positive("x").annotate()


def test_annotate_covers_uniqueness_and_references():
    ds = bt.from_pydict({"k": [1, 1, 2], "fk": [9, 1, 1]})
    ref = bt.from_pydict({"fk": [1]})
    # A reference check lowers to a join, and a join is free to reorder — so pair each
    # verdict with the row it belongs to rather than comparing two lists positionally.
    out = ds.dq.unique("k").references("fk", to=ref).annotate().to_pydict()
    verdicts = sorted(zip(out["k"], out["fk"], out["dq_failed"], strict=True))
    assert verdicts == [
        (1, 1, "unique(k)"),
        (1, 9, "unique(k),references(fk)"),
        (2, 1, ""),
    ]


def test_on_rebinds_a_contract_to_another_dataset():
    contract = bt.from_pydict({"x": [1]}).dq.in_range("x", 0, 10).not_null("x")
    other = bt.from_pydict({"x": [1, 2, 99, None]})
    report = contract.on(other).validate()
    assert report.violations == {"in_range(x, 0, 10)": 1, "not_null(x)": 1}
    assert contract.on(other).drop().count() == 2


def test_report_exposes_rows_rates_and_plain_data():
    report = _mixed().dq.positive("x").validate()
    assert report.rows == 4
    assert report.total_violations == 2
    result = report.result("positive(x)")
    assert (result.rows, result.violations, result.pass_rate) == (4, 2, 0.5)
    payload = report.to_dict()
    assert payload["ok"] is False
    assert payload["constraints"][0]["name"] == "positive(x)"


def test_report_of_a_clean_dataset_renders_ok():
    report = bt.from_pydict({"x": [1, 2]}).dq.positive("x").validate()
    assert str(report) == "ValidationReport(ok)"
    assert report.rows in (0, 2)  # 0 when the contract was discharged from metadata


def test_uniqueness_report_agrees_with_the_split():
    ds = bt.from_pydict({"k": [1, 1, 1, 2]})
    report = ds.dq.unique("k").validate()
    clean, bad = ds.dq.unique("k").quarantine()
    assert report.violations["unique(k)"] == bad.count() == 3
    assert clean.count() == 1


def test_empty_relation_passes_every_row_constraint():
    ds = bt.from_pydict({"x": [1]}).filter(bt.col("x") > 100)
    report = ds.dq.positive("x").not_null("x").validate()
    assert report.ok
    assert ds.dq.positive("x").drop().count() == 0


def test_multi_batch_input_agrees_with_single_batch():
    values = list(range(1000))
    values[7] = -1
    single = bt.from_pydict({"x": values})
    batched = bt.from_pydict({"x": values[:500]}).union(bt.from_pydict({"x": values[500:]}))
    assert single.dq.positive("x").validate().violations == (
        batched.dq.positive("x").validate().violations
    )
    assert single.dq.positive("x").drop().count() == batched.dq.positive("x").drop().count()


def test_every_constraint_result_reaches_the_event_bus():
    """A contract that is checked and never charted is one nobody notices degrading."""
    from batcher._internal import events

    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        _mixed().dq.not_null("id").positive("x").validate()
    finally:
        unsubscribe()
    published = {e.name: e.fields for e in seen if e.kind == events.DQ}
    assert set(published) == {"not_null(id)", "positive(x)"}
    # The passing check is published too — a series that appears only on failure has no
    # baseline to compare against.
    assert published["not_null(id)"]["violations"] == 0
    assert published["not_null(id)"]["ok"] is True
    assert published["positive(x)"]["violations"] == 2
    assert published["positive(x)"]["ok"] is False
    assert published["positive(x)"]["rows"] == 4


def test_a_relation_level_result_publishes_its_measured_value():
    from batcher._internal import events

    seen: list[events.Event] = []
    unsubscribe = events.subscribe(seen.append)
    try:
        bt.from_pydict({"x": [1.0, 3.0]}).dq.mean_between("x", 0.0, 1.0).validate()
    finally:
        unsubscribe()
    fields = next(e.fields for e in seen if e.kind == events.DQ)
    assert fields["check"] == "aggregate"
    assert fields["value"] == 2.0
    assert fields["ok"] is False


def test_streaming_split_matches_the_collected_split():
    ds = bt.from_pydict({"x": [1, -2, 3, -4, 5]})
    clean, bad = ds.dq.positive("x").quarantine()
    streamed = [row for batch in clean.iter_batches() for row in batch.to_pydict()["x"]]
    assert sorted(streamed) == sorted(clean.to_pydict()["x"])
    assert clean.count() + bad.count() == 5
