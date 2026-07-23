"""Merge / data-quality / SCD ergonomics: fluent builders and actionable errors.

Unit-level checks on the control-plane builders — clause legality, reprs, aliases,
and typed error messages that name the fix. In-memory data only.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import source_col, target_col
from batcher._internal.errors import DataQualityError, PlanError
from batcher.api.merge import simple_clauses
from batcher.api.merge.clauses import (
    NOT_MATCHED,
    NOT_MATCHED_BY_SOURCE,
    MergeClause,
    legal_actions,
    validate_clause,
)

pytestmark = pytest.mark.unit


def _changes():
    return bt.from_pydict({"id": [2, 3], "v": [99, 30]})


def _target(tmp_path):
    p = str(tmp_path / "t.parquet")
    bt.from_pydict({"id": [1, 2], "v": [10, 20]}).write.parquet(p)
    return p


# --- MergeBuilder / MergeWhen ----------------------------------------------
def test_merge_into_requires_keys(tmp_path):
    with pytest.raises(PlanError, match="at least one key column"):
        _changes().write.merge_into(_target(tmp_path), on=[])


def test_merge_builder_repr_shows_target_and_clauses(tmp_path):
    p = _target(tmp_path)
    b = _changes().write.merge_into(p, on="id")
    assert p in repr(b)
    assert "no clauses yet" in repr(b)
    b = b.when_matched().update({"v": source_col("v")})
    assert "matched:update" in repr(b)


def test_merge_when_repr_names_population(tmp_path):
    w = _changes().write.merge_into(_target(tmp_path), on="id").when_matched()
    assert "WHEN MATCHED" in repr(w)


def test_merge_when_rejects_impossible_action_early(tmp_path):
    p = _target(tmp_path)
    with pytest.raises(PlanError, match="WHEN MATCHED clause cannot insert"):
        _changes().write.merge_into(p, on="id").when_matched().insert(v=source_col("v"))
    with pytest.raises(PlanError, match="WHEN NOT MATCHED clause cannot delete"):
        _changes().write.merge_into(p, on="id").when_not_matched().delete()


def test_merge_delta_aliases(tmp_path):
    p = _target(tmp_path)
    b = _changes().write.merge_into(p, on="id").when_matched().updateAll()
    assert b.clauses[0].action == "update"
    b2 = _changes().write.merge_into(p, on="id").when_not_matched().insertAll()
    assert b2.clauses[0].action == "insert"


def test_merge_empty_update_names_star(tmp_path):
    p = _target(tmp_path)
    with pytest.raises(PlanError, match="update_all"):
        _changes().write.merge_into(p, on="id").when_matched().update()


def test_merge_execute_without_clauses_rejected(tmp_path):
    p = _target(tmp_path)
    with pytest.raises(PlanError, match="no clauses were added"):
        _changes().write.merge_into(p, on="id").execute()


def test_legal_actions_by_population():
    assert legal_actions("matched") == ("update", "delete")
    assert legal_actions("not_matched") == ("insert",)


def test_simple_clauses_suggests_on_typo():
    with pytest.raises(PlanError, match="when_matched must be one of"):
        simple_clauses("updat", "insert")
    with pytest.raises(PlanError, match="when_not_matched must be one of"):
        simple_clauses("update", "insrt")


def test_validate_clause_wrong_side_names_the_fix():
    with pytest.raises(PlanError, match="use source_col"):
        validate_clause(
            MergeClause(kind=NOT_MATCHED, action="insert", values={"v": target_col("v")}),
            ["id", "v"],
        )
    with pytest.raises(PlanError, match="use target_col"):
        validate_clause(
            MergeClause(kind=NOT_MATCHED_BY_SOURCE, action="update", values={"v": source_col("v")}),
            ["id", "v"],
        )


def test_validate_clause_unknown_column_suggests():
    with pytest.raises(PlanError, match="Did you mean 'v'"):
        validate_clause(
            MergeClause(kind="matched", action="update", values={"vv": source_col("v")}),
            ["id", "v"],
        )


# --- Data quality -----------------------------------------------------------
def test_dq_empty_constraints_are_plan_errors():
    ds = bt.from_pydict({"id": [1]})
    with pytest.raises(PlanError, match="not_null"):
        ds.dq.not_null()
    with pytest.raises(PlanError, match="unique"):
        ds.dq.unique([])


def test_dq_in_range_swapped_bounds_rejected():
    ds = bt.from_pydict({"age": [1, 2, 3]})
    with pytest.raises(PlanError, match="swap the arguments"):
        ds.dq.in_range("age", 100, 0)


def test_dq_fail_message_names_constraints_and_suggests_recovery():
    ds = bt.from_pydict({"age": [40, -1, 25]})
    with pytest.raises(DataQualityError, match="in_range") as exc:
        ds.dq.in_range("age", 0, 120).fail()
    assert exc.value.violations["in_range(age, 0, 120)"] == 1
    assert ".drop()" in str(exc.value) or ".quarantine()" in str(exc.value)


def test_dq_quarantine_splits_valid_and_invalid():
    ds = bt.from_pydict({"x": [1, 2, -3]})
    clean, rejected = ds.dq.in_range("x", 0, 10).quarantine()
    assert clean.to_pydict() == {"x": [1, 2]}
    assert rejected.to_pydict() == {"x": [-3]}


def test_dq_report_str_and_bool():
    ds = bt.from_pydict({"x": [1, 2, -3]})
    report = ds.dq.in_range("x", 0, 10).validate()
    assert bool(report) is False
    assert "in_range" in str(report)
    ok = bt.from_pydict({"x": [1, 2]}).dq.in_range("x", 0, 10).validate()
    assert bool(ok) is True
    assert str(ok) == "ValidationReport(ok)"


# --- SCD --------------------------------------------------------------------
def test_scd_apply_changes_requires_keys(tmp_path):
    feed = bt.from_pydict({"id": [1], "seq": [1]})
    with pytest.raises(PlanError, match="key column"):
        feed.scd.apply_changes(str(tmp_path / "x.parquet"), keys=[], sequence_by="seq")


def test_scd_apply_changes_missing_sequence_column_is_named(tmp_path):
    feed = bt.from_pydict({"id": [1], "city": ["A"]})
    with pytest.raises(PlanError, match="sequence_by column 'nope' is not in the change feed"):
        feed.scd.apply_changes(str(tmp_path / "y.parquet"), keys="id", sequence_by="nope")


def test_scd_type2_requires_track(tmp_path):
    ds = bt.from_pydict({"id": [1], "city": ["A"]})
    with pytest.raises(PlanError, match="track"):
        ds.scd.type2(str(tmp_path / "z.parquet"), keys="id", track=[], as_of="2024-01-01")


def test_scd_type1_requires_keys(tmp_path):
    ds = bt.from_pydict({"id": [1], "city": ["A"]})
    with pytest.raises(PlanError, match="key column"):
        ds.scd.type1(str(tmp_path / "q.parquet"), keys=[])
