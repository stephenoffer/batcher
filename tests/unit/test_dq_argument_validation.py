"""What `ds.dq` refuses at the API edge, before anything executes.

A data contract is configuration, and configuration arrives wrong: an empty allow-list, a
tolerance expressed as a percentage, a bound pair the caller swapped. Each of those has a
silent reading — reject every row, treat 95 as 1.0, match nothing — so each is a typed error
naming the check and the argument instead.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


@pytest.fixture
def ds():
    return bt.from_pydict({"x": [1, 2, 3], "s": ["a", "b", "c"], "y": [1, 2, 3]})


def test_mostly_outside_zero_to_one_is_refused(ds):
    with pytest.raises(PlanError, match="fraction in"):
        ds.dq.positive("x", mostly=95)
    with pytest.raises(PlanError, match="fraction in"):
        ds.dq.positive("x", mostly=-0.1)


def test_unknown_severity_is_refused(ds):
    with pytest.raises(PlanError, match="'error' or 'warn'"):
        ds.dq.positive("x", severity="fatal")


def test_swapped_bounds_are_refused(ds):
    with pytest.raises(PlanError, match="swap the arguments"):
        ds.dq.in_range("x", 10, 0)
    with pytest.raises(PlanError, match="swap the arguments"):
        ds.dq.str_length_between("s", 5, 2)
    with pytest.raises(PlanError, match="swap the arguments"):
        ds.dq.mean_between("x", 10, 0)


def test_empty_value_sets_are_refused(ds):
    with pytest.raises(PlanError, match="non-empty"):
        ds.dq.accepted_values("s", [])
    with pytest.raises(PlanError, match="non-empty"):
        ds.dq.rejected_values("s", [])


def test_unbounded_relation_constraint_is_refused(ds):
    with pytest.raises(PlanError, match="at least one of low/high"):
        ds.dq.row_count_between()
    with pytest.raises(PlanError, match="at least one of low/high"):
        ds.dq.mean_between("x")


def test_rates_outside_zero_to_one_are_refused(ds):
    with pytest.raises(PlanError, match=r"\[0, 1\]"):
        ds.dq.null_rate_below("x", 50)
    with pytest.raises(PlanError, match=r"\[0, 1\]"):
        ds.dq.unique_ratio_above("x", 1.5)
    with pytest.raises(PlanError, match=r"\[0, 1\]"):
        ds.dq.quantile_between("x", 50, 0, 1)


def test_bad_regex_names_the_check_and_the_column(ds):
    with pytest.raises(PlanError, match=r"matches\('s'"):
        ds.dq.matches("s", "[")
    with pytest.raises(PlanError, match=r"not_matches\('s'"):
        ds.dq.not_matches("s", "(")


def test_numeric_bounds_on_a_text_column_are_refused(ds):
    with pytest.raises(PlanError, match="numeric bounds"):
        ds.dq.in_range("s", 0, 10)


def test_unknown_comparison_operator_is_refused(ds):
    with pytest.raises(PlanError, match="use one of"):
        ds.dq.compare_columns("x", "=<", "y")


def test_unknown_closed_side_is_refused(ds):
    with pytest.raises(PlanError, match="'both', 'left'"):
        ds.dq.in_range("x", 0, 10, closed="inner")


def test_mismatched_key_arity_is_refused(ds):
    other = bt.from_pydict({"a": [1], "b": [2]})
    with pytest.raises(PlanError, match="pair up one to one"):
        ds.dq.references(["x"], to=other, ref_columns=["a", "b"])
    with pytest.raises(PlanError, match="pair up one to one"):
        ds.dq.foreign_key(["x"], references=other, ref_columns=["a", "b"])


def test_empty_column_lists_are_refused(ds):
    with pytest.raises(PlanError, match="at least one column"):
        ds.dq.not_null()
    with pytest.raises(PlanError, match="at least one key column"):
        ds.dq.unique([])
    with pytest.raises(PlanError, match="at least one column name"):
        ds.dq.has_columns()


def test_where_refuses_a_constraint_it_cannot_scope(ds):
    with pytest.raises(PlanError, match="scopes row-level constraints"):
        ds.dq.where(bt.col("x") > 1).unique("x")
    with pytest.raises(PlanError, match="scopes row-level constraints"):
        ds.dq.where(bt.col("x") > 1).row_count_between(1)


def test_bad_duration_is_refused(ds):
    frame = bt.from_pydict({"ts": [1]})
    with pytest.raises(PlanError, match="cannot parse"):
        frame.dq.fresh_within("ts", "yesterday")
    with pytest.raises(PlanError, match="calendar unit"):
        frame.dq.fresh_within("ts", "1mo")
    with pytest.raises(PlanError, match="positive duration"):
        frame.dq.fresh_within("ts", 0)


def test_unknown_type_name_is_refused(ds):
    with pytest.raises(PlanError, match="not a type name"):
        ds.dq.column_types({"x": "int65"})


def test_repr_lists_the_accumulated_chain(ds):
    chain = ds.dq.not_null("x").positive("x")
    assert repr(chain) == "DatasetDQ(not_null(x), positive(x))"
    assert repr(ds.dq) == "DatasetDQ(no constraints)"


def test_the_chain_is_immutable(ds):
    base = ds.dq.not_null("x")
    extended = base.positive("x")
    assert len(base._constraints) == 1
    assert len(extended._constraints) == 2
