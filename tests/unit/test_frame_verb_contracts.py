"""The plan-time refusals of the W8 frame verbs, which the differential files do not reach.

Each verb refuses the shapes it cannot answer before any data is read: a positional verb
without an order, a join with no predicate or key, a grouping filter that is not over
aggregates, and a row method under `having`. The messages name the fix, so they are pinned
by the phrase that carries it. Results are covered against DuckDB, Polars and Ray Data
elsewhere; this file needs no engine run.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture
def ds() -> bt.Dataset:
    return bt.from_pydict({"k": [1, 2], "v": [1.5, 2.5], "s": ["a", "b"]})


def test_join_where_needs_a_predicate(ds):
    with pytest.raises(bt.PlanError, match="at least one predicate"):
        ds.join_where(ds)
    with pytest.raises(bt.PlanError, match="must be expressions"):
        ds.join_where(ds, "k < k_right")


def test_update_needs_a_key_and_a_known_how(ds):
    with pytest.raises(bt.PlanError, match="with_row_index"):
        ds.update(ds)
    with pytest.raises(bt.PlanError, match="how must be one of"):
        ds.update(ds, on="k", how="outer")


@pytest.mark.parametrize("verb", ["zip", "split", "transpose"])
def test_positional_verbs_refuse_an_empty_order(ds, verb):
    call = {
        "zip": lambda: ds.zip(ds, order_by=[]),
        "split": lambda: ds.split(2, order_by=[]),
        "transpose": lambda: ds.transpose(order_by=[]),
    }[verb]
    with pytest.raises(bt.PlanError, match="with_row_index"):
        call()


def test_zip_needs_another_dataset(ds):
    with pytest.raises(bt.PlanError, match="at least one other dataset"):
        ds.zip(order_by="k")


def test_split_needs_a_positive_part_count(ds):
    with pytest.raises(bt.PlanError):
        ds.split(0, order_by="k")


def test_partition_by_refuses_an_unknown_key(ds):
    with pytest.raises(bt.PlanError, match="unknown column"):
        ds.partition_by("nope")


def test_having_needs_a_predicate_over_aggregates(ds):
    with pytest.raises(bt.PlanError, match="at least one predicate"):
        ds.group_by("s").having()
    with pytest.raises(bt.PlanError, match="over aggregates"):
        ds.group_by("s").having(bt.col("v") > 1)


@pytest.mark.parametrize("method", ["head", "tail"])
def test_having_refuses_the_row_methods(ds, method):
    grouped = ds.group_by("s").having(bt.count() > 1)
    with pytest.raises(bt.PlanError, match="returns rows"):
        getattr(grouped, method)(1, order_by="k")


def test_drop_nans_with_no_float_column_is_the_identity(ds):
    plain = bt.from_pydict({"k": [1, 2]})
    assert plain.drop_nans() is plain
