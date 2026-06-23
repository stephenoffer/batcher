"""Predicate-pushdown plumbing: IR translation + Kyber's per-source extraction.

These run without the native engine. They check (a) the IR→backend-filter
translators handle the pushable subset and reject the rest, and (b) Kyber records
a `Filter` sitting directly above a `Scan` as that source's pushed predicate, but
NOT a filter separated from the scan by a pipeline breaker.
"""

from __future__ import annotations

import batcher as bt
from batcher.io.predicate import to_pyarrow_expression, to_sql_where
from batcher.kyber.rules.projections import required_predicates_per_source


def test_required_predicates_for_filter_over_scan():
    plan = bt.from_pydict({"x": [1], "y": [2]}).filter(bt.col("x") > 5)._plan
    preds = required_predicates_per_source(plan)
    assert preds[0]["op"] == "gt"
    assert preds[0]["left"]["name"] == "x"


def test_no_predicate_when_separated_by_breaker():
    # Filter above an Aggregate (a pipeline breaker) is not adjacent to the Scan,
    # so it is not pushed to the source.
    plan = (
        bt.from_pydict({"x": [1, 1], "y": [2, 3]})
        .group_by("x")
        .agg(s=bt.col("y").sum())
        .filter(bt.col("x") > 5)
        ._plan
    )
    assert required_predicates_per_source(plan) == {}


def test_no_predicate_without_filter():
    plan = bt.from_pydict({"x": [1], "y": [2]})._plan
    assert required_predicates_per_source(plan) == {}


def test_pyarrow_translation_pushable_subset():
    ir = ((bt.col("x") > 5) & (bt.col("y") == 3)).to_ir()
    assert to_pyarrow_expression(ir) is not None  # comparisons + AND are pushable


def test_pyarrow_translation_rejects_unpushable():
    # A comparison between two columns is not pushable (no literal).
    ir = (bt.col("x") > bt.col("y")).to_ir()
    assert to_pyarrow_expression(ir) is None


def test_sql_where_translation():
    ir = ((bt.col("x") >= 5) & (bt.col("y") == "a")).to_ir()
    where = to_sql_where(ir)
    assert where == "(x >= 5 AND y = 'a')"


def test_sql_where_rejects_unpushable():
    assert to_sql_where((bt.col("x") > bt.col("y")).to_ir()) is None
