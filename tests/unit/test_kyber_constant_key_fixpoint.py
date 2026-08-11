"""`infer_join_predicate_from_constant_key` must not fight the zone-map rule forever.

The rule mirrors a join key's *proven* constant onto the other side as a filter, so a scan
can prune on it. `drop_filter_conjunct_implied_by_zonemap` exists to delete a conjunct the
zone map already implies. When both sides' statistics prove the key is the **same** constant,
the first rule adds a predicate the second immediately removes, and the PUSHDOWN phase cycles
`infer → push → merge → drop → infer` until the iteration cap stops it.

Nothing was wrong with the answers — every rule is semantics-preserving — which is why this
survived: the only symptoms were a "phase did not reach a fixpoint" warning on every such
query, a plan that depended on `fixpoint_iterations`, and the whole budget spent on rewrites
that cancelled (19 plan-changing rule applications where 1 was needed).

The trigger observed is an **in-memory** relation — `from_pydict`/`from_arrow`, where the
estimator knows the column exactly — whose join key holds a single distinct value. Parquet
sources were checked and do not cycle on the shapes tried, so this is not claimed to reach a
footer-backed scan; the guard is written against the statistic rather than the source kind, so
it holds either way.

The fix is the same shape as the rule's existing `is_cartesian_key_pair` skip: don't mirror a
constant the target side already provably holds.

That the rule still *fires* where it should is covered next door, by
`test_diff_join_constant_key_inference.py::test_the_rule_actually_fires_and_only_where_it_should`
— which counts firings against real Parquet footers, the proof this rule consumes. These
tests cover the other half: that it stops firing where firing achieved nothing.
"""

from __future__ import annotations

import logging

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.expr_ir import col

pytestmark = pytest.mark.unit

#: Substring of the driver's non-convergence diagnostic.
_WARNING = "did not reach a fixpoint"


def _one_row() -> pa.Table:
    """A single-row relation: every column's min == max, so every key is provably constant."""
    return pa.table(
        {
            "k": pa.array([0], pa.int64()),
            "g": pa.array(["a"], pa.string()),
            "i": pa.array([7], pa.int64()),
        }
    )


def _fixpoint_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if _WARNING in r.getMessage()]


def test_a_self_join_on_a_provably_constant_key_reaches_a_fixpoint(caplog):
    ds = bt.from_arrow(_one_row())
    with caplog.at_level(logging.WARNING, logger="batcher.kyber"):
        ds.select("k", "g").join(ds.select("k", "i"), on="k").explain()
    assert _fixpoint_warnings(caplog) == []


def test_two_distinct_one_row_sources_joined_on_an_equal_constant_converge(caplog):
    """Not a self-join artifact: any two sides whose stats pin the key to one value."""
    left = bt.from_arrow(_one_row())
    right = bt.from_arrow(_one_row())
    with caplog.at_level(logging.WARNING, logger="batcher.kyber"):
        left.select("k", "g").join(right.select("k", "i"), on="k").explain()
    assert _fixpoint_warnings(caplog) == []


def test_the_join_still_returns_the_right_rows():
    """The cycle was invisible in the answers, so the fix must stay invisible there too."""
    ds = bt.from_arrow(_one_row())
    got = ds.select("k", "g").join(ds.select("k", "i"), on="k").collect().to_pydict()
    assert got["k"] == [0]
    assert got["g"] == ["a"]
    assert got["i"] == [7]


def test_an_empty_and_a_multi_row_relation_were_never_affected(caplog):
    """Bracketing the fix: only the exactly-one-distinct-value case ever cycled."""
    for rows in (0, 2, 5):
        table = pa.table(
            {
                "k": pa.array(list(range(rows)), pa.int64()),
                "g": pa.array([f"g{i}" for i in range(rows)], pa.string()),
                "i": pa.array(list(range(rows)), pa.int64()),
            }
        )
        ds = bt.from_arrow(table)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="batcher.kyber"):
            ds.select("k", "g").join(ds.select("k", "i"), on="k").explain()
        assert _fixpoint_warnings(caplog) == [], f"{rows} rows"


def test_a_repeated_key_value_keeps_its_join_multiplicity(caplog):
    """A guard that turned the inference into a filter-and-dedup would show up here.

    3 x 2 rows all matching on `k = 7` must still produce 6. This shape does not itself
    cycle — only the one-row cases above do — so it is a non-regression check on the guard's
    blast radius rather than a reproduction of the bug.
    """
    left = bt.from_arrow(pa.table({"k": pa.array([7, 7, 7], pa.int64()), "a": [1, 2, 3]}))
    right = bt.from_arrow(pa.table({"k": pa.array([7, 7], pa.int64()), "b": [8, 9]}))
    with caplog.at_level(logging.WARNING, logger="batcher.kyber"):
        joined = left.join(right, on="k")
        joined.explain()
    assert _fixpoint_warnings(caplog) == []
    assert joined.collect().num_rows == 6


def test_a_contradictory_constant_is_not_skipped():
    """Two *different* proven constants make the join empty — that is worth inferring.

    The guard keys on the constants being equal, so this path is untouched: the predicate
    is a contradiction rather than a tautology, and the zone-map rule has no reason to
    delete it.
    """
    left = bt.from_arrow(pa.table({"k": pa.array([1], pa.int64()), "a": pa.array([9], pa.int64())}))
    right = bt.from_arrow(
        pa.table({"k": pa.array([2], pa.int64()), "b": pa.array([8], pa.int64())})
    )
    assert left.join(right, on="k").collect().num_rows == 0


def test_the_rewrite_is_stable_under_the_iteration_cap():
    """The user-visible symptom: the plan must not depend on `fixpoint_iterations`."""
    ds = bt.from_arrow(_one_row())
    plans = set()
    for iterations in (1, 2, 3, 8, 20):
        cfg = bt.Config().replace(optimizer=bt.OptimizerConfig(fixpoint_iterations=iterations))
        with bt.config_context(cfg):
            built = ds.select("k", "g").join(ds.select("k", "i"), on="k")
            plans.add(str(built.with_columns(z=col("i") + 1).explain()))
    assert len(plans) == 1, f"plan varies with the iteration cap: {plans}"
