"""Physical properties: what a node delivers, and what a consumer requires.

Cardinality says how many rows; this says in what shape. The two shapes a mature optimizer
reasons about are **ordering** (so a redundant sort can be removed) and **distribution** (so
a redundant shuffle can be skipped). These pin both, and pin the gap that made the ordering
property useless in practice: a `Project` dropped it, and a `Project` sits between a sort and
its consumer in essentially every real query.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.properties import (
    PhysicalProperties,
    delivered,
    hash_partitioned_on,
    project_ordering,
    satisfies,
)
from batcher.kyber.stats.estimator import StatsEstimator
from batcher.plan.logical import Sort
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _props(ds) -> PhysicalProperties:
    est = StatsEstimator(ds._sources)
    return delivered(ds._plan, est.estimate(ds._plan))


def _sorts(ds) -> int:
    plan = Optimizer(sources=ds._sources).optimize_full(ds._plan)[1]
    return sum(1 for n in walk(plan) if isinstance(n, Sort))


def _ds():
    return bt.from_arrow(pa.table({"a": [3, 1, 2], "b": [1, 2, 3], "c": [9, 8, 7]}))


# --- ordering ---------------------------------------------------------------------


def test_a_sort_delivers_its_ordering():
    assert _props(_ds().sort("a")).ordering == ("a",)


def test_ordering_survives_a_renaming_projection():
    """The gap this closes: a `Project` used to drop the delivered order entirely."""
    assert _props(_ds().sort("a").select(x=col("a"), y=col("b"))).ordering == ("x",)


def test_ordering_truncates_at_a_key_the_projection_drops():
    ds = _ds().sort("a", "b").select(x=col("a"))  # `b` is not carried
    assert _props(ds).ordering == ("x",)


def test_a_computed_output_cannot_carry_the_order():
    """`a + 1` is not `a`; it must not be claimed as the ordering key."""
    assert project_ordering.__doc__  # documented
    ds = _ds().sort("a").select(x=col("a") + 1)
    assert _props(ds).ordering == ()


def test_a_descending_sort_is_not_expressible_and_is_not_claimed():
    """The canonical form is ascending/nulls-last; anything else is simply not recorded."""
    assert _props(_ds().sort("a", descending=True)).ordering == ()


def test_a_stronger_ordering_satisfies_a_weaker_requirement():
    have = PhysicalProperties(ordering=("a", "b"))
    assert satisfies(have, PhysicalProperties(ordering=("a",)))
    assert not satisfies(PhysicalProperties(ordering=("a",)), PhysicalProperties(("a", "b")))


def test_the_redundant_sort_across_a_select_is_eliminated():
    """The whole point: `ORDER BY a` … `SELECT` … `ORDER BY a` should sort once."""
    ds = _ds().sort("a").select(x=col("a"), y=col("b")).sort("x")
    assert _sorts(ds) == 1
    assert ds.collect().to_pydict() == {"x": [1, 2, 3], "y": [2, 3, 1]}


def test_a_genuinely_different_sort_is_kept():
    """The rule must not eat a sort that changes the order."""
    ds = _ds().sort("a").select(x=col("a"), y=col("b")).sort("y")
    assert _sorts(ds) == 2
    assert ds.collect().to_pydict()["y"] == [1, 2, 3]


# --- distribution -----------------------------------------------------------------


def test_a_hash_join_output_is_partitioned_by_its_join_keys():
    left = bt.from_arrow(pa.table({"k": [1, 2], "v": [1, 2]}))
    right = bt.from_arrow(pa.table({"k": [1, 2], "w": [3, 4]}))
    assert hash_partitioned_on(left.join(right, on="k")._plan) == ("k",)


def test_an_outer_join_claims_no_partitioning():
    """A null-extended row's NULL key did not come from the bucket its row now sits in."""
    left = bt.from_arrow(pa.table({"k": [1, 2], "v": [1, 2]}))
    right = bt.from_arrow(pa.table({"k": [1], "w": [3]}))
    assert hash_partitioned_on(left.join(right, on="k", how="left")._plan) == ()


def test_an_aggregate_is_partitioned_by_its_group_keys():
    ds = _ds().group_by("a").agg(s=col("b").sum())
    assert hash_partitioned_on(ds._plan) == ("a",)


def test_a_filter_preserves_the_partitioning_of_its_input():
    """Row-shrinking never moves a row to a different bucket."""
    left = bt.from_arrow(pa.table({"k": [1, 2], "v": [1, 2]}))
    right = bt.from_arrow(pa.table({"k": [1, 2], "w": [3, 4]}))
    ds = left.join(right, on="k").filter(col("v") > 0)
    assert hash_partitioned_on(ds._plan) == ("k",)


def test_a_subset_partitioning_satisfies_a_superset_grouping():
    """A group-by on a *superset* of the join keys needs no second shuffle.

    Partitioning contains the opposite way from ordering: the *delivered* keys must be a
    subset of the *required* grouping. Rows in one `hash(k)` bucket contain every `(k, other)`
    group whole, so grouping by `(k, other)` is safe; the reverse is not.
    """
    delivered = PhysicalProperties(hash_partitioned_on=("k",))
    assert satisfies(delivered, PhysicalProperties(hash_partitioned_on=("k",)))
    assert satisfies(delivered, PhysicalProperties(hash_partitioned_on=("k", "other")))


def test_a_superset_partitioning_does_not_satisfy_a_subset_grouping():
    """The wrong-answer direction: `hash(k, other)` splits the `k` groups.

    Two rows sharing `k` but differing in `other` hash to different buckets, so a reducer
    that skipped the shuffle would emit a partial `k` group. This must be refused.
    """
    delivered = PhysicalProperties(hash_partitioned_on=("k", "other"))
    assert not satisfies(delivered, PhysicalProperties(hash_partitioned_on=("k",)))


def test_no_delivered_partitioning_satisfies_nothing_but_an_empty_requirement():
    none_delivered = PhysicalProperties()
    assert satisfies(none_delivered, PhysicalProperties())
    assert not satisfies(none_delivered, PhysicalProperties(hash_partitioned_on=("k",)))


# --- partitioning through a projection ---------------------------------------------


def test_partitioning_survives_a_projection_under_its_output_name():
    # A projection is map-only, so the bucket a row sits in never changes — but the key is
    # renamed, exactly as `project_ordering` renames the ordering.
    from batcher.kyber.properties import hash_partitioned_on

    left = bt.from_pydict({"k": [1, 2], "a": [1, 2]})
    right = bt.from_pydict({"k": [1], "b": [9]})
    projected = left.join(right, on="k", how="inner").select(kk=col("k"))
    assert hash_partitioned_on(projected._plan) == ("kk",)


def test_a_projection_that_drops_the_key_unclaims_the_partitioning():
    # The rows are still physically co-located, but the partitioning can no longer be *named*
    # — and a partitioning is a set, so a partial rename must not be claimed.
    from batcher.kyber.properties import hash_partitioned_on

    left = bt.from_pydict({"k": [1, 2], "a": [1, 2]})
    right = bt.from_pydict({"k": [1], "b": [9]})
    projected = left.join(right, on="k", how="inner").select(aa=col("a"))
    assert hash_partitioned_on(projected._plan) == ()
