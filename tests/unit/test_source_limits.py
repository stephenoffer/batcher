"""How many rows a scan's source is allowed to stop after.

`required_columns_per_source` says which columns to read and
`required_predicates_per_source` says which rows to skip; neither says when to *stop*, so
``read.postgres(...).limit(100)`` issued an unbounded SELECT and pulled the whole table
across the network to keep a hundred rows.

A `LIMIT` is a positional prefix, so the analysis is mostly about what it must *refuse*.
The dangerous case is `Filter`: ``Limit(Filter(p, x), n)`` keeps the first `n` rows that
pass `p`, while capping the source at `n` keeps the passing rows of the first `n` — fewer,
or none. Every test below that expects `{}` is guarding a rewrite that would lose rows.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.rules.source_limits import required_limits_per_source
from batcher.plan.logical import Limit, Scan, Union

pytestmark = pytest.mark.unit


def _limits(ds):
    return required_limits_per_source(ds._plan)


def test_a_limit_directly_over_a_scan_caps_the_source():
    assert _limits(bt.from_pydict({"a": [1, 2, 3]}).limit(2)) == {0: 2}


def test_the_offset_is_added_because_the_engine_skips_those_rows_itself():
    assert _limits(bt.from_pydict({"a": [1, 2, 3]}).limit(2, offset=5)) == {0: 7}


def test_a_projection_passes_the_cap_through():
    # Row-for-row and order-preserving: the nth projected row is the nth scanned row.
    assert _limits(bt.from_pydict({"a": [1, 2, 3]}).select("a").limit(2)) == {0: 2}


def test_stacked_limits_take_the_tighter_cap():
    assert _limits(bt.from_pydict({"a": [1, 2, 3]}).limit(9).limit(2)) == {0: 2}


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda ds: ds.filter(bt.col("a") > 1).limit(2), id="filter"),
        pytest.param(lambda ds: ds.distinct().limit(2), id="distinct"),
        pytest.param(
            lambda ds: ds.group_by("a").agg(n=bt.col("a").count()).limit(2), id="aggregate"
        ),
    ],
)
def test_an_operator_that_does_not_preserve_a_prefix_blocks_the_cap(build):
    assert _limits(build(bt.from_pydict({"a": [1, 2, 3]}))) == {}


def test_a_sort_no_longer_blocks_the_cap_because_the_ordering_goes_with_it():
    """`sort` was in the list above until the cap learned to carry its ordering.

    The first `n` rows of a sorted relation are still not the first `n` of its input —
    what changed is that the source is now told the ordering too, which is what makes the
    cap mean the same thing on both sides. See `test_topn_pushdown.py`, and note the
    ordering is only ever pushed *with* a cap.
    """
    from batcher.kyber.rules.source_limits import required_orderings_per_source

    dataset = bt.from_pydict({"a": [1, 2, 3]}).sort("a").limit(2)
    assert _limits(dataset) == {0: 2}
    assert required_orderings_per_source(dataset._plan) == {0: (("a", False, False),)}


def test_an_unlimited_plan_caps_nothing():
    assert _limits(bt.from_pydict({"a": [1, 2, 3]})) == {}


def _two_scans_of_one_source(left_limit: int | None, right_limit: int | None):
    """`Union` of two scans that share a `source_id`, each optionally capped.

    Built from plan nodes rather than from two `Dataset`s, because two bindings of one
    dataset get *separate* `source_id`s in the plan as written. They are collapsed onto a
    single index later, by `api.subplan_reuse._one_id_per_source` — deliberately, since
    that is what makes a repeated subplan visible to common-subplan elimination. So the
    shared-source shape this analysis has to survive is created by an optimization, and
    reaching it through the public API would be testing that pass instead of this one.
    """
    schema = bt.from_pydict({"a": [1, 2, 3]})._plan.schema
    branches = tuple(
        Scan(0, schema) if cap is None else Limit(Scan(0, schema), cap)
        for cap in (left_limit, right_limit)
    )
    return Union(branches)


def test_one_unbounded_scan_of_a_source_blocks_the_cap_for_every_scan_of_it():
    # A source read twice is capped only by something true of *both* reads. Capping it at
    # the limited branch's `n` would starve the unbounded branch of rows it needs, and the
    # engine's `Limit` only removes rows — it cannot put them back.
    assert required_limits_per_source(_two_scans_of_one_source(2, None)) == {}
    assert required_limits_per_source(_two_scans_of_one_source(None, 2)) == {}


def test_two_capped_scans_of_one_source_take_the_looser_cap():
    assert required_limits_per_source(_two_scans_of_one_source(2, 5)) == {0: 5}


def test_the_optimizer_publishes_the_cap_on_the_physical_plan():
    import batcher.kyber as kyber

    opt, _ = kyber.optimize_traced(bt.from_pydict({"a": [1, 2, 3]}).limit(2)._plan, sources=[])
    assert opt.source_limits == {0: 2}
