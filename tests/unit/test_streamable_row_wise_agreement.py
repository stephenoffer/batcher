"""`is_streamable` and `_is_row_wise` must classify every node the same way.

`plan/logical/transforms.py::is_streamable` documents that it "mirrors
`dist.executors.plan_analysis._is_row_wise`" and that "the two MUST agree or a plan
streams one way and shuffles the other" — but nothing enforced it. A node type added to
one and not the other is a silent divergence: the single-node path streams a plan the
distributed path shuffles, and the two produce different results with no error.

This is the enforcement. Both predicates answer "is this node partition-independent" —
running it per batch / per partition yields exactly the single-node result.

`is_streamable` recurses over a whole plan; `_is_row_wise` classifies one node. The test
compares them per node type, with `MapBatches` called out because `_is_row_wise` excludes
it while its callers add it back explicitly (`plan_analysis.py` line ~125).
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.dist.executors.plan_analysis import _is_row_wise
from batcher.plan.logical import is_streamable

_BASE = {"x": [3, 1, 2], "y": ["a", "b", "a"], "l": [[1, 2], [3], [4]]}


def _ds():
    return bt.from_pydict(_BASE)


# (label, plan-producing callable, expected partition-independence)
_CASES = [
    ("scan", lambda d: d, True),
    ("filter", lambda d: d.filter(bt.col("x") > 1), True),
    ("project", lambda d: d.select("x"), True),
    ("with_columns", lambda d: d.with_columns(z=bt.col("x") * 2), True),
    ("sample_fraction", lambda d: d.sample(fraction=0.5), True),
    ("sample_fixed_n", lambda d: d.sample(n=2), False),
    ("aggregate", lambda d: d.group_by("y").agg(s=bt.col("x").sum()), False),
    ("sort", lambda d: d.sort("x"), False),
    ("limit", lambda d: d.limit(2), False),
    ("distinct", lambda d: d.distinct(), False),
]


@pytest.mark.unit
@pytest.mark.parametrize(("label", "build", "expected"), _CASES, ids=[c[0] for c in _CASES])
def test_is_streamable_matches_expected_partition_independence(label, build, expected):
    assert is_streamable(build(_ds())._plan) is expected


@pytest.mark.unit
@pytest.mark.parametrize(("label", "build", "expected"), _CASES, ids=[c[0] for c in _CASES])
def test_the_two_predicates_agree_on_every_node(label, build, expected):
    """The invariant the comment asserts but nothing checked."""
    node = build(_ds())._plan
    from batcher.plan.logical import Scan

    if isinstance(node, Scan):
        pytest.skip("`_is_row_wise` classifies transforms; a Scan is the recursion base")
    assert _is_row_wise(node) is expected, (
        f"{label}: is_streamable says {expected}, _is_row_wise says {_is_row_wise(node)} — "
        "the single-node and distributed paths would disagree"
    )


@pytest.mark.unit
def test_limit_is_not_partition_independent():
    """Pins WHY `Limit` is excluded: per batch it would keep n rows from EVERY batch."""
    ds = _ds().limit(2)
    assert is_streamable(ds._plan) is False
    # And the whole-relation answer really is 2 rows, not 2-per-partition.
    assert ds.collect().num_rows == 2


@pytest.mark.unit
def test_fixed_count_sample_is_not_partition_independent():
    """The subtle one both predicates call out: `n=` samples the WHOLE relation."""
    assert is_streamable(_ds().sample(n=2)._plan) is False
    assert is_streamable(_ds().sample(fraction=0.5)._plan) is True


@pytest.mark.unit
def test_map_batches_is_streamable_but_not_row_wise_by_design():
    """`_is_row_wise` excludes `MapBatches`; its callers add it back explicitly."""
    from batcher.plan.logical import MapBatches

    node = _ds().map_batches(lambda b: b)._plan
    assert isinstance(node, MapBatches)
    assert is_streamable(node) is True
    assert _is_row_wise(node) is False
