"""`chain_projection` — what a single-scan GPU chain must read, and when it must read everything.

The sharded aggregate and the sharded join fan-outs read their relation without narrowing it,
because neither had a way to ask this question: `shard_descriptors` has taken a `projection` all
along, the tree fan-out passes one, and these two passed `None`. So the commonest accelerated
shape there is — a group-by over a scan — moved every column of the fact table to answer a
three-column query, off storage, across the host link, and as resident device memory the shard
was then priced against.

The direction of caution is what these cases pin. Reading a column nobody wanted is slow;
dropping one somebody did is wrong. So every operator that cannot narrow safely must answer
"all of them", and the answer for a chain must cover everything the chain's *whole output*
needs — not merely what its last operator names.
"""

from __future__ import annotations

import pytest

from batcher.core.gpu_plan.pruning import chain_projection

pytestmark = pytest.mark.unit


def _col(name: str) -> dict:
    return {"e": "col", "name": name}


def _project(*names: str) -> dict:
    return {"op": "project", "exprs": [{"alias": n, "expr": _col(n)} for n in names]}


def _filter(name: str) -> dict:
    return {"op": "filter", "predicate": _col(name)}


def _aggregate(keys: list[str], aggs: list[tuple[str, str]]) -> dict:
    return {
        "op": "aggregate",
        "group_keys": [{"alias": k, "expr": _col(k)} for k in keys],
        "aggregates": [{"alias": a, "func": "sum", "input": _col(c)} for a, c in aggs],
    }


def test_an_empty_chain_reads_everything():
    """A bare scan narrows to nothing, which must read the relation as it is."""
    assert chain_projection([]) is None


def test_an_aggregate_reads_its_keys_and_its_inputs():
    got = chain_projection([_aggregate(["g"], [("s", "v")])])
    assert got == ["g", "v"]


def test_a_filter_under_an_aggregate_contributes_its_predicate():
    """The commonest real shape: the filter's column is read even though nothing above names it."""
    got = chain_projection([_filter("d"), _aggregate(["g"], [("s", "v")])])
    assert got == ["d", "g", "v"]


def test_a_projection_does_not_hide_what_the_operators_above_it_need():
    """The whole chain's output is wanted, so a projection keeps every surviving expression."""
    assert chain_projection([_project("a", "b")]) == ["a", "b"]


def test_an_opaque_operator_refuses_to_narrow():
    """`distinct` decides row identity from every column it is given, so pruning under it would
    change the answer rather than fail. It must read everything."""
    assert chain_projection([{"op": "distinct"}, _aggregate(["g"], [("s", "v")])]) is None


def test_a_window_refuses_to_narrow():
    assert chain_projection([{"op": "window"}]) is None
