"""The single-device rung must read the columns the chain names, not the whole relation.

`whole_source_descriptor` has taken a projection all along, and the three `*_on_worker`
dispatchers did not pass one. So the rung reached when the fan-out declines read the relation
**whole**: measured on ClickBench, whose `hits` table has 105 columns, every query that got
here moved all of them onto one board to answer a two-column question and took **10.5 s**
against the CPU engine's 0.07 s.

It is the same defect the sharded aggregate had — recorded in `dist/gpu/aggregate.py` as "the
commonest accelerated shape there is read every column of the fact table to answer a
three-column query" — surviving on the path taken when sharding is not available.
"""

from __future__ import annotations

import pytest

from batcher.dist.gpu import dispatch

pytestmark = pytest.mark.unit


@pytest.fixture
def described(monkeypatch):
    """Capture the `(source, projection)` pairs the dispatchers describe, and decline the run."""
    seen = []

    def _describe(source, projection=None):
        seen.append((source, projection))
        return None  # `None` stops before the remote call, which needs a cluster

    monkeypatch.setattr(dispatch, "whole_source_descriptor", _describe)
    return seen


def _chain(*names: str) -> list[dict]:
    """A projection-only chain naming `names`, which is what `chain_projection` reads."""
    return [
        {
            "op": "project",
            "exprs": [{"alias": n, "expr": {"e": "col", "name": n}} for n in names],
        }
    ]


def test_a_chain_reads_only_the_columns_it_names(described):
    assert dispatch.gpu_chain_on_worker("src", _chain("a", "b")) is None
    assert described == [("src", ["a", "b"])]


def test_each_side_of_a_join_reads_only_its_own_chains_columns(described):
    join_ir = {"join_type": "inner", "left_keys": ["k"], "right_keys": ["k"], "output": []}
    assert (
        dispatch.gpu_join_on_worker("L", "R", _chain("a", "k"), _chain("b", "k"), join_ir, [])
        is None
    )
    assert described == [("L", ["a", "k"]), ("R", ["b", "k"])]


def test_each_union_input_reads_only_its_own_chains_columns(described):
    assert dispatch.gpu_union_on_worker(["A", "B"], [_chain("x"), _chain("y")], False, []) is None
    assert described == [("A", ["x"]), ("B", ["y"])]


def test_a_chain_that_cannot_be_narrowed_reads_the_relation(described):
    """`distinct` and `window` decide row identity from every column, so `chain_projection`
    answers `None` — which must reach the descriptor as "read it all", not as an empty list."""
    assert dispatch.gpu_chain_on_worker("src", [{"op": "distinct", "keys": []}]) is None
    assert described == [("src", None)]
