"""Every distributed reducer must send the node the single-node path lowers.

`dist/` runs each breaker per co-partitioned bucket, and builds that bucket's IR itself with
per-task scans substituted for the inputs. Everything else about the node has to survive the
substitution — and nothing else checks it, because the distributed path is the only sender of
this IR. CI installs no Ray, so a dropped field is invisible until cluster scale, where it
surfaces as a wrong answer or a silently worse plan rather than an error.

It had already happened twice. `AsofJoin` gained `tolerance` and `direction`, and the reducer
went on sending `backward` and no tolerance at all — a trade priced against a stale quote,
distributed only. And `Join`'s IR was restated in **four** places, two of which omitted
`strategy`, so a planner-chosen `broadcast` or `sort_merge` reverted to `hash` on the Flight
and spilling paths.

Both are now one `shape_ir()` per node, read by every builder. This pins that: the assertion
compares the *whole* dict rather than a named list of fields, so a field added tomorrow and
forgotten in `dist/` fails here rather than at cluster scale.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_LEFT = pa.table(
    {
        "k": pa.array([1, 2, 3], type=pa.int64()),
        "t": pa.array([10, 20, 30], type=pa.int64()),
        "v": pa.array(["a", "b", "c"]),
    }
)
_RIGHT = pa.table(
    {
        "k": pa.array([1, 2], type=pa.int64()),
        "t": pa.array([5, 15], type=pa.int64()),
        "w": pa.array(["x", "y"]),
    }
)


def _inputs_are_the_per_task_scans(ir: dict) -> None:
    assert ir["left"] == {"op": "scan", "source_id": 0}
    assert ir["right"] == {"op": "scan", "source_id": 1}


def _matches_but_for_the_inputs(reducer: dict, single: dict) -> None:
    drop = ("left", "right")
    assert {k: v for k, v in reducer.items() if k not in drop} == {
        k: v for k, v in single.items() if k not in drop
    }


@pytest.mark.parametrize("strategy", ["hash", "broadcast", "sort_merge"])
def test_every_hash_join_reducer_carries_the_whole_node(strategy):
    """All four builders, against the same node — including the planner's strategy.

    Two of the four used to omit `strategy`. It is documented as a hint that never changes
    the relation, so the cost was a silently worse plan rather than a wrong answer — which
    is exactly the kind of drift no result comparison can find.
    """
    from batcher.dist.executors.join import _join_reducer_ir

    node = dataclasses.replace(
        bt.from_arrow(_LEFT).join(bt.from_arrow(_RIGHT), on="k")._plan, strategy=strategy
    )
    single = node.to_ir()
    reducer = _join_reducer_ir(node)
    _inputs_are_the_per_task_scans(reducer)
    _matches_but_for_the_inputs(reducer, single)
    assert reducer["strategy"] == strategy


def test_the_asof_reducer_carries_the_whole_node():
    from batcher.dist.executor import _asof_reducer_ir

    node = (
        bt.from_arrow(_LEFT)
        .join_asof(
            bt.from_arrow(_RIGHT),
            on="t",
            by="k",
            direction="nearest",
            tolerance=7,
            allow_exact_matches=False,
        )
        ._plan
    )
    reducer = _asof_reducer_ir(node)
    _inputs_are_the_per_task_scans(reducer)
    _matches_but_for_the_inputs(reducer, node.to_ir())
    assert reducer["direction"] == "nearest"
    assert reducer["tolerance"] == 7
    assert reducer["allow_exact_matches"] is False


def test_the_range_join_reducer_carries_the_whole_node():
    from batcher.dist.executor import _range_join_reducer_ir
    from batcher.plan.logical import JoinOutputCol, RangeCondition, RangeJoin

    node = RangeJoin(
        bt.from_arrow(_LEFT)._plan,
        bt.from_arrow(_RIGHT)._plan,
        (RangeCondition(left_key="t", right_key="t", op="lt"),),
        "inner",
        (
            JoinOutputCol(side="left", name="v", alias="v"),
            JoinOutputCol(side="right", name="w", alias="w"),
        ),
    )
    reducer = _range_join_reducer_ir(node)
    _inputs_are_the_per_task_scans(reducer)
    _matches_but_for_the_inputs(reducer, node.to_ir())


def test_the_sort_builders_read_the_seam_rather_than_restating_it():
    """The shuffle sort and the streaming top-N driver build their IR inline, so there is
    no reducer function to call — what is checkable is that they *spread* `shape_ir()`
    rather than listing fields, which is what makes them correct by construction."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "python" / "batcher"
    for rel in ("dist/executors/sort.py", "core/streaming/drivers.py"):
        src = (root / rel).read_text()
        assert "**sort.shape_ir()" in src, f"{rel} restates the sort IR by hand"
        assert '"keys": sort_keys_ir(sort.keys)' not in src, f"{rel} still lists fields"


@pytest.mark.parametrize("node_type", ["join", "asof", "range", "sort"])
def test_the_seam_is_the_single_source_of_truth(node_type):
    """`shape_ir()` must be exactly `to_ir()` minus the inputs, for every node that has it.

    That equality is what makes the reducers correct by construction rather than by
    remembering, so it is asserted directly rather than only through the builders.
    """
    from batcher.plan.logical import JoinOutputCol, RangeCondition, RangeJoin

    left, right = bt.from_arrow(_LEFT), bt.from_arrow(_RIGHT)
    if node_type == "sort":
        node = left.sort("t", descending=True)._plan
        single = node.to_ir()
        assert node.shape_ir() == {k: v for k, v in single.items() if k != "input"}
        return
    if node_type == "join":
        node = left.join(right, on="k")._plan
    elif node_type == "asof":
        node = left.join_asof(right, on="t", by="k", tolerance=3)._plan
    else:
        node = RangeJoin(
            left._plan,
            right._plan,
            (RangeCondition(left_key="t", right_key="t", op="lt"),),
            "inner",
            (JoinOutputCol(side="left", name="v", alias="v"),),
        )
    single = node.to_ir()
    assert node.shape_ir() == {k: v for k, v in single.items() if k not in ("left", "right")}


# --- the ratchet ---------------------------------------------------------------------
#
# The three fixes above were each found by reading, one at a time, after the damage was
# already in the tree. This is the check that finds the *next* one: any relational node
# built by hand in `dist/` must be covered by a contract test above, or listed here with
# the reason it does not need one.

#: Ops emitted by hand in `dist/` that carry no node-shaped payload to drift, with why.
_NO_NODE_TO_DRIFT = {
    # `Sort` now carries its own `shape_ir()` too, and the one remaining literal in
    # `dist/flight_sort.py` is a *generic* helper taking `keys`/`limit` rather than a
    # `Sort` node, so there is no node to copy from.
    "sort": "the last literal is a generic helper over keys/limit, not a node copy",
    # An aggregate reducer is assembled from partial/finalize specs rather than copied
    # from an `Aggregate` node, so there is no single-node lowering to compare against.
    "aggregate": "built from mergeable partial/finalize specs, not copied from a node",
    # Not a relational node: an expression tag inside a GPU projection.
    "div": "a scalar expression tag, not a RelOp",
    # The per-task input every reducer substitutes.
    "scan": "the per-task input itself",
}

#: Ops a contract test above compares field-for-field against the node's own `to_ir()`.
_COVERED = {"hash_join", "asof_join", "range_join"}


def test_a_hand_built_op_is_covered_or_explained():
    """Every `{"op": ...}` literal in `dist/` is either contract-tested or explained.

    A new distributed path that hand-writes a node's IR is exactly how the `tolerance` and
    `strategy` fields went missing. Adding one now fails this test until its author either
    routes it through `shape_ir()` and adds a comparison above, or records here why the op
    has nothing to drift.
    """
    import re
    from pathlib import Path

    dist = Path(__file__).resolve().parents[2] / "python" / "batcher" / "dist"
    assert dist.is_dir(), dist
    found: dict[str, set[str]] = {}
    for path in dist.rglob("*.py"):
        for op in re.findall(r'"op":\s*"(\w+)"', path.read_text()):
            found.setdefault(op, set()).add(str(path.relative_to(dist)))
    unexplained = {
        op: sorted(files)
        for op, files in found.items()
        if op not in _COVERED and op not in _NO_NODE_TO_DRIFT
    }
    assert not unexplained, (
        "these ops are built by hand in dist/ with no reducer-IR contract test:\n  "
        + "\n  ".join(f"{op}: {files}" for op, files in sorted(unexplained.items()))
        + "\nRoute the builder through the node's shape_ir() and compare it above, or "
        "add the op to _NO_NODE_TO_DRIFT with the reason it cannot drift."
    )
