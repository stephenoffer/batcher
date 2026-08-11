"""The distributed ASOF reducer must send the same node the single-node path lowers.

`dist.executor` runs an ASOF join per co-partitioned bucket, and it builds that bucket's IR
itself with the per-task scans substituted for the two inputs. Everything else about the node
has to survive the substitution, and nothing else checks that: the distributed path is the
only sender of this IR, so a dropped field is a *wrong answer at cluster scale* rather than
an error — an ASOF `tolerance` honoured on one node and ignored across the shuffle silently
prices every trade against a stale quote.

That is not hypothetical. The reducer used to restate the field list, and when `tolerance`
and `direction` were added it went on sending `backward` and no tolerance at all. Both paths
now share `AsofJoin.shape_ir`, and this pins the sharing.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def _asof(**kwargs):
    """A `Dataset.join_asof` plan over two small tables, with `kwargs` passed through."""
    trades = bt.from_arrow(
        pa.table(
            {
                "sym": pa.array(["A", "A"]),
                "t": pa.array([10, 40], type=pa.int64()),
                "size": pa.array([100, 200], type=pa.int64()),
            }
        )
    )
    quotes = bt.from_arrow(
        pa.table(
            {
                "sym": pa.array(["A"]),
                "t": pa.array([8], type=pa.int64()),
                "price": pa.array([1.0]),
            }
        )
    )
    return trades.join_asof(quotes, on="t", by="sym", **kwargs)._plan


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"tolerance": 5},
        {"direction": "forward"},
        {"direction": "nearest", "tolerance": 100},
        {"allow_exact_matches": False},
        {"direction": "nearest", "tolerance": 2, "allow_exact_matches": False},
    ],
    ids=["default", "tolerance", "forward", "nearest_tol", "strict", "all_three"],
)
def test_the_reducer_ir_equals_the_single_node_ir_but_for_the_inputs(kwargs):
    from batcher.dist.executor import _asof_reducer_ir

    node = _asof(**kwargs)
    reducer = _asof_reducer_ir(node)
    single = node.to_ir()

    assert reducer["left"] == {"op": "scan", "source_id": 0}
    assert reducer["right"] == {"op": "scan", "source_id": 1}
    # Every other key, by value. Comparing the whole dict rather than a named list is what
    # makes this catch a field nobody remembered to add here.
    assert {k: v for k, v in reducer.items() if k not in ("left", "right")} == {
        k: v for k, v in single.items() if k not in ("left", "right")
    }


def test_the_match_settings_actually_reach_the_reducer():
    """The comparison above would also pass if both sides dropped a field, so name them."""
    from batcher.dist.executor import _asof_reducer_ir

    reducer = _asof_reducer_ir(_asof(direction="nearest", tolerance=7, allow_exact_matches=False))
    assert reducer["direction"] == "nearest"
    assert reducer["tolerance"] == 7
    assert reducer["allow_exact_matches"] is False
