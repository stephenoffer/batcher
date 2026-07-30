"""After a build-side swap, the *right* input is the build — everywhere, not just in prose.

`adaptive_build_side` reassigns `node` to the swapped join and then reported the decision's
build side as `node.left if swap else node.right`: the pre-swap convention applied to a
post-swap node, which names the **probe**. Two consumers read the resulting `build_bytes`
and both were wrong.

The shape is the ordinary one — a small dimension on the left of a large fact — so the swap
fires on essentially every star-schema join written in that order.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.source_stats import collect_source_stats, column_bounds_needed
from batcher.kyber.optimizer import Optimizer

pytestmark = pytest.mark.unit

_SMALL, _WIDE_COLS = 5_000, 2


def _sides(big_rows: int):
    """A small single-column left and a large two-column right, joined left-to-right."""
    rng = np.random.default_rng(0)
    left = bt.from_arrow(pa.table({"k": pa.array(rng.integers(0, 1000, _SMALL).astype("int64"))}))
    right = bt.from_arrow(
        pa.table(
            {
                "k": pa.array(rng.integers(0, 1000, big_rows).astype("int64")),
                "v": pa.array(rng.random(big_rows)),
            }
        )
    )
    return left, right


def _decision(left, right):
    joined = left.join(right, on="k")
    stats = collect_source_stats(
        joined._sources, None, need_columns=column_bounds_needed(joined._plan)
    )
    _, decisions = Optimizer(sources=joined._sources, source_stats=stats).optimize_traced(
        joined._plan
    )
    return decisions[0]


@pytest.mark.parametrize("big_rows", [100_000, 2_000_000, 20_000_000])
def test_build_bytes_names_the_side_that_is_actually_built(big_rows):
    """The regression: `build_bytes` was the probe's size whenever a swap fired.

    At 20M rows on the right it reported 320,000,000 bytes for a build side that is 5,000
    rows of one `int64` column.
    """
    left, right = _sides(big_rows)
    decision = _decision(left, right)
    assert decision.swapped is True  # the small side really is on the left
    # The build is the 5,000-row left; its bytes must not scale with the probe.
    assert decision.build_bytes < 1_000_000, decision.build_bytes


def test_build_bytes_does_not_grow_with_the_probe_side():
    """The sharpest form: hold the build fixed, grow the probe 200x, and the figure must
    not move. It previously tracked the probe exactly."""
    sizes = [_decision(*_sides(n)).build_bytes for n in (100_000, 2_000_000, 20_000_000)]
    assert len(set(sizes)) == 1, sizes


def test_an_unswapped_join_is_unchanged():
    """The safety property: with the small side already on the right, no swap fires and
    the reported build is the right input exactly as before."""
    left, right = _sides(100_000)
    decision = _decision(right, left)  # large on the left, small on the right
    assert decision.swapped is False
    assert decision.build_bytes < 1_000_000


def test_the_join_still_returns_the_right_rows():
    """A build-side choice is a strategy, so it may never change the relation."""
    left, right = _sides(100_000)
    swapped_order = left.join(right, on="k").collect().num_rows
    natural_order = right.join(left, on="k").collect().num_rows
    assert swapped_order == natural_order
