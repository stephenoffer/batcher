"""The allocator's arena is handed back when a query actually goes out of core.

`tests/unit/test_reclaim_before_spill.py` covers the trim itself and the seam it hangs off,
with the allocator stubbed. Neither can see the thing that was actually wrong: the call was
wired to the *live pressure* branch of the spill decision, and a plan whose estimated peak
exceeds the budget returns from the branch above it. That is the ordinary way a large query
spills, so the valve fired on a minority of the spills it was written for and the unit tests
agreed with it, because they asked the branch rather than the query.

So this file runs a real query over the real engine and asks the counter afterwards.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.carbonite.memory.reclaim import reclaim_stats, reset_reclaim_state
from batcher.config import active_config, set_config
from batcher.config.config import MemoryConfig

#: Small enough that a million-row group-by cannot be admitted, large enough to be a budget
#: rather than a degenerate zero. The same figure the spill matrix in `test_spilling.py` uses.
_TIGHT_BUDGET = 8 * 1024 * 1024


@pytest.fixture
def tight_memory_budget():
    """Run under a budget a large group-by cannot fit, then put the config back."""
    before = active_config()
    set_config(before.replace(memory=MemoryConfig(max_memory_bytes=_TIGHT_BUDGET)))
    reset_reclaim_state()
    try:
        yield
    finally:
        set_config(before)
        reset_reclaim_state()


def _wide_group_by(rows: int):
    data = {"k": [i % 997 for i in range(rows)], "v": [float(i) for i in range(rows)]}
    return bt.from_pydict(data).group_by("k").agg(a=bt.col("v").sum())


def test_a_query_the_estimate_sends_out_of_core_hands_the_arena_back(tight_memory_budget):
    # The regression this pins: the trim used to hang off the live-pressure branch of
    # `spill_reason`, which an over-budget estimate returns above. Nothing about this query
    # involves live pressure -- it is decided by the estimate alone -- and it must still trim.
    result = _wide_group_by(1_100_000).collect()

    assert result.num_rows == 997, "the spill must still answer correctly"
    stats = reclaim_stats()
    assert stats["attempts"] == 1, "one trim, at the point the query committed to disk"


def test_a_query_that_fits_never_pays_for_a_trim():
    # The other half, and the one that would make this a regression if it were wrong: a forced
    # walk of every heap on a query that comfortably fits is pure cost. The budget fixture is
    # deliberately not used here.
    reset_reclaim_state()
    try:
        result = _wide_group_by(50_000).collect()
        assert result.num_rows == 997
        assert reclaim_stats()["attempts"] == 0
    finally:
        reset_reclaim_state()


def test_the_answer_is_the_same_with_the_arena_handed_back(tight_memory_budget):
    # Result-invariance, stated as a test rather than as a claim in a docstring. The trim
    # returns pages the engine has finished with; it cannot move a value, and this is what
    # says so.
    spilled = _wide_group_by(1_100_000).collect().to_pydict()
    set_config(active_config().replace(memory=MemoryConfig(max_memory_bytes=None)))
    in_memory = _wide_group_by(1_100_000).collect().to_pydict()

    assert sorted(zip(spilled["k"], spilled["a"], strict=True)) == sorted(
        zip(in_memory["k"], in_memory["a"], strict=True)
    )
