"""The widest column in a row is the one nothing filters on, and it was never measured.

`learn_column_stats` sketches only `learnable_columns(plan)` — join keys, group keys, and the
columns a `Filter` mentions — on the sound reasoning that a KLL grid for a column nothing
predicates on is pure loss. That reasoning covers distinct counts, quantiles and
most-common-values. It does not cover the **byte width**, because `StatsEstimator.row_width`
sums per-column widths over every *output* column.

So the payload — the embedding, the document, the frame — was both the dominant term in a row's
width and the one column never measured, and no amount of re-running the query fixed it.
`annotate.py`'s own table says what that costs: at the flat prior a 768-dim embedding sizes
12 GB tasks and a 1080p frame sizes 25 TB ones.

The fix has one non-obvious constraint, and the second half of this file is about it: the
presence of an entry in `AVG_BYTES_KEY` is `learn_column_stats`'s "already sketched" marker,
chosen deliberately over `ndv` (which a cheaper seeding pass also writes). Recording a payload
column's width *there* would mark it done and cost it its quantiles forever, so the cheap widths
get their own table and the marker keeps meaning what it meant.
"""

from __future__ import annotations

import pytest

import batcher as bt
import batcher.core as core
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.column_tables import (
    AVG_BYTES_KEY,
    ROW_BYTES_KEY,
    STATS_NAMESPACE,
)
from batcher.kyber.learning import load_learned_stats

pytestmark = pytest.mark.unit

_PAYLOAD = "x" * 4096
_ROWS = 5_000


def _table(hub_reset: None = None):
    return bt.from_pydict(
        {"k": [i % 50 for i in range(_ROWS)], "doc": [_PAYLOAD for _ in range(_ROWS)]}
    )


def _columns(hub, key) -> dict[str, float]:
    """The learned table `key`, with the source qualifier stripped off each column."""
    raw = hub.get_keyed_param(STATS_NAMESPACE, key) or {}
    return {k.split("\x1f")[-1]: v for k, v in raw.items()}


def test_a_payload_column_gets_a_width_from_a_query_that_never_mentions_it():
    """The defect, stated as the case it produces."""
    hub = core.default_hub()
    frame = _table()
    query = frame.filter(bt.col("k") < 40)

    cold = CardinalityEstimator(frame._sources).row_width(query._plan, 64.0)
    query.collect()
    warm = CardinalityEstimator(frame._sources, load_learned_stats(hub)).row_width(
        query._plan, 64.0
    )
    # 8 bytes of key plus a 4 KiB payload. The prior guesses at the payload and is ~93x low.
    assert cold < 100.0
    assert warm == pytest.approx(4104.0, rel=0.01)


def test_the_width_is_the_arrow_byte_size_per_row():
    """Not a sketch and not a sample — the buffer size Arrow already knows, over the rows."""
    hub = core.default_hub()
    _table().filter(bt.col("k") < 40).collect()
    cheap = _columns(hub, ROW_BYTES_KEY)
    assert cheap["k"] == pytest.approx(8.0)
    # A 4 KiB string plus its offset; exact, not estimated.
    assert cheap["doc"] == pytest.approx(4100.0, rel=0.01)


def test_a_cheap_width_does_not_mark_a_column_as_sketched():
    """The trap. `AVG_BYTES_KEY`'s presence is the "already measured" marker for the sketches.

    Recording the payload's width there would retire the column from sketching, so a later
    query that *does* filter on it could never learn its quantiles or most-common-values —
    a silent regression traded for the fix.
    """
    hub = core.default_hub()
    _table().filter(bt.col("k") < 40).collect()
    assert set(_columns(hub, ROW_BYTES_KEY)) == {"k", "doc"}
    assert set(_columns(hub, AVG_BYTES_KEY)) == {"k"}, "doc must stay unsketched"


def test_a_column_with_a_cheap_width_is_still_sketched_when_a_query_needs_it():
    """The other half of the trap, end to end: the marker still admits the column later."""
    hub = core.default_hub()
    frame = _table()
    frame.filter(bt.col("k") < 40).collect()
    assert "doc" not in _columns(hub, AVG_BYTES_KEY)

    frame.filter(bt.col("doc") > "a").collect()
    assert "doc" in _columns(hub, AVG_BYTES_KEY), "the sketch pass must still reach it"


def test_a_sketched_width_outranks_the_cheap_one():
    """Two tables, and the sketched figure wins where both have an entry.

    The cheap reading is a buffer size; the sketched one is measured over the sample the rest
    of the column statistics come from. Where both exist they describe the same column, and
    fixing the order makes which one a plan used deterministic.
    """
    from batcher.kyber.stats.estimator import StatsEstimator

    est = StatsEstimator(
        [],
        {
            AVG_BYTES_KEY: {"src\x1fc": 99.0},
            ROW_BYTES_KEY: {"src\x1fc": 11.0, "src\x1fonly_cheap": 7.0},
        },
    )
    est._source_key = lambda source_id: "src"  # type: ignore[method-assign]
    cols = est.learned_columns(0)
    assert cols["c"].avg_bytes == 99.0, "the sketched width wins"
    assert cols["only_cheap"].avg_bytes == 7.0, "a cheap-only column is still visible"


def test_an_unkeyable_source_records_nothing():
    """A width that cannot be attributed to a source would be applied to the wrong one."""
    from batcher.api.terminal._metadata import _learn_row_bytes

    hub = core.default_hub()
    before = dict(_columns(hub, ROW_BYTES_KEY))
    _learn_row_bytes(hub, [[]], [None])  # no batches, no source
    assert _columns(hub, ROW_BYTES_KEY) == before


def test_learning_never_raises_into_a_query():
    """Best-effort, like every other measurement path here."""
    from batcher.api.terminal._metadata import _learn_row_bytes

    class Boom:
        ephemeral = False

        def __getattr__(self, name):
            raise RuntimeError("boom")

    _learn_row_bytes(core.default_hub(), [[object()]], [Boom()])


def test_a_steady_state_query_writes_nothing():
    """`merge_column_table` is a whole-table read-modify-write, and this runs every query.

    Ungated, a served workload would pay that write forever to re-record numbers that cannot
    move: a column's byte width is a property of its schema. The gate is the learning loop's
    own `is_material_change`, so "material" means one thing across the codebase.
    """
    hub = core.default_hub()
    frame = _table()
    frame.filter(bt.col("k") < 40).collect()
    first = dict(_columns(hub, ROW_BYTES_KEY))
    assert first, "the first run must record"

    writes = []
    original = hub.put_keyed_param

    def counting(namespace, key, value):
        writes.append(key)
        return original(namespace, key, value)

    hub.put_keyed_param = counting  # type: ignore[method-assign]
    try:
        frame.filter(bt.col("k") < 40).collect()
    finally:
        hub.put_keyed_param = original  # type: ignore[method-assign]
    assert ROW_BYTES_KEY not in writes, "an unchanged width must not be rewritten"
    assert _columns(hub, ROW_BYTES_KEY) == first
