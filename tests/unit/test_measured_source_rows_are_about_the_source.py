"""A measured row count is written down as the *source's* size only when it is one.

`dist.executors.map._record_source_rows` persists what a distributed run produced under the
scanned source's identity, so the next run can size its partition count from a measurement
instead of the blunt cluster-fill worker count. That is a fact about the source only when the
plan that produced it neither drops rows nor adds them — and the executor runs plenty that
does.

The shape that made this concrete: `_distributed_distinct_limit` runs a per-partition
`Distinct(limit=k)` and passed the hub, so a billion-row table that answered one
`distinct().limit(10)` recorded `workers x 10` as its size. `_adaptive_partition_count` seeds
the *next* run from that, sizes itself to one partition, runs the whole query on one worker,
and records a smaller number still. Nothing fails; the cluster just stops being used.

The dispatcher had been withholding the hub by hand at six call sites to avoid exactly this,
each with a paragraph of comment, and the seventh was missed. `preserves_source_row_count` is
that rule stated once in the neutral `plan` layer, so a call site that forgets is no longer a
silent perf cliff — and, because the recording is now safe at the source, those six sites got
their hub back and with it the learned partition sizing they had been forgoing.

This pins both halves: the predicate's classification, and that `_record_source_rows` honours
it. The predicate is tested through the public `Dataset` builders rather than by constructing
nodes, so a node whose row arithmetic changes is caught by the plan it actually produces.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.logical import preserves_source_row_count

pytestmark = pytest.mark.unit

_T = pa.table(
    {
        "a": pa.array([1, 2, 2, 3], pa.int64()),
        "b": pa.array([10, 20, 30, 40], pa.int64()),
        "l": pa.array([[1, 2], [3], [], [4, 5]], pa.list_(pa.int64())),
    }
)


#: `(builder, whether its output has one row per source row)`. The `True` set is exactly the
#: operators that reorder or widen; everything else resizes and must not be believed.
_SHAPES = {
    "scan": (lambda ds: ds, True),
    "project": (lambda ds: ds.select("a"), True),
    "with_columns": (lambda ds: ds.with_columns(c=bt.col("a") * 2), True),
    "sort": (lambda ds: ds.sort("a"), True),
    "window": (
        lambda ds: ds.window(partition_by=["a"], order_by=["b"], functions={"r": "row_number"}),
        True,
    ),
    "row_id": (lambda ds: ds.with_row_index(), True),
    "project_over_sort": (lambda ds: ds.sort("a").select("a"), True),
    "filter": (lambda ds: ds.filter(bt.col("a") > 1), False),
    "limit": (lambda ds: ds.limit(2), False),
    "sample_n": (lambda ds: ds.sample(n=2), False),
    "distinct": (lambda ds: ds.distinct(), False),
    "distinct_limit": (lambda ds: ds.distinct().limit(2), False),
    "aggregate": (lambda ds: ds.group_by("a").agg(n=bt.col("b").count()), False),
    "unnest": (lambda ds: ds.explode("l"), False),
    "project_over_filter": (lambda ds: ds.filter(bt.col("a") > 1).select("a"), False),
    "sort_over_filter": (lambda ds: ds.filter(bt.col("a") > 1).sort("a"), False),
}


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_predicate_classifies_the_shape(shape):
    builder, expected = _SHAPES[shape]
    assert preserves_source_row_count(builder(bt.from_arrow(_T))._plan) is expected


@pytest.mark.parametrize("shape", sorted(_SHAPES))
def test_the_recorder_writes_only_what_is_about_the_source(monkeypatch, shape):
    """`_record_source_rows` must consult the predicate, not just be called carefully."""
    from batcher.dist import adaptive_sizing
    from batcher.dist.executors import map as dist_map

    builder, expected = _SHAPES[shape]
    written: list[float] = []
    # Patched where it is *defined*: `_record_source_rows` imports it inside the function, so
    # a patch on `dist.executors.map` binds a name nothing resolves and every shape would look
    # as though it recorded nothing — the assertion would pass for the wrong half of the table.
    monkeypatch.setattr(
        adaptive_sizing,
        "record_partition_rows",
        lambda hub, identity, rows: written.append(rows),
    )

    class _Src:
        def identity(self):
            return "src"

    dist_map._record_source_rows(None, _Src(), builder(bt.from_arrow(_T))._plan, 1234)
    assert bool(written) is expected
