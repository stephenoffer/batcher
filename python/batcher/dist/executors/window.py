"""Distributed window functions over a disk Arrow-IPC shuffle.

Window functions are computed *per partition*, so hash-shuffling input rows by the
window's partition keys co-locates every whole partition on a single reducer; the
reducer runs the ordinary window operator over its rows and the concatenation of
all reducers is identical to single-node execution. Unlike the aggregate shuffle
(which moves partial *state*), this moves the raw rows and reuses the *same* hash
partitioner so a partition is never split across reducers.

That decomposition is `keyed_shuffle.keyed_row_shuffle`, shared with the keyed dedup, which
shuffles by key the same way and differs only in pre-reducing on the map side. This module is
what a window puts into it: its partition keys as the shuffle key, its raw rows as the map
plan, and itself as the reduce plan.

Restricted to windows whose partition keys are plain columns and whose input is a
breaker-free single source; anything else falls back to single-node.
"""

from __future__ import annotations

from batcher.dist.executors.keyed_shuffle import keyed_row_shuffle, scan_rooted_ir
from batcher.dist.executors.plan_analysis import _relabel_single_source, empty_result_table
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan, Window


def _distributed_window(
    above: list[LogicalPlan],
    window: Window,
    sources: list[Source],
    workers: int,
    hub=None,
    metrics_out=None,
    *,
    materialize: bool = True,
):
    """Run `window` across `workers` by hash-shuffling rows by its partition keys.

    `materialize=False` hands the reducers' output back as a `MaterializedSource` instead
    of collecting it. A window emits one row per input row, so the collect is the whole
    relation through the driver — the shuffle underneath has supported keeping it in place
    all along (`keyed_row_shuffle`), and this is what connects the two. Only honored when
    nothing is stacked `above`, so the caller must handle either return type."""
    # Partition-key column positions in the window input's output (caller guarantees
    # every partition key is a plain `Col`).
    cols = window.input.available_columns()
    key_indices = [cols.index(k.name) for k in window.partition_keys]
    map_plan, source_id = _relabel_single_source(window.input)
    return keyed_row_shuffle(
        above,
        map_plan=map_plan,
        reduce_ir=scan_rooted_ir(window),
        key_indices=key_indices,
        out_schema=empty_result_table(window, window.available_columns()).schema,
        source=sources[source_id],
        workers=workers,
        hub=hub,
        tag="win",
        metrics_out=metrics_out,
        materialize=materialize,
    )
