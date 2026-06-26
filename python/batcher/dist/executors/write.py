"""Distributed write — parallel data-file writers + one driver-side commit.

Each worker writes its own shard's data files (its own ``part-NNNNN`` files, with
Hive partitioning if requested) and returns a list of `WrittenFile` locators —
no data flows back through the driver. The driver concatenates the locators into
one `WriteManifest` (a commutative merge) and the caller performs a single
`commit`. This is the file-sink form of the two-phase write the lakehouse sinks
build on for ACID commits.

Two entry points:

* `_distributed_write` re-shards an already-collected result table (used for
  reduced results — aggregates/joins — where the output is small).
* `_distributed_write_plan` is the *streaming* path: each worker reads its own
  source partition, runs the (breaker-free) plan, and writes its output directly,
  so a result larger than the driver's memory never lands on the driver.
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa

from batcher.io.manifest import WriteManifest, WrittenFile
from batcher.io.source import Source
from batcher.plan.logical import LogicalPlan


def _distributed_write(
    sink: Any, table: pa.Table, path: str, partition_by: list[str] | None, workers: int
) -> WriteManifest:
    """Write `table` as `workers` shards in parallel, returning the merged manifest.

    The result is always a directory of ``part-*`` files (one per shard), so
    shards never collide. Single-node callers use `Sink.write_partitioned`
    directly; this path is for an already-collected distributed result.
    """
    from batcher.dist.executors.ray_runtime import _ensure_ray, gather_map_results

    _ensure_ray(workers)
    n = table.num_rows
    per = max(1, -(-n // workers))  # ceil
    shards = [table.slice(i * per, per) for i in range(workers) if i * per < n]
    if not shards:  # empty result still writes one (empty) shard for a valid dir
        shards = [table.slice(0, 0)]

    # Gather with preemption recovery: a write-shard whose worker is lost is resubmitted
    # onto a survivor. Each shard writes a deterministic `part-{idx}` file, so a resubmit
    # overwrites any partial file the dead worker left — idempotent, no orphan.
    results: list[list[WrittenFile]] = gather_map_results(
        lambda idx: _write_shard.remote(sink, shards[idx], path, partition_by, idx),
        len(shards),
    )
    return WriteManifest(tuple(f for shard_files in results for f in shard_files))


def _write_shard(
    sink: Any, shard: pa.Table, path: str, partition_by: list[str] | None, idx: int
) -> list[WrittenFile]:
    return sink.write_partitioned(shard, path, partition_by=partition_by, file_index=idx)


def _distributed_write_plan(
    plan: LogicalPlan,
    sources: list[Source],
    path: str,
    fmt: str,
    sink_kwargs: dict[str, Any] | None,
    partition_by: list[str] | None,
    workers: int,
) -> WriteManifest:
    """Streaming distributed write for a breaker-free single-source plan.

    Each worker reads its source partition (with projection + predicate pushed),
    runs the plan, and writes its output files directly to the sink — only
    `WrittenFile` manifests return to the driver, so the full result never
    materializes there and no shared filesystem is required (the input partition is
    a split-manifest the worker reads from storage, or a shipped batch list).
    """
    from batcher.dist.executors.partition_io import partition_descriptors, source_pushdown
    from batcher.dist.executors.plan_analysis import _relabel_single_source
    from batcher.dist.executors.ray_runtime import (
        _ensure_ray,
        engine_config_json,
        gather_map_results,
    )

    _ensure_ray(workers)
    cfg_json = engine_config_json()  # driver config → shipped to workers
    map_plan, sid = _relabel_single_source(plan)
    map_ir = json.dumps(map_plan.to_ir())
    projection, predicate = source_pushdown(map_plan, 0)

    parts = partition_descriptors(sources[sid], workers, projection=projection, predicate=predicate)
    # Recover a preempted write-shard onto a survivor; the worker recomputes its
    # partition from the durable split descriptor and rewrites its `part-{idx}` file
    # (deterministic name ⇒ idempotent overwrite, no orphaned partial output).
    results: list[list[WrittenFile]] = gather_map_results(
        lambda idx: _write_plan_shard.remote(
            map_ir, parts[idx], fmt, sink_kwargs, path, partition_by, idx, cfg_json
        ),
        len(parts),
    )
    return WriteManifest(tuple(f for shard_files in results for f in shard_files))


def _write_plan_shard(
    map_ir: str,
    partition: dict,
    fmt: str,
    sink_kwargs: dict[str, Any] | None,
    path: str,
    partition_by: list[str] | None,
    idx: int,
    engine_config: str,
) -> list[WrittenFile]:
    import batcher._native as nat
    from batcher.dist.executors.partition_io import read_partition_descriptor
    from batcher.io.sink import SINKS

    batches = read_partition_descriptor(partition)
    out = nat.execute_plan(map_ir, [batches], engine_config) if batches else []
    if not out or sum(b.num_rows for b in out) == 0:
        return []
    table = pa.Table.from_batches(out)
    sink = SINKS.get(fmt)(**(sink_kwargs or {}))
    return sink.write_partitioned(table, path, partition_by=partition_by, file_index=idx)
