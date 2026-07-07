"""Learned out-of-core sizing — spill bucket count + compression from measured peak.

A larger measured working set should shard into more, smaller spill buckets (bounded
memory per bucket) and, once it is IO-bound, compress its buckets (trade CPU for less
disk/network). Both are sized from the LEARNED peak (`m_peak_bytes`-blended), fall back
to the caller's default on a cold/unsized plan, and are result-invariant: the number of
spill partitions only shards the shuffle and the IPC codec is lossless, so the collected
result is byte-identical either way (proven against the executed query).
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import Config, col, config_context
from batcher.carbonite import ResourceManager
from batcher.carbonite.manager import (
    _MAX_SPILL_PARTITIONS,
    _MIN_SPILL_PARTITIONS,
    _SPILL_BYTES_PER_PARTITION,
    _SPILL_COMPRESS_ABOVE,
)
from batcher.config.config import MemoryConfig
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan
from batcher.plan.resource import ResourceBounds


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _seed(hub: MetadataHub, kind: str, bytes_per_row: float, *, n: int = 30) -> None:
    for _ in range(n):
        hub.record(
            OperatorFeedback(
                op_id=OpId(1),
                kind=kind,
                n_actual=100,
                t_op_ms=1.0,
                m_peak_bytes=int(bytes_per_row * 1000),
                selectivity=0.1,
                batch_size=16384,
                n_input=1000,
            )
        )


def _plan(kind: str, m_max_bytes: int) -> PhysicalPlan:
    op = PhysicalOp(
        op_id=OpId(1),
        kind=kind,
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=m_max_bytes, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


# --- spill partition count ---------------------------------------------------


def test_partitions_scale_with_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=64.0)  # 1:1 with the assumed width
    rm = ResourceManager(hub=hub)
    small = rm.recommend_spill_partitions(_plan("Aggregate", _SPILL_BYTES_PER_PARTITION))
    big = rm.recommend_spill_partitions(_plan("Aggregate", 10 * _SPILL_BYTES_PER_PARTITION))
    assert small == _MIN_SPILL_PARTITIONS  # one bucket's worth → the floor
    assert big == 10  # ~10 buckets of the target size
    assert big > small


def test_partitions_bounded():
    rm = ResourceManager(hub=_hub())
    huge = rm.recommend_spill_partitions(_plan("Aggregate", 10**18))
    assert huge == _MAX_SPILL_PARTITIONS


def test_partitions_none_when_unsized():
    assert ResourceManager(hub=_hub()).recommend_spill_partitions(_plan("Aggregate", 0)) is None


def test_partitions_reflect_learned_inflation():
    # A family measured 8x wider than assumed shards into more buckets than the plan
    # estimate alone would suggest — the whole point of consuming m_peak_bytes.
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=64.0 * 8)
    plan = _plan("Aggregate", _SPILL_BYTES_PER_PARTITION)
    cold = ResourceManager().recommend_spill_partitions(plan)
    warm = ResourceManager(hub=hub).recommend_spill_partitions(plan)
    assert warm > cold


# --- spill compression -------------------------------------------------------


def test_compression_from_learned_peak():
    rm = ResourceManager(hub=_hub())
    assert rm.recommend_spill_compression(_plan("Aggregate", _SPILL_COMPRESS_ABOVE * 2)) is True
    assert rm.recommend_spill_compression(_plan("Aggregate", _SPILL_COMPRESS_ABOVE // 4)) is False
    assert rm.recommend_spill_compression(_plan("Aggregate", 0)) is None


# --- result-invariance -------------------------------------------------------


def _rows(tbl: pa.Table) -> list[tuple]:
    return sorted(tuple(r.values()) for r in tbl.to_pylist())


def test_spill_partition_count_is_result_invariant():
    # The lever recommend_spill_partitions drives only shards the shuffle: the result is
    # identical for a coarse (2) vs a fine (16) bucket count.
    t = pa.table({"k": [i % 17 for i in range(6000)], "v": list(range(6000))})

    def q(parts: int) -> pa.Table:
        return (
            bt.from_arrow(t)
            .group_by("k")
            .agg(s=col("v").sum(), n=col("v").count())
            .collect(spill=True, num_partitions=parts)
        )

    assert _rows(q(2)) == _rows(q(16))


def test_spill_compression_is_result_invariant():
    # The lever recommend_spill_compression drives is a lossless IPC codec choice: the
    # collected result is identical whether buckets are compressed (zstd) or not (None).
    t = pa.table({"k": [i % 9 for i in range(5000)], "s": [f"row-{i}" for i in range(5000)]})

    def q() -> pa.Table:
        return (
            bt.from_arrow(t)
            .group_by("k")
            .agg(n=col("s").count())
            .collect(spill=True, num_partitions=8)
        )

    with config_context(Config().replace(memory=MemoryConfig(spill_compression="zstd"))):
        compressed = q()
    with config_context(Config().replace(memory=MemoryConfig(spill_compression=None))):
        raw = q()
    assert _rows(compressed) == _rows(raw)
