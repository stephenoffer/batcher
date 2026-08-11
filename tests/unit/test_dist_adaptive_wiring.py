"""The distributed executor consults the learned sizing through the process-wide MetadataHub.

These check the *wiring* (not the Ray path): `shuffle_partitions` trims its reducer fan-out,
`speculation_policy` picks up the learned straggler factor, and `_adaptive_partition_count` seeds a
footer-less source's fan-out from a measured row count. The metadata hub is reset around every test
(conftest), so each starts cold and a seeded run must diverge only in the scheduling number, never
in any result.
"""

from __future__ import annotations

import pytest

from batcher.config import active_config
from batcher.core import default_hub
from batcher.dist.adaptive_sizing import record_partition_rows
from batcher.dist.executors import map as mapmod
from batcher.dist.executors.ray_runtime import shuffle_partitions
from batcher.dist.executors.ray_runtime.policies import speculation_policy
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId

pytestmark = pytest.mark.unit


def _record(kind: str, *, n_input: int = 0, t_ms: float = 0.0) -> None:
    default_hub().record(
        OperatorFeedback(
            op_id=OpId(1),
            kind=kind,
            n_actual=n_input,
            t_op_ms=t_ms,
            m_peak_bytes=0,
            selectivity=1.0,
            batch_size=16_384,
            n_input=n_input,
        )
    )


# --- shuffle_partitions: learned fan-out through the default hub --------------------------


def test_shuffle_partitions_cold_is_one_bucket_per_worker():
    # Cold store: no measured volume to raise the count, so every worker gets one bucket
    # and no more — full reduce parallelism at the smallest stream count.
    assert shuffle_partitions(8) == 8


def test_shuffle_partitions_does_not_trim_below_the_worker_count():
    """A small measured volume no longer strands workers.

    It used to trim to 1, which bounds each reducer's memory and also hands the entire
    reduce to a single worker — the shape that made the reduce phase slower the more
    workers were added.
    """
    for _ in range(4):
        _record("aggregate", n_input=1_000)  # tiny measured shuffle volume
    assert shuffle_partitions(64) == 64


# --- speculation_policy: learned straggler factor ----------------------------------------


def test_speculation_policy_cold_is_config_default():
    d = active_config().distributed
    assert speculation_policy().straggler_factor == d.speculation_straggler_factor


def test_speculation_policy_uses_learned_factor():
    for _ in range(6):
        _record("aggregate", t_ms=50.0)  # uniform task times → raise the factor
    default = active_config().distributed.speculation_straggler_factor
    assert speculation_policy().straggler_factor > default


# --- _adaptive_partition_count: learned rows for a footer-less source ---------------------


class _FooterlessSource:
    """A source with a stable identity but no cheaply-known row count (splits() unavailable)."""

    def identity(self) -> str:
        return "footerless://s"

    def splits(self, target_size=None):
        raise RuntimeError("no cheap split count")


def test_partition_count_falls_back_when_cold():
    import batcher as bt

    plan = bt.from_pydict({"x": [1]})._plan
    assert mapmod._adaptive_partition_count(_FooterlessSource(), plan, fallback=99) == 99


def test_partition_count_uses_learned_rows_when_warm():
    import batcher as bt

    plan = bt.from_pydict({"x": [1]})._plan
    for _ in range(4):
        record_partition_rows(default_hub(), "footerless://s", 40_000_000)
    got = mapmod._adaptive_partition_count(_FooterlessSource(), plan, fallback=99)
    assert got != 99 and got >= 1  # seeded from the measured size, not the blunt fallback


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
