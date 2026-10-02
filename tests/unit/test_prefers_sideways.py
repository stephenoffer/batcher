"""Kyber sends the sideways verdict only when a probe side is small beside the aggregate it reads.

`EngineConfig.prefer_sideways` routes a plan to the executor that restricts a join's build-side
aggregate to the probe side's keys. The engine can see that the shape allows it but not whether
it cuts anything, so the size half is Kyber's: a probe side at least `SIDEWAYS_MIN_RATIO` times
smaller than the aggregate's input, which is itself at least `SIDEWAYS_MIN_ROWS`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.optimizer import facade
from batcher.kyber.pass_base import OptimizerContext

pytestmark = pytest.mark.unit

_FACT_ROWS = 400_000


def _fact() -> bt.Dataset:
    return bt.from_arrow(
        pa.table(
            {
                "k": pa.array([i % 50_000 for i in range(_FACT_ROWS)]),
                "v": pa.array(range(_FACT_ROWS)),
            }
        )
    )


def _outer(rows: int) -> bt.Dataset:
    return bt.from_arrow(pa.table({"ok": pa.array([i * 7 % 50_000 for i in range(rows)])}))


def _verdict(ds: bt.Dataset) -> bool:
    ctx = OptimizerContext(
        config=active_config(),
        sources=ds._sources,
        hub=None,
        estimator=CardinalityEstimator(ds._sources, {}),
    )
    return facade._prefers_sideways(ds._plan, ctx)


def _joined(outer_rows: int, how: str = "left") -> bt.Dataset:
    agg = _fact().group_by("k").agg(s=bt.col("v").sum())
    return _outer(outer_rows).join(agg, left_on="ok", right_on="k", how=how)


@pytest.mark.parametrize("how", ["left", "inner", "semi", "anti"])
def test_a_small_probe_side_against_a_large_aggregate_is_sent(how) -> None:
    assert _verdict(_joined(1_000, how))


def test_a_probe_side_as_large_as_the_aggregate_input_is_not() -> None:
    """q18's shape: the probe side is the fact table itself, so nothing would be cut."""
    assert not _verdict(_joined(_FACT_ROWS))


def test_a_build_side_with_no_aggregate_is_not() -> None:
    assert not _verdict(_outer(1_000).join(_fact(), left_on="ok", right_on="k", how="left"))


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_a_semi_or_anti_build_needs_no_aggregate(how) -> None:
    """A semi/anti join builds its right side's key set, so the rows it reads are the cost."""
    assert _verdict(_outer(1_000).join(_fact(), left_on="ok", right_on="k", how=how))
