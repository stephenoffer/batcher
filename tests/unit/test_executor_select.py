"""The Executor registry selects the same strategy the old dispatch chose.

`api.executors.select` replaced the if/elif/else in `terminal._collect`: distributed
when requested, else the UDF orchestrator for plans with `map_batches`, else the
single-node native engine. This pins that mapping so a future tier added to the
registry can't silently change which path a query takes.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.api.executors import (
    DistributedExecutor,
    LocalNativeExecutor,
    UdfExecutor,
    select,
)


def _plain_plan():
    return bt.from_pydict({"x": [1, 2, 3]}).filter(col("x") > 0)._plan


def _udf_plan():
    return bt.from_pydict({"x": [1, 2, 3]}).map_batches(lambda b: b)._plan


def test_plain_plan_selects_local_native():
    assert isinstance(select(_plain_plan(), distributed=False), LocalNativeExecutor)


def test_map_batches_plan_selects_udf():
    assert isinstance(select(_udf_plan(), distributed=False), UdfExecutor)


def test_distributed_flag_selects_distributed():
    # Distributed wins even over a map_batches plan (the prior dispatch order).
    assert isinstance(select(_udf_plan(), distributed=True), DistributedExecutor)
    assert isinstance(select(_plain_plan(), distributed=True), DistributedExecutor)
