"""Distributed fan-out must follow the *bytes* a task will hold, not only the rows.

`target_rows_per_task` is four million, which at the flat 64 B/row it was tuned against is
a sensible 256 MiB per task. With no width term it sizes tasks that cannot exist on any
machine for anything wider — 12 GB for an embedding column, 602 GB for a decoded image,
25 TB for 1080p frames — so a multimodal pipeline was fanned out as though every row were
sixteen bytes. That is an OOM that only appears at scale.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.config import active_config
from batcher.kyber.annotate import _desired_parallelism, annotate_ops
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import CostModel

pytestmark = pytest.mark.unit

_TARGET_ROWS = active_config().optimizer.target_rows_per_task
_TARGET_BYTES = active_config().optimizer.target_bytes_per_task
_ROW_BYTES = active_config().optimizer.row_bytes


def _tasks(rows: float, width: float) -> int:
    return _desired_parallelism(rows, width, _TARGET_ROWS, _TARGET_BYTES)


def test_a_narrow_relation_gets_exactly_the_fanout_it_always_got():
    # The safety property. The byte term is combined with `max`, never `min`, so it can only
    # ask for *more* parallelism — a relation no wider than the flat default is untouched,
    # and no structured plan is re-shaped by this.
    for rows in (1.0, 4e6, 4e7, 1e11):
        assert _tasks(rows, 8.0) == _tasks(rows, _ROW_BYTES)
        assert _tasks(rows, _ROW_BYTES) == max(1, -(-int(rows) // _TARGET_ROWS))


@pytest.mark.parametrize(
    ("label", "width"),
    [
        ("embedding f32x768", 768 * 4.0),
        ("image 224x224x3", 224.0 * 224 * 3),
        ("1080p RGB frame", 1920.0 * 1080 * 3),
    ],
)
def test_a_wide_relation_is_split_by_its_bytes(label, width):
    # Every wide shape lands on the same per-task byte budget the narrow default implies,
    # rather than on a task holding hundreds of gigabytes.
    rows = 4e7
    tasks = _tasks(rows, width)
    per_task = rows * width / tasks
    assert per_task <= _TARGET_BYTES * 1.01, label


def test_fanout_grows_with_width():
    rows = 1e7
    narrow = _tasks(rows, 16.0)
    embedding = _tasks(rows, 3072.0)
    image = _tasks(rows, 150_528.0)
    assert narrow < embedding < image


def test_an_empty_or_zero_width_relation_still_asks_for_one_task():
    assert _tasks(0.0, 0.0) == 1
    assert _tasks(1.0, 0.0) == 1


def test_kyber_agrees_with_the_partition_sizer_it_documents():
    # The two answers to "how many tasks" must be the same rule. `auto_num_partitions` and
    # `dist/executors/map.py` have always taken the larger of the row- and byte-derived
    # counts; Kyber's `n_max_parallelism` counted only rows.
    import math

    rows, width = 4e7, 150_528.0
    row_parts = math.ceil(rows / _TARGET_ROWS)
    byte_parts = math.ceil(rows * width / _TARGET_BYTES)
    assert _tasks(rows, width) == max(row_parts, byte_parts)


def test_the_annotated_plan_carries_the_wider_fanout():
    # End to end through `annotate_ops`: the same sort over a narrow column and over a
    # tensor column must not request the same number of workers. Run against a small byte
    # target so the shapes separate without allocating a gigabyte of test data.
    import dataclasses

    rows = 4096
    cfg = active_config()
    cfg = cfg.replace(optimizer=dataclasses.replace(cfg.optimizer, target_bytes_per_task=1 << 16))
    narrow = bt.from_pydict({"k": list(range(rows)), "v": list(range(rows))})
    tensor = bt.from_arrow(
        pa.table(
            {
                "k": pa.array(range(rows)),
                "t": pa.FixedShapeTensorArray.from_numpy_ndarray(
                    np.zeros((rows, 64, 64, 3), dtype="uint8")
                ),
            }
        )
    )

    def fanout(ds):
        est = CardinalityEstimator(ds._sources)
        ops = annotate_ops(ds._plan, est, cfg, CostModel(est))
        return max(o.bounds.n_max_parallelism for o in ops)

    assert fanout(tensor.sort("k")) > fanout(narrow.sort("k"))
