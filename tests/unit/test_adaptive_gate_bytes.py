"""The adaptive size floor is about *work*, and a row count assumes a row width.

`_ADAPTIVE_MIN_INPUT_ROWS` exists because stage-by-stage re-optimization trades a ~20-40 ms
re-plan for a better downstream join choice, which only pays once a mis-estimated plan would
cost more than that. Rows are a proxy for work, and the proxy holds only while a row is the
~64 bytes `optimizer.row_bytes` assumes. Across the modality range it inverts at both ends:
20M rows of two `int64` keys is 320 MB and cleared the floor, while 1M rows of decoded
224x224x3 images is 150 GB and did not — so the most expensive query class in the engine was
the one class the adaptive loop never ran on.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher.api.adaptive.gating import (
    _ADAPTIVE_MIN_INPUT_BYTES,
    _ADAPTIVE_MIN_INPUT_ROWS,
    _build_estimator,
    _large_enough,
    _total_input_size,
)

pytestmark = pytest.mark.unit


def _narrow_join(rows: int = 1000):
    left = bt.from_pydict({"k": list(range(rows)), "v": list(range(rows))})
    right = bt.from_pydict({"k": list(range(rows)), "w": list(range(rows))})
    return left.join(right, on="k")


def _tensor_join(rows: int = 8, shape=(224, 224, 3)):
    arr = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((rows, *shape), dtype="uint8"))
    left = bt.from_arrow(pa.table({"k": pa.array(range(rows)), "img": arr}))
    right = bt.from_pydict({"k": list(range(rows)), "w": list(range(rows))})
    return left.join(right, on="k")


def test_the_two_floors_describe_the_same_size():
    # Derived from the existing knobs rather than added as a third, so there is one place
    # that says how big "big" is.
    assert _ADAPTIVE_MIN_INPUT_BYTES == _ADAPTIVE_MIN_INPUT_ROWS * 64


def test_a_small_narrow_query_still_does_not_qualify():
    # The safety property: nothing that used the one-shot route may be moved off it.
    ds = _narrow_join()
    assert _large_enough(ds._plan, ds._sources, None) is False


def test_the_size_probe_reports_bytes_as_well_as_rows():
    ds = _tensor_join(rows=8)
    estimator = _build_estimator(ds._sources, None)
    rows, nbytes = _total_input_size(ds._plan, estimator)
    assert rows > 0
    # The image scan dominates: 8 rows at 147 KiB each is far more than 8 rows at 64 B.
    assert nbytes > rows * 1000


def test_a_wide_query_clears_the_floor_on_bytes_alone():
    # 224x224x3 uint8 is 147 KiB per row, so it takes about 8,900 rows to reach the byte
    # floor -- three orders of magnitude below the row floor, which is the whole point.
    needed = int(_ADAPTIVE_MIN_INPUT_BYTES / (224 * 224 * 3)) + 64
    ds = _tensor_join(rows=needed)
    estimator = _build_estimator(ds._sources, None)
    rows, nbytes = _total_input_size(ds._plan, estimator)
    assert rows < _ADAPTIVE_MIN_INPUT_ROWS  # nowhere near the row floor
    assert nbytes >= _ADAPTIVE_MIN_INPUT_BYTES
    assert _large_enough(ds._plan, ds._sources, None) is True


def test_a_join_is_still_required():
    # The size floor is one condition among several; a joinless plan never qualifies
    # however wide it is.
    arr = pa.FixedShapeTensorArray.from_numpy_ndarray(np.zeros((16, 224, 224, 3), dtype="uint8"))
    ds = bt.from_arrow(pa.table({"img": arr}))
    assert _large_enough(ds._plan, ds._sources, None) is False
