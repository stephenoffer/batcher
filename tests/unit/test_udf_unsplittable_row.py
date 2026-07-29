"""A row too wide for batching to bound must say so before the allocation that fails.

The field guides state this one unconditionally: "a single row larger than available task
memory causes an unrecoverable OOM regardless of any other tuning... a hard architectural
constraint, not a guideline", with ~10 MB as the safe ceiling.

Batcher's batching is byte-adaptive, so it shrinks the chunk as rows get wider — but that
works down to one row and then stops, because rows are atomic. Past that width the engine
has spent its last adaptive move and the remaining fix belongs to the data. Saying so at
that moment is the only warning that can still be acted on.
"""

from __future__ import annotations

import warnings

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PerformanceWarning
from batcher.core.udf import sizing

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_once_flag():
    """The warning fires once per process; each test needs a clean slate."""
    sizing._ROW_WIDTH_WARNED = False
    yield
    sizing._ROW_WIDTH_WARNED = False


def _wide(n_rows: int, mb: int) -> pa.Table:
    return pa.table({"blob": [b"\0" * (mb << 20)] * n_rows})


def test_a_row_wider_than_batching_can_bound_warns():
    with pytest.warns(PerformanceWarning, match="cannot be split"):
        bt.from_arrow(_wide(2, 80)).map_batches(lambda b: b).collect()


def test_the_warning_names_the_measured_width():
    """A generic "rows are large" is not actionable; the number is what tells the user
    whether they are near the edge or far past it."""
    with pytest.warns(PerformanceWarning, match="80 MB"):
        bt.from_arrow(_wide(2, 80)).map_batches(lambda b: b).collect()


def test_ordinary_rows_stay_silent():
    """Byte-adaptive batching handles wide-ish rows perfectly well — warning there would be
    noise on every multimodal pipeline that is working fine."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        bt.from_arrow(_wide(64, 1)).map_batches(lambda b: b).collect()


def test_it_warns_once_not_once_per_batch():
    """This is measured per morsel on the hot path; a corpus of huge rows would otherwise
    emit thousands of identical warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            bt.from_arrow(_wide(2, 80)).map_batches(lambda b: b).collect()
    assert sum("cannot be split" in str(c.message) for c in caught) == 1


def test_the_threshold_is_on_the_row_not_the_batch():
    """Many narrow rows summing to a large batch is the case batching *does* handle, and
    must not be confused with one unsplittable row."""
    assert sizing._ROW_WIDTH_WARNED is False
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        # 4096 rows x 64 KiB = 256 MiB of batch, but every row is tiny.
        bt.from_arrow(pa.table({"b": [b"\0" * (64 << 10)] * 4096})).map_batches(
            lambda b: b
        ).collect()
