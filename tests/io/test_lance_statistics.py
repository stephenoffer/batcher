"""Lance manifest metadata — the predicated-count correctness regression.

Lance answers `row_count()` from its manifest without a scan, and that count seeds an EXACT
`SourceStatistics` a terminal `count()` may be answered from. A `LanceSource` built with
`predicate=` bakes the filter into the source, so its row count is the *filtered*
cardinality — but `count_rows()` was called without the filter, so `count()` on a predicated
Lance source returned the whole dataset's size: an exact-looking answer that was simply wrong.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("lance")

import lance

from batcher.io.formats.structured.lance import LanceSource
from batcher.io.source import source_statistics

pytestmark = pytest.mark.unit


@pytest.fixture
def dataset(tmp_path):
    """A 100-row Lance dataset with a filterable integer column."""
    path = str(tmp_path / "ds")
    lance.write_dataset(pa.table({"x": list(range(100))}), path)
    return path


def test_unpredicated_row_count_is_the_whole_dataset(dataset) -> None:
    """Without a filter the count is the full dataset, exactly."""
    assert LanceSource(dataset).row_count() == 100


def test_predicated_row_count_is_filtered(dataset) -> None:
    """A `predicate=` source reports its filtered cardinality, not the whole dataset."""
    assert LanceSource(dataset, predicate="x < 10").row_count() == 10


def test_predicated_statistics_are_exact_and_filtered(dataset) -> None:
    """The exact stat a `count()` is answered from reflects the source's own filter."""
    stats = source_statistics(LanceSource(dataset, predicate="x < 25"))
    assert stats is not None
    assert stats.row_count == 25
    assert stats.exact_rows is True  # manifest counts are exact, so count() may use them
