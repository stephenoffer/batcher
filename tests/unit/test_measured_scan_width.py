"""A source that measured its own bytes is more authoritative than any type prior.

`SourceStatistics` has always carried `byte_size` alongside `row_count`, and three
consumers read it — the storage shortcut, the read-cost predictor, the distributed map
sizer. The cardinality estimator's `row_width`, which is the single number under every byte
axis in the engine, did not.

That is the whole gap on unstructured and multimodal data. `io/formats/multimodal/media.py`
reports the exact total size and file count from its listing, so a directory of 200 MB
videos is a *measured* 200 MB per row, while `column_bytes` could only offer the 36-byte
prior for the `binary` column it lands in.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.plan.source_stats import SourceStatistics

pytestmark = pytest.mark.unit


def _blobs(rows: int = 10):
    return bt.from_arrow(pa.table({"blob": pa.array([b"x"] * rows, type=pa.binary())}))


def _width(ds, stats: SourceStatistics | None) -> float:
    est = CardinalityEstimator(ds._sources, source_stats=[stats] if stats else None)
    return est.row_width(ds._plan, 64.0)


def _content(**kw) -> SourceStatistics:
    """Statistics from a source whose rows *are* files — a media or text listing."""
    return SourceStatistics(content_byte_size=True, **kw)


def test_a_measured_width_beats_the_type_prior():
    # The multimodal case: a listing that measured 200 MB per row against a 36-byte prior
    # for the binary column those bytes land in.
    ds = _blobs()
    measured = _content(row_count=10, byte_size=2_000_000_000, exact_rows=True)
    assert _width(ds, measured) == pytest.approx(200_000_000.0)
    assert _width(ds, None) < 100.0


def test_the_type_prior_wins_when_it_is_larger():
    # A floor, not a replacement. A footer's `total_byte_size` is the *stored* size, which
    # for a dictionary-encoded column can sit below the materialized Arrow width the type
    # implies — so the two are combined with `max` and neither may suppress the other.
    ds = _blobs()
    tiny = _content(row_count=10, byte_size=80, exact_rows=True)
    assert _width(ds, tiny) == _width(ds, None)


def test_a_source_reporting_nothing_is_unchanged():
    ds = _blobs()
    assert _width(ds, _content(row_count=10)) == _width(ds, None)
    assert _width(ds, _content(byte_size=1_000)) == _width(ds, None)
    assert _width(ds, _content(row_count=0, byte_size=1_000)) == _width(ds, None)


def test_it_applies_only_at_the_scan():
    # Above a scan the projected columns differ, and the per-column widths propagated on
    # `RelStats` are the right mechanism. Attributing a whole relation's bytes to one narrow
    # projected column would invert the estimate rather than sharpen it.
    ds = bt.from_arrow(
        pa.table({"blob": pa.array([b"x"] * 10, type=pa.binary()), "k": pa.array(range(10))})
    )
    measured = _content(row_count=10, byte_size=2_000_000_000, exact_rows=True)
    est = CardinalityEstimator(ds._sources, source_stats=[measured])
    scan = est.row_width(ds._plan, 64.0)
    projected = est.row_width(ds.select(bt.col("k"))._plan, 64.0)
    assert scan == pytest.approx(200_000_000.0)
    assert projected < 100.0


def test_a_columnar_source_is_deliberately_not_trusted():
    # The safety property, and it was earned by measurement rather than assumed. A Parquet
    # footer's `total_byte_size` is the encoded, row-group-padded *stored* size, and taking
    # it as a floor as well moved TPC-H sf1's width from 88 to 142 B/row -- closer to the
    # true 139 -- while making the benchmark WORSE: dimension build sides crossed the
    # broadcast threshold and q9 went from 55.8 ms to 127.9, with ten other queries slower.
    # A sharper estimate against a threshold tuned for the blunter one is a re-tuning.
    ds = _blobs()
    columnar = SourceStatistics(row_count=10, byte_size=2_000_000_000, exact_rows=True)
    assert columnar.content_byte_size is False
    assert _width(ds, columnar) == _width(ds, None)


def test_the_file_listing_connectors_declare_it():
    # The flag is only useful if the sources whose rows *are* files actually set it.
    import inspect

    from batcher.io.formats.multimodal import media
    from batcher.io.formats.unstructured import text

    for mod in (media, text):
        assert "content_byte_size=True" in inspect.getsource(mod)
