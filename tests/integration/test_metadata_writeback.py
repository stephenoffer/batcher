"""Write-back loop: a Batcher-written dataset re-reads from metadata.

A Parquet round-trip is answered metadata-only on re-read via the footer; a
footerless CSV round-trip persists its stats into the MetadataHub so cardinality
is still informed without a re-scan. The persisted stats round-trip through the
store correctly.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.metadata.source_stats_store import load_source_stats, save_source_stats
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance

pytest.importorskip("batcher._native", reason="native engine not built")


def test_parquet_roundtrip_count_is_metadata_only(tmp_path):
    path = str(tmp_path / "out.parquet")
    bt.from_pydict({"x": list(range(500)), "g": [i % 9 for i in range(500)]}).write.parquet(path)
    reread = bt.read.parquet(path)
    # Footer gives an exact count with no scan; the value must be correct.
    assert reread.count() == 500
    assert reread.agg(mx=bt.col("x").max()).to_pydict() == {"mx": [499]}


def test_csv_roundtrip_persists_stats(tmp_path):
    from batcher import core

    path = str(tmp_path / "out.csv")
    bt.from_pydict({"x": list(range(40)), "k": [i % 4 for i in range(40)]}).write.csv(path)
    # The write persisted advisory stats under the read identity for this path.
    cached = load_source_stats(core.default_hub(), f"csv:{path}")
    assert cached is not None
    assert cached.row_count == 40
    assert "k" in cached.columns  # distinct-count estimate captured
    # The CSV itself still re-reads correctly (count executes — cached stats advisory).
    assert bt.read.csv(path).count() == 40


def test_source_stats_store_roundtrip(tmp_path):
    from batcher.metadata.backends.in_process import InProcessBackend
    from batcher.metadata.hub import MetadataHub

    hub = MetadataHub(InProcessBackend())
    stats = SourceStatistics(
        row_count=100,
        byte_size=4096,
        columns={
            "a": ColumnStat(min=0, max=99, null_count=0, ndv=100, provenance=Provenance.EXACT),
            "b": ColumnStat(ndv=7, provenance=Provenance.SKETCH),
        },
        exact_rows=True,
    )
    save_source_stats(hub, "parquet:/x", stats)
    loaded = load_source_stats(hub, "parquet:/x")
    assert loaded.row_count == 100 and loaded.byte_size == 4096
    assert loaded.columns["a"].max == 99 and loaded.columns["a"].provenance is Provenance.EXACT
    assert loaded.columns["b"].ndv == 7 and loaded.columns["b"].provenance is Provenance.SKETCH
    assert load_source_stats(hub, "parquet:/missing") is None
