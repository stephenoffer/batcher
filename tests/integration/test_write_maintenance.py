"""Phase 6 — backfill via dynamic range overwrite (`write(replace_where=...)`)."""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def test_replace_where_backfill(tmp_path):
    tgt = f"{tmp_path}/events.parquet"
    bt.from_arrow(
        pa.table({"dt": ["2024-01-01", "2024-01-01", "2024-01-02"], "v": [1, 2, 3]})
    ).write.parquet(tgt)
    # Reprocess only 2024-01-01: replace just those rows, keep 2024-01-02.
    bt.from_arrow(pa.table({"dt": ["2024-01-01"], "v": [99]})).write(
        tgt, "parquet", replace_where=col("dt") == "2024-01-01"
    )
    out = bt.read.parquet(tgt).collect().to_pydict()
    rows = sorted(zip(out["dt"], out["v"], strict=True))
    assert rows == [("2024-01-01", 99), ("2024-01-02", 3)]


def test_replace_where_into_missing_target_writes_all(tmp_path):
    tgt = f"{tmp_path}/new.parquet"
    bt.from_arrow(pa.table({"dt": ["2024-01-01"], "v": [1]})).write(
        tgt, "parquet", replace_where=col("dt") == "2024-01-01"
    )
    assert bt.read.parquet(tgt).collect().num_rows == 1
