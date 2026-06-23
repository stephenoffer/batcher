"""`Dataset.stats()` — measured per-operator metrics (the `explain()` companion).

Closes Ray Data's documented observability gap (no execution-plan display / weak
per-operator metrics, ray-project/ray#55052): after a run, every operator's
measured rows in/out, wall time, peak bytes, spill, and backend are available,
with a bottleneck classification.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("batcher._native", reason="native engine not built")

import batcher as bt
from batcher import col, count


def test_stats_reports_per_operator_metrics():
    tbl = pa.table({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
    ds = bt.from_arrow(tbl).filter(col("v") > 5).group_by("k").agg(s=col("v").sum(), n=count())
    st = ds.stats()

    kinds = {o.kind for o in st.ops}
    assert "scan" in kinds
    assert "aggregate" in kinds
    assert st.rows == 3  # three groups
    assert st.total_ms >= 0.0
    assert st.bottleneck is not None
    for o in st.ops:
        assert o.rows_in >= 0
        assert o.rows_out >= 0
        assert o.elapsed_ms >= 0.0

    text = str(st)
    assert "bottleneck" in text
    assert "aggregate" in text


def test_stats_rejects_map_batches():
    from batcher._internal.errors import BackendError

    tbl = pa.table({"x": [1, 2, 3]})
    ds = bt.from_arrow(tbl).ml.map_batches(lambda b: b, output_columns=["x"])
    with pytest.raises(BackendError, match="map_batches"):
        ds.stats()
