"""A distributed scan that falls back from the native Parquet reader must leave a trace.

`_native_scan_batches` returns `None` when the native read fails, and the worker reads the
same row-groups through pyarrow instead. That fallback is right, but it is also
indistinguishable from "the native path never applied": a native reader broken on every
worker would cost the cluster its fastest read path with every result still correct. The
fallback stays; the suppressed error goes on the record.
"""

from __future__ import annotations

import pytest

from batcher._internal.logging import _FIELDS_ATTR
from batcher.dist.executors import scan_read
from batcher.io.formats.structured import _parquet_native
from batcher.io.splits import RowGroupSplit

pytestmark = pytest.mark.unit


def _noted(caplog, step: str) -> list[dict]:
    fields = [getattr(r, _FIELDS_ATTR, {}) for r in caplog.records]
    return [f for f in fields if f.get("step") == step]


def _splits():
    return [RowGroupSplit(path="/fake/f0.parquet", row_groups=[0])]


def test_a_failed_native_read_falls_back_and_is_recorded(monkeypatch, caplog) -> None:
    def boom(*_a):
        raise OSError("native reader exploded")

    monkeypatch.setattr(_parquet_native, "read_row_groups_filtered", boom)
    with caplog.at_level("DEBUG", logger="batcher.dist"):
        assert scan_read._native_scan_batches(_splits(), None) is None
    noted = _noted(caplog, "native parquet read")
    assert noted, f"the native-read fallback must be recorded\n{caplog.text}"
    assert noted[0]["error"] == "OSError"


def test_a_failed_batch_sizing_is_recorded(monkeypatch, caplog) -> None:
    def boom(*_a, **_k):
        raise ValueError("cannot size")

    monkeypatch.setattr(_parquet_native, "native_read_batch", boom)
    monkeypatch.setattr(_parquet_native, "read_row_groups_filtered", lambda *_a: [])
    with caplog.at_level("DEBUG", logger="batcher.dist"):
        out = scan_read._native_scan_batches(_splits(), None)
    assert out is not None and list(out) == []
    noted = _noted(caplog, "native parquet read batch sizing")
    assert noted, f"the batch-sizing fallback must be recorded\n{caplog.text}"
