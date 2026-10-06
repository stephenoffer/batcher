"""`resume=True` is refused on a sink with no resume contract (AP-439).

The Delta and Iceberg sinks took `resume` for the shared `FileSink.write` signature and
ignored it, so `write.delta(path, mode="append", resume=True)` run twice committed every
row twice: four rows from two, under a flag documented as making the write idempotent.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError


def _ds() -> bt.Dataset:
    return bt.from_pydict({"x": [1, 2]})


def test_delta_append_with_resume_is_refused_before_any_commit(tmp_path):
    pytest.importorskip("deltalake")
    path = str(tmp_path / "t")
    _ds().write.delta(path)  # the table exists; the refusal is not about a missing path
    with pytest.raises(PlanError, match="for 'delta': the write is a single atomic"):
        _ds().write.delta(path, mode="append", resume=True)
    # Nothing was committed by the refused call.
    assert bt.read.delta(path).count() == 2


@pytest.mark.parametrize("fmt", ["delta", "iceberg"])
def test_table_sinks_refuse_resume_through_the_generic_writer(tmp_path, fmt):
    with pytest.raises(PlanError, match=f"not supported for {fmt!r}"):
        _ds().write(str(tmp_path / "t"), fmt, mode="append", resume=True)


def test_file_sinks_still_resume(tmp_path):
    # The control: the refusal is scoped to the table sinks, not to `resume` itself.
    path = str(tmp_path / "out")
    _ds().write.parquet(path)
    _ds().write.parquet(path, resume=True)
    assert bt.read.parquet(path).count() == 2
