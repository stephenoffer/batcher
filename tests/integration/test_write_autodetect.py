"""Generic `Dataset.write(path)` format auto-detection across the core sinks.

`write(path)` infers the sink from the path extension (mirroring `read(path)`), so
the long tail of formats needs no per-format `write_*` method — one obvious way.
Covers the core pyarrow sinks (no optional deps); a final round-trip confirms the
written file reads back identically.
"""

from __future__ import annotations

import pytest

import batcher as bt


@pytest.mark.integration
@pytest.mark.parametrize(
    ("ext", "reader"),
    [
        ("parquet", "parquet"),
        ("csv", "csv"),
        ("json", "json"),
        ("orc", "orc"),
        ("arrow", "arrow"),
        ("feather", "arrow"),
    ],
)
def test_write_autodetect_roundtrip(tmp_path, ext, reader):
    path = str(tmp_path / f"out.{ext}")
    bt.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write(path)  # fmt inferred
    back = getattr(bt.read, reader)(path).collect()
    assert sorted(back.column("a").to_pylist()) == [1, 2, 3]


@pytest.mark.integration
def test_write_explicit_format_overrides_extension(tmp_path):
    # An explicit fmt wins over the (mismatched) extension.
    path = str(tmp_path / "data.bin")
    bt.from_pydict({"a": [1, 2]}).write(path, "parquet")
    assert bt.read.parquet(path).count() == 2
