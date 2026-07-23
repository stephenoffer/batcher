"""Point-cloud source coverage: KITTI `.bin`, PCD (ascii/binary), PLY (ascii/binary).

Exercises the reader across every supported on-disk layout, the appended ``frame``
column, format auto-detection, and that the columnar result feeds native engine
operators (a ground-plane filter, per-frame grouping) — the robotics preprocessing path.
"""

from __future__ import annotations

import numpy as np
import pytest

import batcher as bt
from batcher.io.formats.ml.point_cloud import PointCloudSource

_PTS = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.9]], dtype=np.float32)


def _write_bin(path: str) -> None:
    _PTS.tofile(path)


def _write_pcd(path: str, *, binary: bool) -> None:
    kind = "binary" if binary else "ascii"
    header = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        f"TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA {kind}\n"
    )
    if binary:
        with open(path, "wb") as fh:
            fh.write(header.encode())
            fh.write(_PTS.tobytes())
    else:
        rows = "\n".join(" ".join(str(v) for v in row) for row in _PTS)
        with open(path, "w") as fh:
            fh.write(header + rows + "\n")


def _write_ply(path: str, *, binary: bool) -> None:
    fmt = "binary_little_endian 1.0" if binary else "ascii 1.0"
    header = (
        f"ply\nformat {fmt}\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\nproperty float intensity\n"
        "end_header\n"
    )
    if binary:
        with open(path, "wb") as fh:
            fh.write(header.encode())
            fh.write(_PTS.tobytes())
    else:
        rows = "\n".join(" ".join(str(v) for v in row) for row in _PTS)
        with open(path, "w") as fh:
            fh.write(header + rows + "\n")


@pytest.mark.parametrize(
    ("suffix", "writer"),
    [
        (".bin", lambda p: _write_bin(p)),
        (".pcd", lambda p: _write_pcd(p, binary=False)),
        (".pcd", lambda p: _write_pcd(p, binary=True)),
        (".ply", lambda p: _write_ply(p, binary=False)),
        (".ply", lambda p: _write_ply(p, binary=True)),
    ],
)
def test_reads_every_layout(tmp_path, suffix, writer):
    """Each on-disk layout decodes to the same x/y/z/intensity points."""
    path = str(tmp_path / f"sweep{suffix}")
    writer(path)
    out = bt.read.point_cloud(path, frame_column=None).to_pydict()
    assert out["x"] == [1.0, 4.0]
    assert out["y"] == [2.0, 5.0]
    assert out["z"] == [3.0, 6.0]
    assert out["intensity"] == pytest.approx([0.5, 0.9])


def test_frame_column_names_the_source_file(tmp_path):
    """The appended ``frame`` column carries the file stem for per-sweep separation."""
    _write_bin(str(tmp_path / "000042.bin"))
    out = bt.read.point_cloud(str(tmp_path / "000042.bin")).to_pydict()
    assert out["frame"] == ["000042", "000042"]


def test_custom_bin_columns(tmp_path):
    """A raw ``.bin`` with a non-default field layout is read by the given columns."""
    path = str(tmp_path / "xyz.bin")
    np.array([[1.0, 2.0, 3.0]], dtype=np.float32).tofile(path)
    out = bt.read.point_cloud(path, columns=("x", "y", "z"), frame_column=None).to_pydict()
    assert out == {"x": [1.0], "y": [2.0], "z": [3.0]}


def test_autodetect_and_native_ops(tmp_path):
    """`.pcd`/`.ply` auto-detect, and the columnar cloud filters/groups in the engine."""
    _write_pcd(str(tmp_path / "a.pcd"), binary=True)
    assert bt.read(str(tmp_path / "a.pcd")).count() == 2  # auto-detected as point_cloud

    _write_bin(str(tmp_path / "f0.bin"))
    _write_bin(str(tmp_path / "f1.bin"))
    ds = bt.read.point_cloud(str(tmp_path / "*.bin"))
    assert ds.count() == 4  # two sweeps, two points each
    # Ground-plane removal is a native filter; z in {3, 6}, so z > 4 keeps half.
    assert ds.filter(bt.col("z") > 4.0).count() == 2
    # Per-sweep separation via the frame column.
    per_frame = ds.group_by("frame").agg(n=bt.col("x").count()).to_pydict()
    assert sorted(per_frame["n"]) == [2, 2]


def test_row_count_from_header_without_loading(tmp_path):
    """The exact point count comes from the header / file size, no point load."""
    _write_pcd(str(tmp_path / "a.pcd"), binary=True)
    src = PointCloudSource(str(tmp_path / "a.pcd"))
    assert src.row_count() == 2

    _write_bin(str(tmp_path / "b.bin"))
    assert PointCloudSource(str(tmp_path / "b.bin")).row_count() == 2
