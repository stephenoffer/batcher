"""A point cloud's schema comes from its header, not from parsing every point.

`ds.schema` used to parse the whole file, build the whole Arrow table, and keep the field
names. On a 2-million-point LiDAR sweep that is **7.9 seconds and 59 MB** to answer a
question the header already contained, and an autonomous-driving corpus is thousands of
sweeps in one directory.

The risk in fixing it is that the two paths disagree: a header schema that reports what the
file *declares* rather than what the reader *produces* is worse than the slow one, because
every later stage is planned against a lie. PCD and PLY both read an `ascii` body through
`numpy.loadtxt(dtype=float64)`, so an ascii file's columns come back Float64 whatever its
`TYPE` line says. So the test that matters is agreement, on every format and both
encodings, and it is parametrized rather than sampled for exactly that reason.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("numpy")

import numpy as np

from batcher.io.formats.ml.point_cloud import PointCloudSource

pytestmark = pytest.mark.unit

_PCD_HEADER = (
    "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
    "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA "
)
_PLY_PROPS = "element vertex 2\nproperty float x\nproperty float y\nproperty uchar red\n"


def _pcd_ascii(d: pathlib.Path) -> pathlib.Path:
    p = d / "a.pcd"
    p.write_bytes((_PCD_HEADER + "ascii\n1 2 3 4\n5 6 7 8\n9 1 2 3\n").encode())
    return p


def _pcd_binary(d: pathlib.Path) -> pathlib.Path:
    p = d / "b.pcd"
    p.write_bytes((_PCD_HEADER + "binary\n").encode() + np.arange(12, dtype="<f4").tobytes())
    return p


def _ply_ascii(d: pathlib.Path) -> pathlib.Path:
    p = d / "a.ply"
    header = f"ply\nformat ascii 1.0\n{_PLY_PROPS}end_header\n"
    p.write_bytes((header + "1 2 3\n4 5 6\n").encode())
    return p


def _ply_binary(d: pathlib.Path) -> pathlib.Path:
    p = d / "b.ply"
    header = f"ply\nformat binary_little_endian 1.0\n{_PLY_PROPS}end_header\n"
    rows = np.array(
        [(1.0, 2.0, 3), (4.0, 5.0, 6)], dtype=[("x", "<f4"), ("y", "<f4"), ("red", "u1")]
    )
    p.write_bytes(header.encode() + rows.tobytes())
    return p


def _bin(d: pathlib.Path) -> pathlib.Path:
    p = d / "sweep.bin"
    p.write_bytes(np.arange(12, dtype="float32").tobytes())
    return p


@pytest.mark.parametrize(
    "build", [_pcd_ascii, _pcd_binary, _ply_ascii, _ply_binary, _bin], ids=lambda f: f.__name__
)
def test_the_header_schema_equals_the_parsed_one(tmp_path, build):
    """The whole safety property. A faster schema that differs is not a faster schema."""
    path = build(tmp_path)
    source = PointCloudSource(str(tmp_path), frame_column=None)

    with open(path, "rb") as fh:
        from_header = source._read_schema(fh)
    with open(path, "rb") as fh:
        from_parse = source._parse(fh.read(), None).schema

    assert from_header.equals(from_parse)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_raw_bin_schema_follows_the_declared_layout(tmp_path, dtype):
    """A raw buffer has no header, so its schema is the caller's `columns`/`dtype`.

    Which means it costs no file access at all — the case that matters most, since KITTI
    `.bin` is the format an autonomous-driving corpus is actually stored in.
    """
    (tmp_path / "s.bin").write_bytes(np.arange(12, dtype=dtype).tobytes())
    source = PointCloudSource(str(tmp_path), dtype=dtype, columns=("a", "b", "c", "d"))
    with open(tmp_path / "s.bin", "rb") as fh:
        schema = source._read_schema(fh)

    assert schema.names == ["a", "b", "c", "d", "frame"]


def test_the_frame_column_survives_the_header_path(tmp_path):
    """The reads append `frame`; a schema that omitted it would drop the column."""
    _pcd_binary(tmp_path)
    source = PointCloudSource(str(tmp_path))
    with open(tmp_path / "b.pcd", "rb") as fh:
        assert source._read_schema(fh).names == ["x", "y", "z", "intensity", "frame"]


def test_a_header_that_does_not_fit_falls_through_to_the_real_parser(tmp_path):
    """Slow beats wrong: an unparseable header must not become a guessed schema.

    A PCD with no `DATA` line is not a PCD, and the full parser says so with a message
    naming the problem — which a header reader inventing `FIELDS x y` would have hidden.
    """
    from batcher._internal.errors import BackendError

    (tmp_path / "t.pcd").write_bytes(b"# .PCD v0.7\nVERSION 0.7\nFIELDS x y\n")
    source = PointCloudSource(str(tmp_path))

    with pytest.raises(BackendError, match="DATA"), open(tmp_path / "t.pcd", "rb") as fh:
        source._read_schema(fh)


def test_the_schema_read_does_not_touch_the_points(tmp_path):
    """The point of the change, asserted on bytes rather than on a stopwatch.

    A timing assertion would be flaky on a shared box; how much of the file was read is
    exact. The header here is ~200 bytes against a 1.6 MB body.
    """
    header = (_PCD_HEADER + "binary\n").encode()
    body = np.zeros(400_000, dtype="<f4").tobytes()
    (tmp_path / "big.pcd").write_bytes(header + body)
    source = PointCloudSource(str(tmp_path), frame_column=None)

    with open(tmp_path / "big.pcd", "rb") as fh:
        source._read_schema(fh)
        consumed = fh.tell()

    assert consumed < len(header) + 65_536, f"read {consumed} bytes for a {len(header)}-byte header"
    assert consumed < len(body) / 4
