"""Point-cloud sources — LiDAR / depth sensor frames as Arrow columns.

The native on-disk point-cloud formats robotics and autonomous-driving pipelines use,
read with no third-party dependency (pure Python + NumPy):

* **KITTI / raw ``.bin``** — a flat ``float32`` buffer reshaped to ``(N, C)``; the columns
  (default ``x, y, z, intensity`` — the KITTI Velodyne layout) are given by the caller
  because a raw buffer carries no schema.
* **PCD** (Point Cloud Data, the PCL / ROS format) — an ASCII header (``FIELDS`` / ``SIZE``
  / ``TYPE`` / ``POINTS`` / ``DATA``) over ``ascii`` or ``binary`` point data.
* **PLY** (Polygon File Format) — an ASCII header over ``ascii`` / ``binary_little_endian``
  / ``binary_big_endian`` vertex data; only the ``vertex`` element is read (faces, if any,
  are ignored — a point cloud is its vertices).

Each file is one frame; every point becomes a row, and each field becomes a column
(``x``/``y``/``z``/``intensity``/…), plus a ``frame`` column naming the source file so a
directory of sweeps stays separable (``group_by("frame")`` reconstitutes clouds). This
columnar layout is exactly what the engine wants: cropping a region, removing the ground
plane (``filter(col("z") > -1.5)``), or binning into voxels are then native operators, and
files are read concurrently like any `FileSource`.
"""

from __future__ import annotations

import os
from typing import IO, TYPE_CHECKING, Any, ClassVar

import pyarrow as pa

from batcher._internal.errors import BackendError
from batcher.io.base import FileSource
from batcher.io.formats.base import SOURCES

if TYPE_CHECKING:
    # Annotation-only: `dict[str, np.ndarray]` return types resolve for type checkers
    # without importing numpy at runtime (see `_np` for the deferred runtime import).
    import numpy as np

__all__ = ["PointCloudSource"]


def _np() -> Any:
    # Deferred so `import batcher` (which eagerly imports every IO format to self-register)
    # never pulls numpy. numpy is not a core dependency, and a module-scope import here made
    # the whole package — and the docs autodoc build that imports it — fail without numpy
    # installed. Matches the sibling `numpy` format's accessor.
    try:
        import numpy as np
    except ImportError as exc:  # numpy is near-ubiquitous but kept optional/deferred
        raise BackendError("reading point-cloud files needs numpy: pip install numpy") from exc
    return np


_SUFFIXES = (".bin", ".pcd", ".ply")
_FRAME = "frame"

# How much of a file to read when inferring its schema. PCD and PLY headers are a few
# hundred bytes; this is generous enough for a PLY declaring dozens of properties and
# still nothing next to the sweep behind it.
_HEADER_PEEK = 64 << 10

# PCD/PLY scalar type -> NumPy dtype. Keyed by (kind, size) for PCD, by name for PLY.
_PCD_TYPES = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
    ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
}  # fmt: skip
_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}  # fmt: skip


@SOURCES.register("point_cloud")
class PointCloudSource(FileSource):
    """One or more point-cloud files (``.bin`` / ``.pcd`` / ``.ply``) as point rows.

    Args:
        path: a point-cloud file, directory, or glob.
        columns: field names for a raw ``.bin`` file (ignored for self-describing PCD/PLY),
            in buffer order. The count also sets the stride (``float32`` values per point).
        dtype: the NumPy dtype of a raw ``.bin`` buffer (default ``float32``).
        frame_column: name of the appended source-file column (``None`` to omit it).
    """

    suffix: ClassVar[str] = ".pcd"  # `_files` widens this to every point-cloud suffix
    format_name: ClassVar[str] = "point_cloud"

    __slots__ = ("_bin_cols", "_bin_dtype", "_frame_col")

    def __init__(
        self,
        path: str,
        *,
        columns: tuple[str, ...] | list[str] = ("x", "y", "z", "intensity"),
        dtype: str = "float32",
        frame_column: str | None = _FRAME,
        **kwargs: Any,
    ) -> None:
        # Forward the base options. `schema_mode` was forwarded but `on_error` was not,
        # so a corrupt sweep in a directory of thousands still failed the whole read.
        super().__init__(path, **kwargs)
        self._bin_cols = tuple(columns)
        self._bin_dtype = dtype
        self._frame_col = frame_column

    def _reader_kwargs(self) -> dict[str, object]:
        # A raw `.bin` point cloud has no self-describing header, so a worker that rebuilds the
        # reader with the default `columns`/`dtype`/`frame_column` re-strides the bytes wrong and
        # silently returns different data than single-node. Carry the layout to the worker.
        return {
            **super()._reader_kwargs(),
            "columns": self._bin_cols,
            "dtype": self._bin_dtype,
            "frame_column": self._frame_col,
        }

    def _files(self) -> list[str]:
        if self._files_cache is None:
            self._files_cache = self._fs.expand(self._path, suffix=_SUFFIXES)
        return self._files_cache

    # ---- reads (path-aware, so the `frame` column and format dispatch work) ----
    def _read_by_path(self, path: str, projection: list[str] | None) -> list[pa.RecordBatch]:
        with self._fs.open(path) as handle:
            table = self._parse(handle.read(), path)
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def _iter_file(self, path: str, projection: list[str] | None):
        yield from self._read_by_path(path, projection)

    def _read_file(self, fh: IO[Any], projection: list[str] | None) -> list[pa.RecordBatch]:
        # Fallback (no path → no frame id); the path-aware routes above are the norm.
        table = self._parse(fh.read(), None)
        if projection is not None:
            table = table.select(projection)
        return table.to_batches()

    def _read_schema(self, fh: IO[Any]) -> pa.Schema:
        """The column names and types, from the file's header rather than its points.

        This used to parse the whole file and keep only the field names. A point cloud is
        the one place that is ruinous rather than merely wasteful: a single 2-million-point
        LiDAR sweep took **7.9 seconds and 59 MB** to answer `ds.schema`, and an
        autonomous-driving corpus is thousands of sweeps in a directory. The names were
        knowable the whole time — PCD and PLY both declare their fields in an ASCII header,
        and a raw `.bin` has no header at all because the caller supplied the layout.

        Falls back to the full parse when the header does not fit in the peek, so an
        unusual file is slow rather than wrong.

        Args:
            fh: An open binary handle positioned at the start of the file.

        Returns:
            The Arrow schema of the rows this source produces for that file.
        """
        head = fh.read(_HEADER_PEEK)
        schema = self._header_schema(head)
        if schema is None:
            # `head` is already consumed, so continue from where the peek stopped rather
            # than re-reading it — seeking is not available on every filesystem handle.
            schema = self._parse(head + fh.read(), None).schema
        # The per-file reads append a `frame` column (see `_parse`); the schema, parsed
        # without a path, must carry it too or normalization would drop it.
        if self._frame_col is not None:
            schema = schema.append(pa.field(self._frame_col, pa.string()))
        return schema

    def _header_schema(self, head: bytes) -> pa.Schema | None:
        """The schema from `head` alone, or `None` when the header needs more bytes."""
        kind = _sniff(head)
        if kind == ".bin":
            # A raw buffer carries no header, so the layout is the constructor's — which
            # means the schema of a `.bin` sweep costs no file access at all.
            dtype = _arrow_of(self._bin_dtype)
            return pa.schema([pa.field(name, dtype) for name in self._bin_cols])
        fields = _pcd_header_fields(head) if kind == ".pcd" else _ply_header_fields(head)
        return None if fields is None else pa.schema(fields)

    def _file_row_count(self, path: str) -> int | None:
        """Exact point count from the file header (or size), without loading points."""
        try:
            with self._fs.open(path) as handle:
                head = handle.read(4096)
                ext = os.path.splitext(path)[1].lower()
                if ext == ".pcd":
                    return _pcd_point_count(head)
                if ext == ".ply":
                    return _ply_point_count(head)
                size = self._fs.size(path)
            stride = len(self._bin_cols) * _np().dtype(self._bin_dtype).itemsize
            return size // stride if stride else None
        except Exception:
            return None

    def _parse(self, data: bytes, path: str | None) -> pa.Table:
        ext = os.path.splitext(path)[1].lower() if path else _sniff(data)
        if ext == ".ply":
            columns = _parse_ply(data)
        elif ext == ".pcd":
            columns = _parse_pcd(data)
        else:  # raw .bin (KITTI Velodyne and friends)
            columns = _parse_bin(data, self._bin_cols, self._bin_dtype)
        arrays = {name: pa.array(values) for name, values in columns.items()}
        if self._frame_col is not None and path is not None:
            frame = os.path.splitext(os.path.basename(path))[0]
            n = len(next(iter(columns.values()))) if columns else 0
            arrays[self._frame_col] = pa.array([frame] * n, type=pa.string())
        return pa.table(arrays)


def _sniff(data: bytes) -> str:
    """Detect a format from the file's leading bytes when the path is unavailable."""
    head = data[:64].lstrip()
    if head.startswith(b"ply"):
        return ".ply"
    if head.startswith(b"#") or head.startswith(b"VERSION") or b".PCD" in data[:64]:
        return ".pcd"
    return ".bin"


def _parse_bin(data: bytes, columns: tuple[str, ...], dtype: str) -> dict[str, np.ndarray]:
    """Reshape a raw ``(N, len(columns))`` buffer into one array per column."""
    np = _np()
    stride = len(columns)
    flat = np.frombuffer(data, dtype=dtype)
    if stride == 0 or flat.size % stride:
        raise BackendError(
            f"raw point-cloud buffer of {flat.size} values is not divisible by "
            f"{stride} columns {columns}; pass the right `columns=` for this .bin layout"
        )
    points = flat.reshape(-1, stride)
    return {name: points[:, i] for i, name in enumerate(columns)}


def _parse_pcd(data: bytes) -> dict[str, np.ndarray]:
    """Parse a PCD file (ASCII or binary DATA) into one array per FIELD."""
    np = _np()
    header_end = data.find(b"DATA")
    if header_end < 0:
        raise BackendError("not a PCD file: no DATA line in the header")
    line_end = data.find(b"\n", header_end)
    header = data[:line_end].decode("ascii", "replace")
    body = data[line_end + 1 :]

    fields: dict[str, str] = {}
    for line in header.splitlines():
        parts = line.split()
        if parts:
            fields[parts[0].upper()] = " ".join(parts[1:])
    names = fields["FIELDS"].split()
    sizes = [int(s) for s in fields["SIZE"].split()]
    types = fields["TYPE"].split()
    counts = [int(c) for c in fields.get("COUNT", " ".join("1" * len(names))).split()]
    n_points = int(fields["POINTS"]) if "POINTS" in fields else int(fields["WIDTH"])
    data_kind = fields["DATA"].strip().lower()

    if any(c != 1 for c in counts):
        raise BackendError("PCD fields with COUNT > 1 are not supported")
    if data_kind == "ascii":
        table = np.loadtxt(body.splitlines(), dtype=np.float64, ndmin=2)
        return {name: table[:, i] for i, name in enumerate(names)}
    if data_kind == "binary":
        dt = np.dtype(
            [
                (name, _PCD_TYPES[(kind, size)])
                for name, kind, size in zip(names, types, sizes, strict=True)
            ]
        )
        rec = np.frombuffer(body[: n_points * dt.itemsize], dtype=dt)
        return {name: rec[name] for name in names}
    raise BackendError(
        f"PCD DATA '{data_kind}' is not supported (use ascii or binary; "
        "binary_compressed needs an LZF decoder Batcher does not bundle)"
    )


def _parse_ply(data: bytes) -> dict[str, np.ndarray]:
    """Parse a PLY file (ASCII or binary vertex data) into one array per property."""
    np = _np()
    marker = b"end_header\n"
    header_end = data.find(marker)
    if not data.lstrip().startswith(b"ply") or header_end < 0:
        raise BackendError("not a PLY file: missing 'ply' magic or end_header")
    header = data[:header_end].decode("ascii", "replace")
    body = data[header_end + len(marker) :]

    fmt = "ascii"
    n_vertices = 0
    props: list[tuple[str, str]] = []  # (numpy_dtype, name)
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                n_vertices = int(parts[2])
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list":
                raise BackendError("PLY list properties (faces) are not point-cloud data")
            props.append((_PLY_TYPES[parts[1]], parts[2]))
    names = [name for _, name in props]

    if fmt == "ascii":
        rows = [ln.split() for ln in body.decode("ascii", "replace").splitlines() if ln.strip()]
        table = np.array(rows[:n_vertices], dtype=np.float64)
        return {name: table[:, i] for i, name in enumerate(names)}
    endian = "<" if "little" in fmt else ">"
    dt = np.dtype([(name, endian + t) for t, name in props])
    rec = np.frombuffer(body[: n_vertices * dt.itemsize], dtype=dt)
    return {name: rec[name] for name in names}


def _pcd_point_count(head: bytes) -> int | None:
    for line in head.decode("ascii", "replace").splitlines():
        if line.upper().startswith("POINTS"):
            return int(line.split()[1])
    return None


def _ply_point_count(head: bytes) -> int | None:
    text = head.decode("ascii", "replace")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            return int(parts[2])
    return None


def _arrow_of(numpy_dtype: str) -> pa.DataType:
    """The Arrow type `pa.array` produces for a NumPy dtype string."""
    np = _np()
    return pa.from_numpy_dtype(np.dtype(numpy_dtype))


def _pcd_header_fields(head: bytes) -> list[pa.Field] | None:
    """PCD field names and types from the ASCII header, or `None` if it is not all here.

    The types must match what `_parse_pcd` actually produces rather than what the header
    declares, and those differ: an `ascii` body goes through `np.loadtxt(dtype=float64)`,
    so every field comes back Float64 whatever `TYPE` says. A schema that reported the
    declared type would be a schema the reader then contradicts.
    """
    end = head.find(b"DATA")
    if end < 0:
        return None
    line_end = head.find(b"\n", end)
    if line_end < 0:
        return None
    fields: dict[str, str] = {}
    for line in head[:line_end].decode("ascii", "replace").splitlines():
        parts = line.split()
        if parts:
            fields[parts[0].upper()] = " ".join(parts[1:])
    if "FIELDS" not in fields or "DATA" not in fields:
        return None
    names = fields["FIELDS"].split()
    if fields["DATA"].strip().lower() == "ascii":
        return [pa.field(name, pa.float64()) for name in names]
    try:
        sizes = [int(s) for s in fields["SIZE"].split()]
        types = fields["TYPE"].split()
        return [
            pa.field(name, _arrow_of(_PCD_TYPES[(kind, size)]))
            for name, kind, size in zip(names, types, sizes, strict=True)
        ]
    except (KeyError, ValueError):
        # An unsupported layout: let the full parse raise the specific error rather than
        # guessing a schema for a file that will not read.
        return None


def _ply_header_fields(head: bytes) -> list[pa.Field] | None:
    """PLY vertex property names and types from the header, or `None` if it is truncated.

    As with PCD, an `ascii` body is read as Float64 throughout, so that is what the schema
    reports — the declared per-property types apply only to the binary encodings.
    """
    marker = b"end_header\n"
    end = head.find(marker)
    if end < 0 or not head.lstrip().startswith(b"ply"):
        return None
    fmt = "ascii"
    props: list[tuple[str, str]] = []
    in_vertex = False
    for line in head[:end].decode("ascii", "replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
        elif parts[0] == "property" and in_vertex:
            if parts[1] == "list" or parts[1] not in _PLY_TYPES:
                return None  # faces, or a type the parser will reject anyway
            props.append((_PLY_TYPES[parts[1]], parts[2]))
    if not props:
        return None
    if fmt == "ascii":
        return [pa.field(name, pa.float64()) for _, name in props]
    endian = "<" if "little" in fmt else ">"
    return [pa.field(name, _arrow_of(endian + t)) for t, name in props]
