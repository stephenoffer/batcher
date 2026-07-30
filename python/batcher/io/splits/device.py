"""Which splits a GPU can read for itself, and the locators it needs to do it.

A GPU worker reading Parquet through the host path decodes on the CPU and then copies the
result across PCIe: the device sits idle for the decode and pays for the transfer afterwards.
Its own Parquet reader does both at once, so the file goes to the device and is decoded there.
On a scan-heavy query that decode is most of the query, which makes this the difference
between a GPU that is computing and one that is waiting for a CPU to hand it something.

What that needs is a *locator* — the file and, where the split is finer than a file, its row
groups. This module is where a split says whether it has one. It lives in `io` because
recognizing `io`'s own split types is `io`'s business, and because the answer is useful to
anything that can read a file directly, not only to the GPU backend that wants it first.

Two rules keep it from becoming a source of wrong answers rather than a source of speed:

* **Only types both readers agree on.** The device reader is a second implementation of
  Parquet, and a second implementation is a second set of edge cases. A split whose schema
  carries anything outside the numeric/boolean/string/temporal core reports no locator, so it
  reads through the host path that the engine's own tests cover.
* **Only when nothing was pushed into the read.** A predicate the host reader would have used
  to skip row groups is not expressible here, and a device read that ignores it would move far
  more bytes than the path it replaced. Declining is the honest answer: a selective query keeps
  the reader that can be selective.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

__all__ = ["DeviceReadSpec", "device_read_specs"]


@dataclass(frozen=True, slots=True)
class DeviceReadSpec:
    """One file a device can read for itself, and the row groups of it that are wanted.

    Attributes:
        path: The file's path or URI, as the split holds it.
        row_groups: The row-group indices to read, or `None` for the whole file.
    """

    path: str
    row_groups: tuple[int, ...] | None = None


def device_read_specs(splits: list, projection: list[str] | None) -> list[DeviceReadSpec] | None:
    """Locators for reading `splits` directly on a device, or `None` when one cannot.

    All-or-nothing by design. A descriptor read half on the device and half on the host would
    concatenate two readers' output, and the schemas those two produce are exactly what this
    module cannot promise to be identical.

    Args:
        splits: The splits of one partition descriptor.
        projection: The columns to be read, or `None` for all of them. Used to narrow the
            type check to the columns that will actually cross, so a table with one
            unsupported column can still be read on the device when nobody selected it.

    Returns:
        One spec per split, or `None` when any split is not a plain Parquet locator or reads a
        type the two readers may not agree on.

    Examples:
        .. doctest::

            >>> from batcher.io.splits.device import device_read_specs
            >>> device_read_specs([], None) is None
            True
    """
    from batcher.io.splits.file import FileSplit
    from batcher.io.splits.parquet import RowGroupSplit

    if not splits:
        return None
    specs: list[DeviceReadSpec] = []
    for split in splits:
        if isinstance(split, RowGroupSplit):
            specs.append(DeviceReadSpec(split.path, tuple(split.row_groups)))
        elif isinstance(split, FileSplit) and split.format_name == "parquet" and not split.kwargs:
            specs.append(DeviceReadSpec(split.path))
        else:
            return None
    if not _device_readable_schema(splits[0].schema(), projection):
        return None
    return specs


def _device_readable_schema(schema: pa.Schema, projection: list[str] | None) -> bool:
    """Whether every column that will be read is a type both Parquet readers agree on."""
    fields = schema if projection is None else [schema.field(c) for c in projection]
    return all(_device_readable_type(field.type) for field in fields)


def _device_readable_type(dtype: pa.DataType) -> bool:
    """Whether one Arrow type round-trips through a device Parquet read unchanged.

    Deliberately a small list rather than an exclusion list. Nested types, dictionaries,
    decimals, and the extension types each have at least one representation the two readers
    resolve differently — a dictionary that comes back dense, a decimal that comes back at a
    different scale — and every one of those changes the schema a shard contributes to a
    concatenation. An unlisted type is not a type this cannot eventually read; it is one
    nobody has yet checked, and the two answers should not be confused.
    """
    return (
        pa.types.is_boolean(dtype)
        or pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_string(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_date(dtype)
        or pa.types.is_timestamp(dtype)
    )
