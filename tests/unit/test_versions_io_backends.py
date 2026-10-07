"""`bt.versions()` reports the I/O backends a remote path needs (AP-436)."""

from __future__ import annotations

import importlib.util

import batcher as bt

_DRIVERS = ("fsspec", "s3fs", "gcsfs", "adlfs", "adbc_driver_manager", "connectorx")


def test_every_io_driver_has_a_row():
    info = bt.versions()
    for name in _DRIVERS:
        installed = importlib.util.find_spec(name) is not None
        assert (info[name] != "not installed") == installed, name


def test_pyarrow_object_store_filesystems_are_reported():
    import pyarrow.fs as pafs

    info = bt.versions()
    for key, cls in (
        ("pyarrow_s3", "S3FileSystem"),
        ("pyarrow_gcs", "GcsFileSystem"),
        ("pyarrow_azure", "AzureFileSystem"),
    ):
        assert info[key] == ("available" if hasattr(pafs, cls) else "not built")
