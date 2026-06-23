"""ML / array formats (NumPy, TFRecord, WebDataset, HDF5, Zarr) + training shards
+ the fixed-shape-tensor column type shared across the ML data path."""

from __future__ import annotations

from batcher.io.formats.ml.hdf5 import HDF5Source
from batcher.io.formats.ml.numpy import NumpySource
from batcher.io.formats.ml.shards import ShardIndex, ShardReader, read_shard_index, write_shards
from batcher.io.formats.ml.tensor import is_tensor_column, tensor_type, to_tensor_column
from batcher.io.formats.ml.tfrecord import TFRecordSource
from batcher.io.formats.ml.webdataset import WebDatasetSource
from batcher.io.formats.ml.zarr import ZarrSource

__all__ = [
    "HDF5Source",
    "NumpySource",
    "ShardIndex",
    "ShardReader",
    "TFRecordSource",
    "WebDatasetSource",
    "ZarrSource",
    "is_tensor_column",
    "read_shard_index",
    "tensor_type",
    "to_tensor_column",
    "write_shards",
]
