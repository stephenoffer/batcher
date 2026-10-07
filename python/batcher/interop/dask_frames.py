"""Arrow partitions → a lazy ``dask.dataframe.DataFrame``.

Dask builds a frame from a function and the inputs of its partitions
(``dd.from_map``): nothing runs until a partition is computed, and each partition is
produced by its own task. This module is that mapping for Arrow, and it knows nothing about
where the Arrow comes from: a caller passes the partition inputs and a loader that turns
one input into a `pyarrow.Table`, and each partition's pandas frame is built only when
Dask computes it.

The empty-schema ``meta`` Dask needs up front is the schema's zero-row table converted the
same way the partitions are, so the declared and the computed dtypes cannot disagree.

Not yet verified against a live Dask install; see tests/PENDING_VERIFICATION.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pyarrow as pa

from batcher._internal.optional import require

__all__ = ["ArrowPartitionLoader", "dask_from_arrow_partitions", "import_dask_dataframe"]


def import_dask_dataframe() -> Any:
    """Import ``dask.dataframe`` or raise a guiding ``BackendError``."""
    return require(
        "dask.dataframe", feature="Dataset.to_dask()", provides="dask[dataframe]", extra="dask"
    )


class ArrowPartitionLoader:
    """The function Dask maps over partition inputs: load Arrow, convert to pandas.

    A class rather than a closure so a distributed Dask scheduler can pickle it by
    reference to this module.

    Args:
        load: Turns one partition input into a `pyarrow.Table`.
        schema: The schema every partition is cast to, so partitions agree with ``meta``.
    """

    def __init__(self, load: Callable[[Any], pa.Table], schema: pa.Schema) -> None:
        self._load = load
        self._schema = schema

    def __call__(self, part: Any) -> Any:
        table = self._load(part)
        if not table.schema.equals(self._schema):
            table = table.select(self._schema.names).cast(self._schema)
        return table.to_pandas()


def dask_from_arrow_partitions(
    parts: Sequence[Any], load: Callable[[Any], pa.Table], schema: pa.Schema
) -> Any:
    """A lazy Dask frame with one partition per element of `parts`.

    Args:
        parts: The partition inputs; one Dask partition each, in order.
        load: Turns one partition input into a `pyarrow.Table`; runs inside the task.
        schema: The partitions' Arrow schema, which also gives the frame's ``meta``.

    Returns:
        A ``dask.dataframe.DataFrame``. No partition has been computed.
    """
    dd = import_dask_dataframe()
    loader = ArrowPartitionLoader(load, schema)
    meta = schema.empty_table().to_pandas()
    return dd.from_map(loader, list(parts), meta=meta, label="batcher-to-dask")
