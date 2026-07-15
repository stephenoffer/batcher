"""The `ds.meta` accessor tree (façade) — answer from metadata, execute only when you must.

`DatasetMeta` is the entry point (`Dataset.meta`); the rest are the sub-accessors it hands
out. The implementations live in the sibling modules — this file only re-exports them.
"""

from __future__ import annotations

from batcher.api.dataset.meta.approx import ApproxMeta
from batcher.api.dataset.meta.checks import ColumnChecks
from batcher.api.dataset.meta.column import ColumnMeta
from batcher.api.dataset.meta.frame import DatasetMeta
from batcher.api.dataset.meta.nulls import NullsMeta
from batcher.api.dataset.meta.pair import PairMeta
from batcher.api.dataset.meta.schema import SchemaMeta
from batcher.api.dataset.meta.storage import StorageMeta

__all__ = [
    "ApproxMeta",
    "ColumnChecks",
    "ColumnMeta",
    "DatasetMeta",
    "NullsMeta",
    "PairMeta",
    "SchemaMeta",
    "StorageMeta",
]
