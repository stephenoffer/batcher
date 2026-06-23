"""Terminal/materialization operations for `Dataset` — package façade.

`Dataset`'s terminal methods forward their state here. Split by memory model:
`core` owns the materializing terminals (`collect` / `count` / `to_*` / `write`,
which orchestrate Kyber → Carbonite → Core), and `stream` owns the bounded-memory
`iter_batches` path. Both are re-exported so `from batcher.api.terminal import X`
keeps working across the split.
"""

from __future__ import annotations

from batcher.api.terminal.core import (
    _collect,
    _count,
    _explain,
    _is_empty,
    _resolve_distributed,
    _schema,
    _show,
    _stats,
    _to_pandas,
    _to_polars,
    _to_pydict,
    _to_pylist,
    _write,
)
from batcher.api.terminal.stream import _iter_batches

__all__ = [
    "_collect",
    "_count",
    "_explain",
    "_is_empty",
    "_iter_batches",
    "_resolve_distributed",
    "_schema",
    "_show",
    "_stats",
    "_to_pandas",
    "_to_polars",
    "_to_pydict",
    "_to_pylist",
    "_write",
]
