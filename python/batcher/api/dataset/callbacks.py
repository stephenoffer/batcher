"""Row-callback adapters and the ``@udf`` decorator for the callback transforms.

`map`/`flat_map` let a user write a per-row Python function; these adapters run that
function **inside the worker** over each Arrow batch's rows (the data plane), so the
control-plane driver still only ever ships whole batches — the hot-path invariant
holds. They are module-level classes (not closures) so Ray can pickle them across
the cluster. `udf` bundles a function with its `map_batches` config so it reads as a
reusable, configured transform (Ray Data / Daft ``@udf``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pyarrow as pa


def _to_table(rows: list[dict[str, Any]], template: pa.RecordBatch) -> pa.Table:
    """Build an output table from per-row dicts, falling back to an empty slice of
    the input schema when a batch produces no rows (so the schema is preserved)."""
    if rows:
        return pa.Table.from_pylist(rows)
    return pa.Table.from_batches([template.slice(0, 0)])


class _RowMap:
    """Apply a per-row ``fn(row_dict) -> row_dict`` over each batch's rows."""

    __slots__ = ("fn",)

    def __init__(self, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.fn = fn

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        return _to_table([self.fn(row) for row in batch.to_pylist()], batch)


class _RowFlatMap:
    """Apply a per-row ``fn(row_dict) -> iterable[row_dict]`` and flatten the rows."""

    __slots__ = ("fn",)

    def __init__(self, fn: Callable[[dict[str, Any]], Iterable[dict[str, Any]]]) -> None:
        self.fn = fn

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        out: list[dict[str, Any]] = []
        for row in batch.to_pylist():
            out.extend(self.fn(row))
        return _to_table(out, batch)


class Udf:
    """A function bundled with its `map_batches` configuration (from `@udf`).

    Call it on a dataset to apply the transform: ``cleaned = my_udf(ds)``. The
    wrapped function follows the `map_batches` contract (batch in, batch out) unless
    ``per_row=True`` was set, in which case it is a per-row callback.
    """

    __slots__ = ("config", "fn", "per_row")

    def __init__(self, fn: Callable, *, per_row: bool, config: dict[str, Any]) -> None:
        self.fn = fn
        self.per_row = per_row
        self.config = config

    def __call__(self, ds: Any) -> Any:
        if self.per_row:
            return ds.ml.map(self.fn, **self.config)
        return ds.ml.map_batches(self.fn, **self.config)


def udf(fn: Callable | None = None, *, per_row: bool = False, **config: Any) -> Any:
    """Decorate a function as a reusable, configured column transform (``@udf``).

    Bundles a function with its `map_batches` options (``batch_format``/``num_gpus``/
    ``concurrency``/…); apply the result to a dataset by calling it::

        @udf(batch_format="numpy", num_gpus=1.0)
        def classify(batch):
            ...

        scored = classify(ds)

    Pass ``per_row=True`` to write a ``fn(row) -> row`` per-row callback instead of a
    batch function. Usable bare (``@udf``) or with options (``@udf(...)``).

    Examples:
        .. doctest::

            >>> import pyarrow.compute as pc
            >>> import batcher as bt
            >>> @bt.udf
            ... def add_one(batch):
            ...     return batch.set_column(0, "x", pc.add(batch.column("x"), 1))
            >>> add_one(bt.from_pydict({"x": [1, 2, 3]})).to_pydict()
            {'x': [2, 3, 4]}
    """

    def wrap(f: Callable) -> Udf:
        return Udf(f, per_row=per_row, config=config)

    return wrap(fn) if fn is not None else wrap
