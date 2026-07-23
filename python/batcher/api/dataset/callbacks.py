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


def _to_table(
    rows: list[dict[str, Any]],
    template: pa.RecordBatch,
    out_columns: tuple[str, ...] | None = None,
) -> pa.Table:
    """Build an output table from per-row dicts, preserving the *output* schema when a
    batch produces no rows.

    An empty result carries no rows to infer types from, so the schema must be
    synthesized. When the callback declared `output_columns` that differ from the input
    (it renames/adds/drops columns), falling back to the input schema loses those columns
    — an empty input batch (e.g. a filter that removed every row upstream) then makes a
    downstream reference to a callback-added column fail, while the same query on
    non-empty data succeeds. Emit the declared columns as 0-row null-typed arrays instead
    (a 0-row null column satisfies a downstream projection and unifies with a real-typed
    batch of the same stage); with no declared columns the input schema is the right
    pass-through fallback.
    """
    if rows:
        return pa.Table.from_pylist(rows)
    if out_columns is not None and list(out_columns) != template.schema.names:
        return pa.table({name: pa.array([], type=pa.null()) for name in out_columns})
    return pa.Table.from_batches([template.slice(0, 0)])


class _RowMap:
    """Apply a per-row ``fn(row_dict) -> row_dict`` over each batch's rows."""

    __slots__ = ("fn", "out_columns")

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        out_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        return _to_table([self.fn(row) for row in batch.to_pylist()], batch, self.out_columns)


class _RowFlatMap:
    """Apply a per-row ``fn(row_dict) -> iterable[row_dict]`` and flatten the rows."""

    __slots__ = ("fn", "out_columns")

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], Iterable[dict[str, Any]]],
        out_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        out: list[dict[str, Any]] = []
        for row in batch.to_pylist():
            out.extend(self.fn(row))
        return _to_table(out, batch, self.out_columns)


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
    ``concurrency``/…); apply the result to a dataset by calling it. Pass
    ``per_row=True`` to write a ``fn(row) -> row`` per-row callback instead of a
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

            >>> @bt.udf(concurrency=2)
            ... def double(batch):
            ...     return batch.set_column(0, "x", pc.multiply(batch.column("x"), 2))
            >>> double(bt.from_pydict({"x": [1, 2, 3]})).to_pydict()
            {'x': [2, 4, 6]}

    Args:
        fn: The function to wrap when used bare as ``@udf``; ``None`` when used with
            options as ``@udf(...)``, which returns a decorator.
        per_row: Treat `fn` as a per-row ``fn(row) -> row`` callback rather than a
            whole-batch function.
        **config: `map_batches` options forwarded to the transform (e.g.
            ``batch_format``, ``num_gpus``, ``concurrency``).

    Returns:
        The configured `Udf` when applied to a function, otherwise a decorator that
        produces one.
    """

    def wrap(f: Callable) -> Udf:
        return Udf(f, per_row=per_row, config=config)

    return wrap(fn) if fn is not None else wrap
