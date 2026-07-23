"""Row-oriented terminal consumers: `iter_rows`, `iter_slices`, `first`/`last`, `item`.

These are the "I have finished computing, now give me Python values" end of a
pipeline, and they are the *only* sanctioned place row-shaped Python appears. The
control-plane rule they respect is that no per-row work happens *inside* a query:
every function here consumes already-computed Arrow batches at the boundary, after
the engine has done the work.

`iter_rows` and `iter_slices` stream batch by batch rather than collecting, so
walking a result larger than memory stays bounded — the reason they exist instead
of telling users to call `to_pylist()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa

    from batcher.api.dataset.frame import Dataset

__all__ = [
    "build_first",
    "build_item",
    "build_iter_rows",
    "build_iter_slices",
    "build_last",
]


def build_iter_rows(ds: Dataset, named: bool) -> Iterator[tuple[Any, ...] | dict[str, Any]]:
    """Stream the result one row at a time, as tuples or `{column: value}` dicts."""
    for batch in ds.iter_batches():
        rows = batch.to_pylist()
        if named:
            yield from rows
        else:
            names = batch.schema.names
            for row in rows:
                yield tuple(row[n] for n in names)


def build_iter_slices(ds: Dataset, n_rows: int | None) -> Iterator[pa.RecordBatch]:
    """Stream the result as `RecordBatch` slices of at most `n_rows` rows each."""
    if n_rows is not None and n_rows < 1:
        raise PlanError(f"iter_slices(): n_rows must be >= 1, got {n_rows}")
    return ds.iter_batches(n_rows)


def build_first(ds: Dataset, named: bool) -> tuple[Any, ...] | dict[str, Any] | None:
    """The first result row, or `None` when the result is empty."""
    for row in build_iter_rows(ds.limit(1), named):
        return row
    return None


def build_last(ds: Dataset, named: bool) -> tuple[Any, ...] | dict[str, Any] | None:
    """The last result row, or `None` when the result is empty.

    A relation has no inherent row order, so "last" means the last row the plan
    emits; sort first when the answer must be deterministic.
    """
    out: tuple[Any, ...] | dict[str, Any] | None = None
    for row in build_iter_rows(ds, named):
        out = row
    return out


def build_item(ds: Dataset, column: str | None) -> Any:
    """The single scalar value of a one-row (and, without `column`, one-column) result."""
    columns = ds.columns
    if column is not None:
        if column not in columns:
            raise PlanError(f"item(): unknown column {column!r}; available: {sorted(columns)}")
        target = column
    elif len(columns) != 1:
        raise PlanError(
            f"item() needs a single-column result, but this one has {len(columns)} "
            f"columns ({sorted(columns)}); pass item(column='name') to pick one."
        )
    else:
        target = columns[0]

    # Take two rows so a multi-row result is reported as such rather than silently
    # returning the first value — the classic "it worked in the demo" scalar bug.
    rows = ds.select(target).limit(2).to_pydict()[target]
    if not rows:
        raise PlanError("item() needs exactly one row, but the result is empty")
    if len(rows) > 1:
        raise PlanError(
            "item() needs exactly one row, but the result has more than one; "
            "narrow it with .filter(...) or .limit(1) first"
        )
    return rows[0]
