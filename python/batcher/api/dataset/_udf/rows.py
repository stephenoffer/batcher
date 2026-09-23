"""The output tables of the per-row callbacks behind `ds.map` and `ds.flat_map`.

A row adapter (`callbacks._RowMap` and friends) calls the user's function once per row
inside the worker; this module turns what those calls returned back into one Arrow table,
and runs an ``async def`` row callback's awaits concurrently. Split from `callbacks`, which
holds the adapters and the ``@udf`` decorator, so each stays within the size contract.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pyarrow as pa

__all__ = ["gather_rows", "rows_to_table"]


def gather_rows(fn: Callable, rows: list[dict[str, Any]], limit: int) -> list[Any]:
    """Await `fn(row)` over every row concurrently on one event loop, bounded and in order.

    The per-row analog of the async `map_batches` runner: a per-row `async def` callback (a
    per-row LLM/API call) issues up to `limit` concurrent requests within a batch, instead of
    awaiting them one at a time. Runs a fresh loop per batch (the sync batch path holds no
    running loop), so the whole adapter stays a plain synchronous batch callable to the engine.
    """
    # `asyncio` is imported here rather than at module scope: this module is reached from
    # `batcher/__init__` (it defines the public `udf` decorator), so a module-level import
    # pulled the whole `asyncio` package — ~10 ms and two dozen modules — into every
    # `import batcher`, for a path only an `async def` row callback ever takes.
    import asyncio

    from batcher.core.udf.async_udf import run_coroutine_blocking

    sem = asyncio.Semaphore(max(1, limit))

    async def _one(row: dict[str, Any]) -> Any:
        async with sem:
            return await fn(row)

    async def _run() -> list[Any]:
        return await asyncio.gather(*(_one(r) for r in rows))

    # Safe inside an already-running loop (Jupyter / async app), where `asyncio.run` would raise.
    return run_coroutine_blocking(_run)


def rows_to_table(
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
        return _rows_table(rows)
    if out_columns is not None and list(out_columns) != template.schema.names:
        return pa.table({name: pa.array([], type=pa.null()) for name in out_columns})
    return pa.Table.from_batches([template.slice(0, 0)])


def _rows_table(rows: list[dict[str, Any]]) -> pa.Table:
    """The table of per-row dicts, with every key any row produced as a column.

    `Table.from_pylist` takes its columns from the *first* row, so a callback whose rows
    carry different keys (``{"a": 1}`` for even rows, ``{"b": 2}`` for odd ones) silently
    lost every key the first row lacked. The rule here is the one `map_batches` applies
    across batches (`io.schema.evolution`): the columns are the union of the keys in the
    order they first appear, a row missing a key is null there, and a key that an
    earlier row had and a later one dropped is warned about, since that is usually a
    rename or a bug rather than an optional field.
    """
    first = rows[0].keys()
    names = list(first)
    seen = set(names)
    dropped: set[str] = set()
    for row in rows[1:]:
        keys = row.keys()
        if keys == first:
            continue
        dropped |= seen - keys
        for key in keys:
            if key not in seen:
                seen.add(key)
                names.append(key)
    if len(names) == len(first) and not dropped:
        return pa.Table.from_pylist(rows)
    if dropped:
        from batcher._internal.logging import get_logger, log_kv

        log_kv(
            get_logger("api"),
            logging.WARNING,
            "output schema dropped a column between rows; the missing rows are "
            "null-filled, not removed",
            context="map",
            dropped=sorted(dropped),
        )
    return pa.Table.from_pydict({name: [row.get(name) for row in rows] for name in names})
