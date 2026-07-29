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

# In-flight per-row awaits within one batch for an async row `fn` with no explicit bound. An
# I/O-bound row callback (a per-row LLM / API / vector-DB call) wants many concurrent awaits; 32
# overlaps latency well without hammering a remote service by default.
_DEFAULT_ROW_CONCURRENCY = 32


def _gather_rows(fn: Callable, rows: list[dict[str, Any]], limit: int) -> list[Any]:
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


def _fn_label(fn: Callable) -> str:
    """A readable name for a row callback, for an error message."""
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)


def _check_row_result(value: Any, fn: Callable) -> None:
    """Reject a `map` callback result that is not one row dict, naming the callback.

    Anything else reached `Table.from_pylist` and came back as
    ``AttributeError: 'int' object has no attribute 'keys'`` — raised inside pyarrow, naming
    neither `ds.ml.map` nor the callback nor the shape that was wanted. Checked on the first
    row of each batch only: it is `O(1)` per batch, and a callback that changes its return
    shape partway through a batch is not the mistake this is for.
    """
    if isinstance(value, dict):
        return
    from batcher._internal.errors import PlanError

    raise PlanError(
        f"the ds.ml.map callback {_fn_label(fn)!r} returned {type(value).__name__}, but a "
        f"per-row callback must return one {{column: value}} dict per row. Return "
        f"`{{**row, 'new': ...}}` to add a column, or use `ds.ml.flat_map` if one row "
        f"produces several."
    )


def _check_flat_row_result(value: Any, fn: Callable) -> None:
    """Reject a `flat_map` callback result that is not an iterable of row dicts.

    Returning a single dict is the plausible mistake here, and it was the worst-behaved:
    iterating a dict yields its *keys*, so each row became a bare string and the failure
    surfaced as ``'str' object has no attribute 'keys'`` — pointing at the wrong thing
    entirely. A `None` return produced ``'NoneType' object is not iterable``, which at
    least names the shape but not the callback.
    """
    from batcher._internal.errors import PlanError

    if isinstance(value, dict):
        raise PlanError(
            f"the ds.ml.flat_map callback {_fn_label(fn)!r} returned a single dict. A "
            f"flat_map callback returns an *iterable* of row dicts — wrap it in a list "
            f"(`[{{...}}]`), or use `ds.ml.map` for one row in, one row out."
        )
    if value is None or isinstance(value, str | bytes):
        raise PlanError(
            f"the ds.ml.flat_map callback {_fn_label(fn)!r} returned "
            f"{type(value).__name__}, but it must return an iterable of {{column: value}} "
            f"row dicts (an empty list to drop the row)."
        )
    if isinstance(value, list | tuple) and value and not isinstance(value[0], dict):
        raise PlanError(
            f"the ds.ml.flat_map callback {_fn_label(fn)!r} returned a "
            f"{type(value).__name__} of {type(value[0]).__name__}, but each element must be "
            f"a {{column: value}} row dict."
        )


def _carry_identity(adapter: object, fn: Callable) -> None:
    """Give a row adapter the wrapped callback's module/qualname.

    `strategy._fn_probe_key` keys its measured per-row cost — and `strategy.error_budget`
    keys the ``max_errored_rows`` allowance — on ``fn.__module__`` + ``fn.__qualname__``. A
    `__slots__` adapter exposes neither, so the key came back `None` and the row path took
    the *uncached* branch: the per-row cost probe ran again on every query, and the
    cross-session warm start never applied, on the one path (row-at-a-time Python) where
    that measurement is worth the most. The budget then fell back to ``id(fn)``, which is
    reusable after a garbage collection.

    Carrying the callback's own identity fixes all three, and makes a profile name the stage
    after the function the user wrote rather than after this adapter.
    """
    for attr in ("__module__", "__qualname__", "__name__"):
        value = getattr(fn, attr, None)
        if value is not None:
            setattr(adapter, attr, value)


class _RowMap:
    """Apply a per-row ``fn(row_dict) -> row_dict`` over each batch's rows."""

    #: Marks this as the per-row adapter, so a profile can tell a `map` stage from a
    #: `map_batches` one. They are the same operator to the engine — `map` lowers to
    #: `map_batches` over a row loop — and without the mark the run cannot report the
    #: 10-100x row-at-a-time cost the field guides put at the top of their list.
    batcher_row_adapter = True

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        out_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns
        _carry_identity(self, fn)

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        rows = [self.fn(row) for row in batch.to_pylist()]
        if rows:
            _check_row_result(rows[0], self.fn)
        return _to_table(rows, batch, self.out_columns)


class _RowFlatMap:
    """Apply a per-row ``fn(row_dict) -> iterable[row_dict]`` and flatten the rows."""

    #: Marks this as the per-row adapter, so a profile can tell a `map` stage from a
    #: `map_batches` one. They are the same operator to the engine — `map` lowers to
    #: `map_batches` over a row loop — and without the mark the run cannot report the
    #: 10-100x row-at-a-time cost the field guides put at the top of their list.
    batcher_row_adapter = True

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], Iterable[dict[str, Any]]],
        out_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns
        _carry_identity(self, fn)

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        out: list[dict[str, Any]] = []
        for index, row in enumerate(batch.to_pylist()):
            produced = self.fn(row)
            if index == 0:
                _check_flat_row_result(produced, self.fn)
            out.extend(produced)
        return _to_table(out, batch, self.out_columns)


class _AsyncRowMap:
    """Apply an async ``fn(row_dict) -> row_dict`` over a batch's rows, gathered concurrently.

    The event loop runs inside `__call__` (a synchronous batch callable to the engine), so an
    async per-row callback rides the normal thread path while its per-row awaits overlap up to
    `limit` at a time — the per-row LLM/API-enrichment pattern.
    """

    #: Marks this as the per-row adapter, so a profile can tell a `map` stage from a
    #: `map_batches` one. They are the same operator to the engine — `map` lowers to
    #: `map_batches` over a row loop — and without the mark the run cannot report the
    #: 10-100x row-at-a-time cost the field guides put at the top of their list.
    batcher_row_adapter = True

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], Any],
        out_columns: tuple[str, ...] | None = None,
        limit: int = _DEFAULT_ROW_CONCURRENCY,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns
        self.limit = limit
        _carry_identity(self, fn)

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        rows = _gather_rows(self.fn, batch.to_pylist(), self.limit)
        if rows:
            _check_row_result(rows[0], self.fn)
        return _to_table(rows, batch, self.out_columns)


class _AsyncRowFlatMap:
    """Apply an async ``fn(row_dict) -> iterable[row_dict]`` per row and flatten the results."""

    #: Marks this as the per-row adapter, so a profile can tell a `map` stage from a
    #: `map_batches` one. They are the same operator to the engine — `map` lowers to
    #: `map_batches` over a row loop — and without the mark the run cannot report the
    #: 10-100x row-at-a-time cost the field guides put at the top of their list.
    batcher_row_adapter = True

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], Iterable[dict[str, Any]]],
        out_columns: tuple[str, ...] | None = None,
        limit: int = _DEFAULT_ROW_CONCURRENCY,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns
        self.limit = limit
        _carry_identity(self, fn)

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        per_row = _gather_rows(self.fn, batch.to_pylist(), self.limit)
        if per_row:
            _check_flat_row_result(per_row[0], self.fn)
        out: list[dict[str, Any]] = []
        for rows in per_row:
            out.extend(rows)
        return _to_table(out, batch, self.out_columns)


class _BoundBatchFn:
    """Forward fixed extra arguments to every ``fn(batch, *args, **kwargs)`` call.

    `functools.partial` covers the keyword half of this and is what `_bind_fn` used, but it
    binds positionals to the FRONT — ahead of the batch — so it cannot express `fn_args` at
    all. A module-level class can, and (unlike a closure) it pickles, so the process pool and
    a distributed actor still accept a `fn` carrying arguments.
    """

    __slots__ = ("args", "fn", "kwargs")

    def __init__(self, fn: Callable, args: tuple, kwargs: dict[str, Any]) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, batch: Any) -> Any:
        return self.fn(batch, *self.args, **self.kwargs)


class _AsyncBoundBatchFn(_BoundBatchFn):
    """`_BoundBatchFn` for an ``async def`` `fn`.

    The `__call__` must itself be a coroutine function, not a plain method returning a
    coroutine: `is_async_udf` reads `__call__` statically, so a synchronous wrapper would
    route an async `fn` onto the thread path, where its un-awaited coroutine is coerced as
    a result and the batch silently becomes garbage.
    """

    __slots__ = ()

    async def __call__(self, batch: Any) -> Any:
        return await self.fn(batch, *self.args, **self.kwargs)


class Udf:
    """A function bundled with its `map_batches` configuration (from `@udf`).

    Call it on a dataset to apply the transform: ``cleaned = my_udf(ds)``. The
    wrapped function follows the `map_batches` contract (batch in, batch out) unless
    ``per_row=True`` was set, in which case it is a per-row callback.

    Calling it on a *batch* instead runs the wrapped function directly, so a decorated
    `fn` still works everywhere a plain one does — passed to `map_batches` by hand,
    unit-tested on a `RecordBatch`, or composed inside another UDF. Without that, the
    decorator quietly made the function unusable except through itself.
    """

    def __init__(self, fn: Callable, *, per_row: bool, config: dict[str, Any]) -> None:
        self.fn = fn
        self.per_row = per_row
        self.config = config
        # Carry the wrapped function's identity: the profile names stages by
        # `fn.__qualname__`, and the strategy probe caches its measured per-row cost under
        # `module.qualname`. An undecorated `Udf` reported every stage as the same
        # `callbacks.Udf`, so a profile could not tell two models apart and the probe cache
        # collided across every `@udf` in the process.
        for attr in ("__name__", "__qualname__", "__module__", "__doc__"):
            value = getattr(fn, attr, None)
            if value is not None:
                setattr(self, attr, value)

    def __repr__(self) -> str:
        name = getattr(self.fn, "__qualname__", repr(self.fn))
        opts = ", ".join(f"{k}={v!r}" for k, v in sorted(self.config.items()))
        kind = "per_row" if self.per_row else "batch"
        return f"<udf {name} ({kind}){': ' + opts if opts else ''}>"

    def options(self, **config: Any) -> Udf:
        """Return a copy of this UDF with `config` merged over its options.

        Lets one decorated function be reused at several scales without redefining it:
        ``embed.options(num_gpus=1, concurrency=4)`` for the cluster run and the bare
        ``embed`` for a local smoke test.

        Args:
            **config: `map_batches` options to override (e.g. ``num_gpus``,
                ``concurrency``, ``batch_size``).

        Returns:
            A new `Udf` wrapping the same function with the merged configuration.
        """
        return Udf(self.fn, per_row=self.per_row, config={**self.config, **config})

    def __call__(self, target: Any) -> Any:
        if not hasattr(target, "ml"):  # a batch (or a row), not a Dataset — run the fn itself
            return self.fn(target)
        if self.per_row:
            return target.ml.map(self.fn, **self.config)
        return target.ml.map_batches(self.fn, **self.config)


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
