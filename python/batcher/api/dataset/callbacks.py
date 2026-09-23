"""Callback adapters and the ``@udf`` decorator for the callback transforms.

`map`/`flat_map` let a user write a per-row Python function; these adapters run that
function **inside the worker** over each Arrow batch's rows (the data plane), so the
control-plane driver still only ever ships whole batches — the hot-path invariant
holds. A callable `filter` is batch-level: its adapter hands the predicate a whole batch
and applies the boolean mask it returns as Arrow. The per-row adapters are module-level
classes (not closures) so Ray can pickle them across the cluster. `udf` bundles a function
with its `map_batches` config so it reads as a reusable, configured transform (Ray Data /
Daft ``@udf``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pyarrow as pa

from batcher.api.dataset._udf.rows import gather_rows, rows_to_table

# In-flight per-row awaits within one batch for an async row `fn` with no explicit bound. An
# I/O-bound row callback (a per-row LLM / API / vector-DB call) wants many concurrent awaits; 32
# overlaps latency well without hammering a remote service by default.
_DEFAULT_ROW_CONCURRENCY = 32


def _row_dicts(batch: pa.RecordBatch, fmt: str, writable: bool) -> list[dict[str, Any]]:
    """A batch's rows as ``{column: value}`` dicts, in the row format a callback asked for.

    ``"pyarrow"`` rows hold Python values (``to_pylist``). ``"numpy"`` rows hold NumPy values
    the way Ray Data's rows do, so a tensor column arrives as an ``ndarray`` per row rather than
    a nested list; `writable` copies a read-only column first, so a callback may mutate what it
    is handed.
    """
    if fmt == "pyarrow":
        return batch.to_pylist()
    from batcher.interop.formats import to_format

    arrays = to_format(batch, "numpy")
    if writable:
        arrays = writable_batch(arrays, "numpy")
    return [{name: col[i] for name, col in arrays.items()} for i in range(batch.num_rows)]


#: The batch formats a callback can mutate in place once copied. Arrow, Polars and JAX data is
#: immutable in every engine, so a writable request has nothing to copy there.
WRITABLE_FORMATS = ("numpy", "pandas", "torch")


def _writable_array(arr: Any) -> Any:
    """`arr`, or a copy of it when NumPy marks it read-only."""
    flags = getattr(arr, "flags", None)
    return arr if flags is None or flags.writeable else arr.copy()


def writable_batch(batch: Any, fmt: str) -> Any:
    """`batch` with every read-only buffer copied, for ``zero_copy_batch=False``.

    The zero-copy conversions hand a callback views over Arrow memory, which NumPy marks
    read-only; a callback that writes into one raises. Ray Data copies by default so the write
    succeeds, and this is that copy, paid only when it was asked for.
    """
    if fmt == "numpy":
        return {name: _writable_array(arr) for name, arr in batch.items()}
    if fmt == "pandas":
        return batch.copy(deep=True)
    if fmt == "torch":
        return {name: tensor.clone() for name, tensor in batch.items()}
    return batch


def _fn_label(fn: Callable) -> str:
    """A readable name for a row callback, for an error message."""
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)


def _check_row_result(value: Any, fn: Callable) -> None:
    """Reject a `map` callback result that is not one row dict, naming the callback.

    Anything else reached `Table.from_pylist` and came back as
    ``AttributeError: 'int' object has no attribute 'keys'`` — raised inside pyarrow, naming
    neither `ds.map` nor the callback nor the shape that was wanted. Checked on the first
    row of each batch only: it is `O(1)` per batch, and a callback that changes its return
    shape partway through a batch is not the mistake this is for.
    """
    if isinstance(value, dict):
        return
    from batcher._internal.errors import PlanError

    raise PlanError(
        f"the ds.map callback {_fn_label(fn)!r} returned {type(value).__name__}, but a "
        f"per-row callback must return one {{column: value}} dict per row. Return "
        f"`{{**row, 'new': ...}}` to add a column, or use `ds.flat_map` if one row "
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
            f"the ds.flat_map callback {_fn_label(fn)!r} returned a single dict. A "
            f"flat_map callback returns an *iterable* of row dicts — wrap it in a list "
            f"(`[{{...}}]`), or use `ds.map` for one row in, one row out."
        )
    if value is None or isinstance(value, str | bytes):
        raise PlanError(
            f"the ds.flat_map callback {_fn_label(fn)!r} returned "
            f"{type(value).__name__}, but it must return an iterable of {{column: value}} "
            f"row dicts (an empty list to drop the row)."
        )
    if isinstance(value, list | tuple) and value and not isinstance(value[0], dict):
        raise PlanError(
            f"the ds.flat_map callback {_fn_label(fn)!r} returned a "
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

    The defining line rides along under a private attribute because a *locally* defined
    callback has no unique qualname: every lambda in one enclosing function is
    ``mod.outer.<locals>.<lambda>``, so ``ds.map(lambda r: ...)`` twice in one function
    would share one cost measurement and one error allowance. `strategy._fn_probe_key` reads
    the line off `__code__` when it can; an adapter has no code object, so it is handed one.
    """
    for attr in ("__module__", "__qualname__", "__name__"):
        value = getattr(fn, attr, None)
        if value is not None:
            setattr(adapter, attr, value)
    code = getattr(fn, "__code__", None)
    if code is not None:
        adapter._batcher_def_line = code.co_firstlineno  # type: ignore[attr-defined]


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
        fmt: str = "pyarrow",
        writable: bool = False,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns
        self.fmt = fmt
        self.writable = writable
        _carry_identity(self, fn)

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        rows = [self.fn(row) for row in _row_dicts(batch, self.fmt, self.writable)]
        if rows:
            _check_row_result(rows[0], self.fn)
        return rows_to_table(rows, batch, self.out_columns)


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
        fmt: str = "pyarrow",
        writable: bool = False,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns
        self.fmt = fmt
        self.writable = writable
        _carry_identity(self, fn)

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        out: list[dict[str, Any]] = []
        for index, row in enumerate(_row_dicts(batch, self.fmt, self.writable)):
            produced = self.fn(row)
            if index == 0:
                _check_flat_row_result(produced, self.fn)
            out.extend(produced)
        return rows_to_table(out, batch, self.out_columns)


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
        fmt: str = "pyarrow",
        writable: bool = False,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns
        self.limit = limit
        self.fmt = fmt
        self.writable = writable
        _carry_identity(self, fn)

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        rows = gather_rows(self.fn, _row_dicts(batch, self.fmt, self.writable), self.limit)
        if rows:
            _check_row_result(rows[0], self.fn)
        return rows_to_table(rows, batch, self.out_columns)


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
        fmt: str = "pyarrow",
        writable: bool = False,
    ) -> None:
        self.fn = fn
        self.out_columns = out_columns
        self.limit = limit
        self.fmt = fmt
        self.writable = writable
        _carry_identity(self, fn)

    def __call__(self, batch: pa.RecordBatch) -> pa.Table:
        per_row = gather_rows(self.fn, _row_dicts(batch, self.fmt, self.writable), self.limit)
        if per_row:
            _check_flat_row_result(per_row[0], self.fn)
        out: list[dict[str, Any]] = []
        for rows in per_row:
            out.extend(rows)
        return rows_to_table(out, batch, self.out_columns)


def _as_mask(result: Any, num_rows: int, fn: Callable) -> pa.Array:
    """A filter callback's answer as one Arrow boolean array, or a `PlanError` naming `fn`.

    The answer may be anything that converts to a boolean column: an Arrow array, a NumPy or
    torch boolean array, a pandas or Polars Series, or a list. A null keeps nothing, as a null
    predicate does in SQL. A non-boolean answer is refused rather than coerced, because a
    truthy integer or string column would keep rows for a reason nobody wrote.
    """
    from batcher._internal.errors import PlanError

    if hasattr(result, "detach"):  # a torch tensor, possibly on a device
        result = result.detach().cpu().numpy()
    if not isinstance(result, pa.Array | pa.ChunkedArray):
        try:
            result = pa.array(result, from_pandas=True)
        except (TypeError, ValueError, pa.ArrowException) as exc:
            raise PlanError(
                f"the ds.filter callback {_fn_label(fn)!r} returned {type(result).__name__}, "
                "which is not a boolean column. Return one True/False per row of the batch."
            ) from exc
    if isinstance(result, pa.ChunkedArray):
        result = result.combine_chunks()
    if pa.types.is_null(result.type):  # an empty list, or all None: no value to keep a row
        result = result.cast(pa.bool_())
    if not pa.types.is_boolean(result.type):
        raise PlanError(
            f"the ds.filter callback {_fn_label(fn)!r} returned a {result.type} column, but it "
            "must return a boolean mask with one value per row. Return a comparison such as "
            "`pc.greater(batch['x'], 0)` or `batch['x'] > 0`."
        )
    if len(result) != num_rows:
        raise PlanError(
            f"the ds.filter callback {_fn_label(fn)!r} returned {len(result)} values for a "
            f"batch of {num_rows} rows; the mask needs exactly one value per row."
        )
    return result


class _BatchFilter:
    """Keep the rows of each batch for which ``fn(batch)`` returns True.

    `fn` sees the whole batch in `fmt` and returns one boolean per row, so no Python runs per
    row. The batch is then filtered with that mask as Arrow, so **every column keeps its exact
    type** and no value round-trips through the callback's format: a mask cannot widen a
    `float32` or re-infer a column from its values.

    `read` narrows what `fn` is handed to the columns it declared. That is safe here in a way
    it would not be for a transform, because the output is the original batch masked, so no
    column can go missing from the result by narrowing what the predicate saw.
    """

    def __init__(
        self, fn: Callable, fmt: str = "pyarrow", read: tuple[str, ...] | None = None
    ) -> None:
        self.fn = fn
        self.fmt = fmt
        self.read = read
        _carry_identity(self, fn)

    def _view(self, batch: pa.RecordBatch) -> Any:
        """What `fn` is handed: the declared columns, in the requested format."""
        if self.read:
            batch = batch.select([name for name in self.read if name in batch.schema.names])
        if self.fmt == "pyarrow":
            return batch
        from batcher.interop.formats import to_format

        return to_format(batch, self.fmt)

    def _undeclared(self, exc: KeyError) -> Exception:
        """Explain a `KeyError` from a predicate that read a column it did not declare.

        Without this the failure is a bare ``KeyError: 'y'`` from inside the user's own
        lambda, which points at the lambda rather than at the declaration that removed the
        column, several lines away and easily mistaken for a performance hint.
        """
        from batcher._internal.errors import PlanError

        if not self.read:
            return exc
        return PlanError(
            f"the ds.filter callback {_fn_label(self.fn)!r} read column {exc.args[0]!r}, which "
            f"is not in its declared input_columns={list(self.read)}. A declared predicate is "
            "handed only the columns it declared, so add the column to input_columns."
        )

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        try:
            answer = self.fn(self._view(batch))
        except KeyError as exc:
            raise self._undeclared(exc) from exc
        return batch.filter(_as_mask(answer, batch.num_rows, self.fn))


class _AsyncBatchFilter(_BatchFilter):
    """`_BatchFilter` for an ``async def`` predicate, awaited on the async batch path."""

    async def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:  # type: ignore[override]
        try:
            answer = await self.fn(self._view(batch))
        except KeyError as exc:
            raise self._undeclared(exc) from exc
        return batch.filter(_as_mask(answer, batch.num_rows, self.fn))


def filter_adapter(bound: Callable | type, fmt: str, read: tuple[str, ...] | None) -> Any:
    """The `MapBatches` callable for a batch predicate: an instance, or a class for a class `fn`.

    A class `fn` stays a class, so the engine still builds it once per worker; its instance is
    wrapped in the mask adapter at construction.
    """
    from batcher.core.udf.async_udf import is_async_udf

    adapter = _AsyncBatchFilter if is_async_udf(bound) else _BatchFilter
    if not isinstance(bound, type):
        return adapter(bound, fmt, read)

    class _FilterModel:
        def __init__(self) -> None:
            self._filter = adapter(bound(), fmt, read)

        def close(self) -> None:
            close = getattr(self._filter.fn, "close", None)
            if callable(close):
                close()

    if adapter is _AsyncBatchFilter:

        class _Model(_FilterModel):
            async def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
                return await self._filter(batch)

    else:

        class _Model(_FilterModel):  # type: ignore[no-redef]
            def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
                return self._filter(batch)

    _Model.__name__ = _Model.__qualname__ = f"Filter{bound.__name__}"
    return _Model


def row_adapter(
    bound: Callable | type,
    cols: tuple[str, ...] | None,
    limit: int,
    *,
    flat: bool,
    fmt: str,
    writable: bool,
) -> Any:
    """The `MapBatches` callable for a per-row `map`/`flat_map`, async-aware and class-aware.

    An ``async def`` row `fn` gets the adapter that awaits a batch's rows concurrently, up to
    `limit`. A class `fn` stays a class, built once per worker with its instance wrapped.
    """
    from batcher.core.udf.async_udf import is_async_udf

    if is_async_udf(bound):
        adapter: Any = _AsyncRowFlatMap if flat else _AsyncRowMap

        def build(fn: Callable) -> Any:
            return adapter(fn, cols, limit, fmt, writable)

    else:
        adapter = _RowFlatMap if flat else _RowMap

        def build(fn: Callable) -> Any:
            return adapter(fn, cols, fmt, writable)

    if not isinstance(bound, type):
        return build(bound)

    class _RowModel:
        batcher_row_adapter = True

        def __init__(self) -> None:
            self._rows = build(bound())

        def __call__(self, batch: pa.RecordBatch) -> pa.Table:
            return self._rows(batch)

        def close(self) -> None:
            close = getattr(self._rows.fn, "close", None)
            if callable(close):
                close()

    _RowModel.__name__ = _RowModel.__qualname__ = f"Rows{bound.__name__}"
    return _RowModel


class _BoundBatchFn:
    """Forward fixed extra arguments to every ``fn(batch, *args, **kwargs)`` call.

    `functools.partial` covers the keyword half of this and is what `_bind_fn` used, but it
    binds positionals to the FRONT — ahead of the batch — so it cannot express `fn_args` at
    all. A module-level class can, and (unlike a closure) it pickles, so the process pool and
    a distributed actor still accept a `fn` carrying arguments.

    `writable` names the batch format to copy read-only buffers out of before the call
    (``zero_copy_batch=False``), or is `None` to hand the batch over as converted.
    """

    __slots__ = ("args", "fn", "kwargs", "writable")

    def __init__(
        self, fn: Callable, args: tuple, kwargs: dict[str, Any], writable: str | None = None
    ) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.writable = writable

    def _prepare(self, batch: Any) -> Any:
        return batch if self.writable is None else writable_batch(batch, self.writable)

    def __call__(self, batch: Any) -> Any:
        return self.fn(self._prepare(batch), *self.args, **self.kwargs)


class _AsyncBoundBatchFn(_BoundBatchFn):
    """`_BoundBatchFn` for an ``async def`` `fn`.

    The `__call__` must itself be a coroutine function, not a plain method returning a
    coroutine: `is_async_udf` reads `__call__` statically, so a synchronous wrapper would
    route an async `fn` onto the thread path, where its un-awaited coroutine is coerced as
    a result and the batch silently becomes garbage.
    """

    __slots__ = ()

    async def __call__(self, batch: Any) -> Any:
        return await self.fn(self._prepare(batch), *self.args, **self.kwargs)


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
        from batcher._internal.errors import PlanError
        from batcher.api.dataset.frame import Dataset
        from batcher.plan.expr_ir.core import AggExpr, Expr

        if isinstance(target, (Expr, AggExpr)):
            # Spark's `udf(f)(col)` shape. Running `fn` on the expression evaluated it once, at
            # plan time, on the expression object: `lambda s: s + 1` quietly built a plain
            # expression and `lambda s: s.upper()` raised an unrelated AttributeError.
            raise PlanError(
                "a @udf applies to a Dataset (or a batch), not to a column expression: "
                "write `my_udf(ds)`, or `ds.map_batches(fn)` for a batch function. A "
                "per-row column function is `ds.map(fn)`; most column logic is an "
                "expression (`bt.col(...)...`) and needs no UDF."
            )
        if not isinstance(target, Dataset):  # a batch or a row, not a Dataset: run the fn
            return self.fn(target)
        if self.per_row:
            return target.map(self.fn, **self.config)
        return target.map_batches(self.fn, **self.config)


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

    Raises:
        PlanError: If an option is not one the transform accepts.
    """
    # Checked at decoration, not at application. `**config` reaches `map_batches` only when
    # the `Udf` is finally called on a dataset, so a misspelled option used to surface as a
    # `TypeError` naming `Dataset.map_batches()` — a method the user never wrote — at
    # whatever line applied the transform, arbitrarily far from the decorator.
    from batcher.api.dataset._options import validate_map_options

    validate_map_options("@udf", config, per_row=per_row)

    def wrap(f: Callable) -> Udf:
        return Udf(f, per_row=per_row, config=config)

    return wrap(fn) if fn is not None else wrap
