"""Build the `MapBatches` stage behind `map_batches`, `map`, `flat_map` and `filter(fn)`.

The four verbs are one operator to the engine. `map_batches` hands the user's `fn` to it
directly; `map`/`flat_map` wrap a per-row `fn` in a row adapter that still runs a whole batch
per call inside the worker; `filter` wraps a batch predicate in a mask adapter. So every verb
resolves its Ray Data resource parameters the same way (`ray_options.resolve_placement`),
binds its extra arguments the same way (`bind_fn`), and validates the rest here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api.dataset._udf.checks import (
    normalize_resources,
    normalize_retry,
    require_number,
    validate_bindings,
    validate_column_list,
    validate_fn,
    validate_num_workers,
    validate_output_columns,
    warn_async_combos,
    warn_if_model_reloads,
    warn_if_pushdown_is_defeated,
)
from batcher.api.dataset._udf.ray_options import Placement
from batcher.plan.logical import MapBatches

if TYPE_CHECKING:
    from batcher.api.dataset import Dataset

__all__ = [
    "BINDING_PARAMS",
    "RAY_PARAMS",
    "ROW_OPTIONS",
    "bind_fn",
    "build_filter",
    "build_map_batches",
    "build_rows",
    "refuse_callable_options",
    "writable_format",
]

#: The four argument-binding parameters, in `bind_fn`'s order.
Bindings = tuple[Any, Any, Any, Any]
BINDING_PARAMS = ("fn_args", "fn_kwargs", "fn_constructor_args", "fn_constructor_kwargs")
#: Ray Data's resource parameters, as `ray_options.resolve_placement` takes them.
RAY_PARAMS = (
    "num_cpus",
    "num_gpus",
    "memory",
    "compute",
    "concurrency",
    "ray_remote_args",
    "ray_remote_args_fn",
)
#: The remaining options `map`/`flat_map` forward to `build_rows`.
ROW_OPTIONS = (
    "batch_size",
    "batch_format",
    "zero_copy_batch",
    "input_columns",
    "output_columns",
    "num_workers",
    "max_concurrency",
    "max_errored_rows",
)


def refuse_callable_options(method: Callable, given: dict[str, Any]) -> None:
    """Refuse the callable-only options of `method` when no callable was passed.

    `filter` takes an expression, a SQL string, or a callable, and its UDF options mean
    something only for the last. Accepting ``filter(col("x") > 0, num_gpus=1)`` would read as
    a scheduled predicate and run as an ordinary one, so any option that differs from its
    default is named and refused.

    Args:
        method: The method whose signature holds the defaults.
        given: The option values as the caller passed them.

    Raises:
        PlanError: If any option differs from its default.
    """
    import inspect

    params = inspect.signature(method).parameters
    set_names = sorted(name for name, value in given.items() if value != params[name].default)
    if set_names:
        raise PlanError(
            f"{method.__name__}() got {set_names}, which apply only to a callable predicate; "
            "drop them, or pass the condition as a function of the batch"
        )


def _check_format(verb: str, batch_format: str, allowed: tuple[str, ...]) -> None:
    """Reject a `batch_format` the verb cannot hand its callback, with a did-you-mean."""
    if batch_format in allowed:
        return
    from batcher._internal.errors import suggestion

    hint = suggestion(str(batch_format), allowed)
    tail = f" {hint}" if hint else ""
    raise PlanError(
        f"{verb}(batch_format=...) must be one of {sorted(allowed)}, got {batch_format!r}.{tail}"
    )


def writable_format(verb: str, batch_format: str, zero_copy_batch: object) -> str | None:
    """The format to copy read-only buffers out of, or `None` when the batch is handed over."""
    from batcher.api.dataset.callbacks import WRITABLE_FORMATS

    if not isinstance(zero_copy_batch, bool):
        raise PlanError(f"{verb}(zero_copy_batch=...) must be a bool, got {zero_copy_batch!r}")
    return None if zero_copy_batch or batch_format not in WRITABLE_FORMATS else batch_format


def build_map_batches(
    ds: Dataset,
    fn: Callable | type,
    *,
    verb: str = "map_batches",
    placement: Placement,
    batch_size: int | None,
    batch_format: str,
    input_columns: list[str] | None,
    preserves_columns: list[str] | None,
    output_columns: list[str] | None,
    num_workers: int | str,
    model_memory_gb: float = 0.0,
    multiprocessing: bool = False,
    max_errored_rows: int = 0,
    timeout: float = 0.0,
    max_retries: int = 0,
    retry_backoff: float = 0.5,
    retry_on: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    max_concurrency: int = 0,
) -> Dataset:
    """Validate a batch callable and derive the `MapBatches` stage that applies it.

    `fn` is already bound (`bind_fn`) and adapted for its verb, and `placement` is already
    resolved. Everything else is checked here, at the API edge.

    Returns:
        A new `Dataset` with the stage on top of `ds`.
    """
    from batcher.interop.formats import FORMATS
    from batcher.ml.devices import validate_batch_size, validate_num_gpus
    from batcher.ml.gpu import resolve_num_workers

    _check_format(verb, batch_format, FORMATS)
    validate_batch_size(batch_size)
    validate_num_gpus(placement.num_gpus)
    require_number(max_errored_rows, param="max_errored_rows", minimum=0, whole=True)
    require_number(model_memory_gb, param="model_memory_gb", minimum=0)
    resources = normalize_resources(placement.resources)
    timeout_s, retries, backoff_s, retry_types = normalize_retry(
        timeout, max_retries, retry_backoff, retry_on
    )
    require_number(max_concurrency, param="max_concurrency", minimum=0, whole=True)
    validate_num_workers(num_workers)
    validate_fn(fn)
    validate_output_columns(output_columns)
    available = _all_columns(ds)
    validate_column_list(input_columns, available, param="input_columns", verb=verb)
    validate_column_list(preserves_columns, available, param="preserves_columns", verb=verb)
    warn_async_combos(fn, multiprocessing, placement.num_gpus)
    warn_if_model_reloads(fn, placement.num_gpus)
    warn_if_pushdown_is_defeated(input_columns, ds.columns, output_columns)
    return ds._derive(
        MapBatches(
            ds._plan,
            fn,
            batch_size,
            tuple(output_columns) if output_columns is not None else None,
            input_columns=tuple(input_columns) if input_columns is not None else None,
            preserves_columns=(tuple(preserves_columns) if preserves_columns is not None else None),
            num_workers=resolve_num_workers(num_workers, placement.num_gpus),
            num_gpus=placement.num_gpus,
            concurrency=placement.concurrency,
            batch_format=batch_format,
            accelerator_type=placement.accelerator_type,
            resources=resources,
            model_memory_gb=model_memory_gb,
            multiprocessing=multiprocessing,
            max_errored_rows=max_errored_rows,
            max_retries=retries,
            retry_backoff_s=backoff_s,
            retry_on=retry_types,
            timeout_s=timeout_s,
            max_concurrency=max_concurrency,
        )
    )


def build_rows(
    ds: Dataset,
    fn: Callable | type,
    *,
    flat: bool,
    placement: Placement,
    bindings: Bindings,
    batch_size: int | None,
    batch_format: str,
    zero_copy_batch: bool,
    input_columns: list[str] | None,
    output_columns: list[str] | None,
    num_workers: int | str,
    max_concurrency: int,
    max_errored_rows: int,
) -> Dataset:
    """The `map`/`flat_map` stage: a per-row `fn` wrapped in its batch adapter.

    Returns:
        A new `Dataset` with the row stage on top of `ds`.
    """
    from batcher.api.dataset.callbacks import row_adapter

    verb = "flat_map" if flat else "map"
    validate_fn(fn)  # the adapter is itself callable, so check the user's fn before wrapping
    _check_format(verb, batch_format, ("pyarrow", "numpy"))
    writable = writable_format(verb, batch_format, zero_copy_batch) is not None
    cols = tuple(output_columns) if output_columns is not None else None
    adapter = row_adapter(
        bind_fn(fn, *bindings),
        cols,
        _row_concurrency(max_concurrency),
        flat=flat,
        fmt=batch_format,
        writable=writable,
    )
    return build_map_batches(
        ds,
        adapter,
        verb=verb,
        placement=placement,
        batch_size=batch_size,
        batch_format="pyarrow",
        input_columns=input_columns,
        preserves_columns=None,
        output_columns=output_columns,
        num_workers=num_workers,
        max_errored_rows=max_errored_rows,
    )


def build_filter(
    ds: Dataset,
    fn: Callable | type,
    *,
    placement: Placement,
    bindings: Bindings,
    batch_size: int | None,
    batch_format: str,
    zero_copy_batch: bool,
    input_columns: list[str] | None,
    num_workers: int | str,
    max_concurrency: int,
    max_errored_rows: int,
) -> Dataset:
    """The callable form of `filter`: a batch predicate wrapped in its mask adapter.

    Every input column is declared preserved, because the adapter only ever drops rows. That
    lets Kyber push a later expression filter below the Python one.

    Returns:
        A new `Dataset` holding the rows the predicate kept.
    """
    from batcher.api.dataset.callbacks import filter_adapter
    from batcher.interop.formats import FORMATS

    validate_fn(fn)
    _check_format("filter", batch_format, FORMATS)
    bound = bind_fn(fn, *bindings, writable_format("filter", batch_format, zero_copy_batch))
    read = tuple(input_columns) if input_columns is not None else None
    return build_map_batches(
        ds,
        filter_adapter(bound, batch_format, read),
        verb="filter",
        placement=placement,
        batch_size=batch_size,
        batch_format="pyarrow",
        input_columns=input_columns,
        preserves_columns=_all_columns(ds),
        output_columns=None,
        num_workers=num_workers,
        max_concurrency=max_concurrency,
        max_errored_rows=max_errored_rows,
    )


def _all_columns(ds: Dataset) -> list[str] | None:
    """Every input column, or `None` when the plan cannot name its columns without IO."""
    try:
        columns = ds._plan.available_columns()
    except Exception:  # an un-inferable schema declares nothing, which is the safe default
        return None
    return list(columns) or None


def _row_concurrency(max_concurrency: int) -> int:
    """The in-flight await bound for an async row callback, validated."""
    from batcher.api.dataset.callbacks import _DEFAULT_ROW_CONCURRENCY

    require_number(max_concurrency, param="max_concurrency", minimum=0, whole=True)
    return max_concurrency or _DEFAULT_ROW_CONCURRENCY


def bind_fn(
    fn: Callable | type,
    fn_args: tuple | None,
    fn_kwargs: dict | None,
    fn_constructor_args: tuple | None,
    fn_constructor_kwargs: dict | None,
    writable: str | None = None,
) -> Callable | type:
    """Bind extra call / constructor arguments onto `fn`, preserving load-once semantics.

    `fn_args`/`fn_kwargs` are forwarded to every batch call as ``fn(batch, *args, **kwargs)``,
    and `fn_constructor_args`/`fn_constructor_kwargs` to a class's one-per-worker
    construction (the Ray Data ``map_batches`` convention). A class stays a class after
    binding, so the engine still loads the model once per worker rather than per batch.

    The positional halves matter more than symmetry: the natural spelling of a model class is
    ``Classifier("bert-base-uncased", device="cuda")``, and with only the keyword forms a
    user whose ``__init__`` takes a positional checkpoint path had no way to pass it at all
    short of subclassing.

    `writable` names a batch format whose read-only buffers are copied before each call
    (``zero_copy_batch=False``); `None` hands the batch over as converted.
    """
    validate_bindings(fn_args, fn_kwargs, fn_constructor_args, fn_constructor_kwargs)
    fargs, fkw = tuple(fn_args or ()), fn_kwargs or {}
    cargs, ckw = tuple(fn_constructor_args or ()), fn_constructor_kwargs or {}
    if (cargs or ckw) and not isinstance(fn, type):
        raise PlanError(
            "fn_constructor_args/fn_constructor_kwargs only apply to a class fn (loaded once "
            f"per worker); got {type(fn).__name__}. Pass a class, or move the values into "
            "fn_args/fn_kwargs."
        )
    if not (fargs or fkw or cargs or ckw or writable):
        return fn
    if isinstance(fn, type):
        return _bound_model(fn, cargs, ckw, fargs, fkw, writable)
    from batcher.api.dataset.callbacks import _AsyncBoundBatchFn, _BoundBatchFn
    from batcher.core.udf.async_udf import is_async_udf

    binder = _AsyncBoundBatchFn if is_async_udf(fn) else _BoundBatchFn
    return binder(fn, fargs, fkw, writable)


def _bound_model(
    base: type, cargs: tuple, ckw: dict, fargs: tuple, fkw: dict, writable: str | None
) -> type:
    """A class that builds `base(*cargs, **ckw)` once and calls it with `fargs`/`fkw`.

    Still a class, so `build_udf_callable` keeps instantiating it exactly once per worker —
    binding arguments must never turn a load-once model into a per-batch reload.

    The wrapper forwards `close()`, which the previous version did not: `teardown_udf` looks
    for `close` on the *built* object, found none on the wrapper, and silently skipped the
    teardown of every model configured with `fn_constructor_kwargs`. A load-once model's
    `close` is exactly where a GPU allocation or an HTTP session is released, so the
    difference showed up as VRAM that never came back between partitions.
    """
    from batcher.api.dataset.callbacks import writable_batch
    from batcher.core.udf.async_udf import is_async_udf

    def prepare(batch: object) -> object:
        return batch if writable is None else writable_batch(batch, writable)

    class _Bound:
        def __init__(self) -> None:
            self._inner = base(*cargs, **ckw)

        def close(self) -> None:
            close = getattr(self._inner, "close", None)
            if callable(close):
                close()

    if is_async_udf(base):
        # An async model must stay async through the wrapper, or the coroutine `__call__`
        # returns is never awaited (it is routed to the sync path and coerced as garbage).
        class _BoundModel(_Bound):
            async def __call__(self, batch: object) -> object:
                return await self._inner(prepare(batch), *fargs, **fkw)

    else:

        class _BoundModel(_Bound):  # type: ignore[no-redef]
            def __call__(self, batch: object) -> object:
                return self._inner(prepare(batch), *fargs, **fkw)

    _BoundModel.__name__ = f"Bound{base.__name__}"
    _BoundModel.__qualname__ = _BoundModel.__name__
    return _BoundModel
