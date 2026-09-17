"""Edge validation and advisory warnings for the batch-UDF verbs.

`Dataset.map_batches`, `map`, `flat_map` and the callable form of `filter` all lower to one
`MapBatches` node, so they share one set of checks: every option is validated here, at the
API edge, rather than failing opaquely inside a worker, and the performance foot-guns a
Python callback invites (an undeclared read set over a wide table, a GPU model rebuilt per
batch, an async `fn` given knobs it ignores) are named at the call site that caused them.
"""

from __future__ import annotations

__all__ = [
    "normalize_resources",
    "normalize_retry",
    "require_number",
    "validate_fn",
    "validate_output_columns",
    "warn_async_combos",
    "warn_if_model_reloads",
    "warn_if_pushdown_is_defeated",
]


def require_number(value: object, *, param: str, minimum: float, whole: bool = False) -> None:
    """Reject a non-numeric or out-of-range `map_batches` option, naming it.

    The retry options were range-checked but not *type*-checked, so a string or `None` got
    as far as the comparison and raised Python's own
    ``'<' not supported between instances of 'str' and 'int'`` — which names neither the
    option nor what it wanted. `model_memory_gb` and `max_errored_rows` were not checked at
    all: a negative or non-numeric model size fed the resource layer and Kyber's cost model
    silently, and a fractional error budget was accepted as if it meant something.
    """
    from batcher._internal.errors import PlanError

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(
            f"{param} must be a number >= {minimum:g}, got {type(value).__name__} {value!r}."
        )
    if value < minimum:
        raise PlanError(f"{param} must be >= {minimum:g}, got {value!r}.")
    if whole and float(value) != int(value):
        raise PlanError(f"{param} must be a whole number, got {value!r}.")


def normalize_resources(resources: object) -> tuple[tuple[str, float], ...]:
    """Validate and normalize the custom-resource request into the node's tuple form.

    These names and amounts go straight to Ray's scheduler, so a negative amount, a
    non-numeric one, or a non-string name is a request that can never be satisfied — and
    every one of them was accepted in silence. A non-dict got as far as `.items()` and
    raised `AttributeError`.
    """
    from batcher._internal.errors import PlanError

    if resources is None:
        return ()
    if not isinstance(resources, dict):
        raise PlanError(
            f"resources must be a {{name: amount}} dict, e.g. {{'TPU': 4}}, got "
            f"{type(resources).__name__}."
        )
    for name, amount in resources.items():
        if not isinstance(name, str) or not name:
            raise PlanError(f"resources keys must be non-empty resource names, got {name!r}.")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount <= 0:
            raise PlanError(
                f"resources[{name!r}] must be a positive number of units, got {amount!r}."
            )
    return tuple(sorted((str(n), float(a)) for n, a in resources.items()))


def normalize_retry(
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    retry_on: type[BaseException] | tuple[type[BaseException], ...] | None,
) -> tuple[float, int, float, tuple[type[BaseException], ...]]:
    """Validate and normalize the `map_batches` retry/timeout options at the API edge.

    Turns a deferred, opaque failure deep in the worker into an eager `PlanError` here, and
    coerces `retry_on` (a single exception type, a tuple of them, or ``None``) to the tuple the
    `MapBatches` node stores. Every exception type must be a `BaseException` subclass.
    """
    from batcher._internal.errors import PlanError

    require_number(timeout, param="timeout", minimum=0)
    require_number(max_retries, param="max_retries", minimum=0, whole=True)
    require_number(retry_backoff, param="retry_backoff", minimum=0)
    if retry_on is None:
        types: tuple[type[BaseException], ...] = ()
    else:
        types = retry_on if isinstance(retry_on, tuple) else (retry_on,)
    for t in types:
        if not (isinstance(t, type) and issubclass(t, BaseException)):
            raise PlanError(f"retry_on must be an exception type or a tuple of them, got {t!r}")
    return float(timeout), int(max_retries), float(retry_backoff), types


def validate_fn(fn: object) -> None:
    """Reject a `map_batches` `fn` that cannot be called, eagerly at the API edge.

    Turns the two common foot-guns into an actionable `PlanError` here instead of a deferred,
    opaque failure deep in a worker: a non-callable object, and a class whose instances are not
    callable (a model class that forgot ``def __call__(self, batch)``, so loading it once per
    worker leaves nothing to score each batch).
    """
    from batcher._internal.errors import PlanError

    if isinstance(fn, type):
        if not any("__call__" in klass.__dict__ for klass in fn.__mro__ if klass is not object):
            raise PlanError(
                f"map_batches got the class {fn.__name__!r}, but its instances are not callable. "
                "Define __call__(self, batch) so the model loaded once per worker can score each "
                "batch, or pass a function instead."
            )
        return
    if not callable(fn):
        raise PlanError(
            "map_batches fn must be callable — a function, or a class to load once per worker; "
            f"got {type(fn).__name__}."
        )


def validate_output_columns(
    output_columns: list[str] | None, *, param: str = "output_columns"
) -> None:
    """Reject an empty, non-string, or duplicated `output_columns` name at the API edge.

    A duplicate or blank output name otherwise surfaces as an opaque Arrow schema error deep in
    the engine (or worse, a silently shadowed column); catching it here names the offender.
    """
    if output_columns is None:
        return
    from batcher._internal.errors import PlanError

    if len(output_columns) == 0:
        # An empty list is stored as a non-None () and makes the plan believe the stage produces
        # zero columns (`available_columns() == []`) while the `fn` actually keeps the input
        # schema — a silent plan/execution mismatch. Use None to mean "unchanged".
        raise PlanError(
            f"{param} cannot be empty; pass None to keep the input columns, or list the "
            "columns the fn produces."
        )
    seen: set[str] = set()
    for name in output_columns:
        if not isinstance(name, str) or not name:
            raise PlanError(f"{param} must be non-empty strings, got {name!r}")
        if name in seen:
            raise PlanError(f"{param} has a duplicate column name {name!r}")
        seen.add(name)


def warn_async_combos(fn: object, multiprocessing: bool, num_gpus: float) -> None:
    """Warn about knobs an ``async def`` `fn` silently ignores.

    Async runs on one event loop — its point is overlapping I/O awaits, not filling cores or a
    device. `multiprocessing=True` (the process pool) is never used, and the GPU auto-batching /
    autocast a synchronous `num_gpus` stage gets are skipped. Surfacing the ignored intent beats
    dropping it silently, since the user asked for a behavior they will not get.
    """
    if not (multiprocessing or num_gpus > 0):
        return
    from batcher.core.udf.async_udf import is_async_udf

    if not is_async_udf(fn):
        return
    import warnings

    from batcher._internal.errors import PerformanceWarning

    if multiprocessing:
        warnings.warn(
            "map_batches got an async fn with multiprocessing=True; async runs on one event "
            "loop and never uses the process pool, so multiprocessing is ignored. Drop it, or "
            "pass a synchronous fn to run CPU-bound work across processes.",
            PerformanceWarning,
            stacklevel=4,
        )
    if num_gpus > 0:
        warnings.warn(
            "map_batches got an async fn with num_gpus > 0; the GPU auto-batching and autocast "
            "that a synchronous GPU stage gets are skipped on the async event-loop path. Use a "
            "synchronous class fn for a GPU model, or async only for I/O-bound (API) work.",
            PerformanceWarning,
            stacklevel=4,
        )


#: Column count above which an undeclared `input_columns` is worth a warning. Below it the
#: unpruned read costs little and the advice would be noise on every narrow table; the field
#: guides put the interesting range at "wide tables (50+ columns), 10-50x I/O difference",
#: and 12 is where a scan is already reading several columns nothing downstream will touch.
_WIDE_TABLE_COLUMNS = 12


def warn_if_pushdown_is_defeated(
    input_columns: object, columns: list[str], output_columns: object
) -> None:
    """Warn when an opaque UDF over a wide table forces the scan to read every column.

    Projection pushdown is the single highest-impact IO optimization in the field guides
    (2-10x on a wide table, 10-50x past 50 columns), and it is the one Batcher does
    automatically — right up to a `map_batches`. The `fn` is a Python callback, so the
    optimizer cannot see which columns it reads and must assume *all* of them; the scan then
    reads the whole table to feed a stage that may touch two columns.

    `input_columns` is the declaration that restores it, and there is no way to infer it. So
    the one case where Batcher's automatic pushdown silently stops working is worth saying
    out loud, at the call site that caused it, rather than leaving it to be discovered in a
    profile.

    A stage whose `output_columns` carry every input column through is exempt: it genuinely
    needs all of them, so there is nothing to declare and the advice would be wrong. That is
    the shape of every append-a-column UDF — `ds.ml.generate`, `embed`, `classify` — which
    would otherwise be told to prune columns it is contractually obliged to return.
    """
    if input_columns is not None or len(columns) < _WIDE_TABLE_COLUMNS:
        return
    if output_columns is not None and set(columns) <= set(output_columns):
        return
    import warnings

    from batcher._internal.errors import PerformanceWarning

    warnings.warn(
        f"map_batches over {len(columns)} columns did not declare input_columns, so the "
        f"optimizer must assume the fn reads all of them and the scan cannot prune. Pass "
        f"input_columns=[...] naming what the fn actually reads to restore projection "
        f"pushdown.",
        PerformanceWarning,
        stacklevel=4,
    )


def warn_if_model_reloads(fn: object, num_gpus: float) -> None:
    """Warn when a GPU stage gets a plain function (rebuilt per batch → model reload).

    Passing a class/factory instead loads the model once per worker (the GPU-inference
    pattern); a plain function is re-created on every batch — the most common Ray Data
    inference foot-gun.
    """
    if num_gpus > 0 and not isinstance(fn, type):
        import warnings

        from batcher._internal.errors import PerformanceWarning

        warnings.warn(
            "map_batches got a plain function with num_gpus > 0; the model will be "
            "re-created on every batch (reloaded each time). Pass a class/factory "
            "instead so it loads once per worker (the GPU-inference pattern).",
            PerformanceWarning,
            stacklevel=4,
        )
