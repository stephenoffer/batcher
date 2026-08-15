"""The per-batch `map_batches` call boundary (Core, layer 3).

One Arrow batch in, Arrow batches out: this module owns everything that happens
*around* a single user `fn` call — reframing the batch to the requested
`batch_format`, isolating a failing batch by bisection (CUDA-OOM halving and
dirty-row tolerance), and normalizing whatever the `fn` returns back to
`RecordBatch`es. The callers own the *scheduling* of those calls: `execute`
walks the plan tree, `stream` overlaps the stages, `strategy` picks threads vs
processes — all three share this one boundary, so a UDF behaves identically on
every path.
"""

from __future__ import annotations

import threading
from typing import Any

import pyarrow as pa

__all__: list[str] = ["shared_error_budget"]

# The `max_errored_rows` allowances, one shared list per (`fn` identity, allowance) per
# worker process — see `shared_error_budget`. It lives here, at the per-batch boundary every
# path funnels through, because two of those paths need it and cannot see each other:
# `strategy` (threads, on the driver) and `processes._child_call` (inside a pool worker).
_ERROR_BUDGETS: dict[tuple[str, int], list[int]] = {}
_BUDGET_LOCK = threading.Lock()


def shared_error_budget(key: str, allowance: int) -> list[int]:
    """The ``[remaining, dropped]`` budget shared by every call to `key`'s UDF in this process.

    The public contract says `max_errored_rows` is "per worker". The code did not mean that:
    a fresh ``[allowance]`` was built per invocation, so each partition, each streaming
    window, and each execution path a single query routes through got its OWN full allowance
    — the real bound grew with parallelism, which is the reverse of what a budget is for.
    One list per (`fn`, allowance) pair per process makes the code mean the documented thing.

    It is deliberately NOT global across workers: a cluster-wide counter would need a round
    trip per dropped row on the hot path. So the honest cluster bound is
    ``workers x max_errored_rows``, and that is what the public docstring says.

    Args:
        key: A stable identity for the UDF, shared by every path that runs it.
        allowance: The stage's `max_errored_rows`.

    Returns:
        The mutable budget list `_resilient_call` draws down.
    """
    with _BUDGET_LOCK:
        return _ERROR_BUDGETS.setdefault((key, allowance), [allowance])


def _resilient_call(
    call, sub: pa.RecordBatch, budget: list[int], is_gpu: bool
) -> list[pa.RecordBatch]:
    """Run a per-batch `call`, isolating failures by bisection — the unified OOM-halving +
    dirty-data-tolerance path.

    On a CUDA OOM (GPU stage) the batch is halved and retried (a too-large batch often fits at
    N/2; the per-row-independent outputs concatenate to the whole result); a single row that
    still OOMs is a genuine over-allocation and re-raises. On any OTHER error the batch is
    bisected to isolate the offending row(s): a failing single row is DROPPED (charged against
    `budget`, the ``max_errored_rows`` allowance) so a corrupt image / malformed record doesn't
    kill a long job — until the budget is exhausted, when it re-raises. With ``budget == 0``
    and a CPU stage this reduces to strict behavior (any error propagates), so a real bug on
    clean data still fails fast.

    Every drop is **surfaced**, because a silently vanishing row is unrecoverable: at PB scale
    you cannot tell afterwards which rows were skipped or how many. `budget` therefore doubles
    as the out-parameter: ``budget[0]`` is the remaining allowance (as before) and ``budget[1]``
    is the running drop count, appended on the first drop so existing callers keep passing a
    one-element list unchanged. Each drop also publishes to the observability bus with the
    running count and the error text, so a running job reports the loss as it happens rather
    than at the end."""
    try:
        return _coerce_udf_result(call(sub))
    except Exception as exc:
        # Imported here, not at the top of the function: this is the only branch that needs
        # it, and on a clean batch the lookup — and the ML/torch import behind it — is work
        # every single call was paying for an error path it never took.
        from batcher.ml.inference import _empty_cuda_cache, _is_cuda_oom

        oom = is_gpu and _is_cuda_oom(exc)
        if oom:
            _empty_cuda_cache()
        if sub.num_rows <= 1:
            if oom or not _claim_dropped_row(budget, exc):
                raise  # genuine single-row over-allocation, or the error budget is spent
            return []  # drop the one corrupt row and carry on
        mid = sub.num_rows // 2
        left = _resilient_call(call, sub.slice(0, mid), budget, is_gpu)
        return left + _resilient_call(call, sub.slice(mid), budget, is_gpu)


#: Guards the whole allowance. `_resilient_call` runs under a `ThreadPoolExecutor` on the
#: `execute` path, so several threads can drop a row at the same instant; without this the
#: read-modify-write would lose counts and the lazy append could run twice. Taken only on the
#: (rare) drop path, so the clean path stays lock-free.
_DROP_LOCK = threading.Lock()


def _claim_dropped_row(budget: list[int], exc: Exception) -> bool:
    """Take one row off the shared allowance, reporting whether it was still there.

    The **check and the decrement are one atomic step**, which is the whole point. They used
    to straddle the lock — `budget[0] <= 0` was read, and `budget[0] -= 1` applied, with only
    the *counter* update inside it. Two threads reaching the last allowance both saw it, both
    took it, and the stage dropped more rows than `max_errored_rows` permits. A budget that
    parallelism can overshoot is not a budget, and the overshoot is silent by construction:
    the rows are gone and the count that would have shown it is the one that was raced.

    Only the exception's type and message are reported, never the row's values: the row that
    failed is frequently the one carrying malformed or sensitive data, and an observability
    sink is not an appropriate place to leak it.

    Args:
        budget: The shared ``[remaining, dropped]`` list (see `shared_error_budget`).
        exc: The failure that made this row undroppable-in-place.

    Returns:
        True when a drop was granted; False when the allowance is spent and the caller
        must re-raise.
    """
    from batcher._internal.events import LOG, MALFORMED, publish

    with _DROP_LOCK:
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        if len(budget) < 2:
            budget.append(0)
        budget[1] += 1
        dropped = budget[1]
        remaining = budget[0]
    publish(
        LOG,
        name="map_batches",
        dropped_rows=dropped,
        remaining_budget=remaining,
        error=f"{type(exc).__name__}: {exc}",
    )
    # Also as a *countable* drop, not only as a log line. The `LOG` event above carries the
    # running total in a field, which `observe` folds into a per-level log count and nothing
    # else — so the one number a fleet needs to alert on ("how many rows did this job throw
    # away?") existed on the bus and reached no metric. `MALFORMED` is the same unit the IO
    # readers publish for a record they could not parse, and `source` is what tells the two
    # apart, so one series answers the question across the whole pipeline.
    publish(
        MALFORMED,
        name="map_batches",
        count=1,
        reason=type(exc).__name__,
        source="map_batches",
    )
    return True


def _formatted(fn: Any, fmt: str) -> Any:
    """Wrap `fn` so it receives/returns `fmt` batches while the caller stays Arrow."""
    from batcher.interop.formats import result_to_arrowable, to_format

    def _call(batch: pa.RecordBatch) -> object:
        return result_to_arrowable(fn(to_format(batch, fmt)), fmt)

    return _call


def _coerce_udf_result(result: object) -> list[pa.RecordBatch]:
    """Normalize a `map_batches` return to Arrow batches the engine can import.

    Two steps, deliberately separate. `_coerce_parts` decides *what shape* the user
    returned; `importable_batch` then respells any Arrow **layout** the FFI reader cannot
    take. A UDF is the one place a column type is chosen by user code rather than by a
    source, so it is exactly where a layout the engine has never seen arrives -- a
    ``list_view`` built by a NumPy or Polars round-trip used to reach the user as
    ``Extracting byte ranges not supported for type list_view<item: int64>``.

    Args:
        result: Whatever the user's `fn` returned.

    Returns:
        Record batches, every column of which arrow-rs can import.
    """
    from batcher.plan.types.layout import importable_batch

    return [importable_batch(b) for b in _coerce_parts(result)]


def _coerce_parts(result: object) -> list[pa.RecordBatch]:
    """Normalize a `map_batches` return to Arrow batches.

    Accepts, in the order a user is likely to produce them: a `RecordBatch`, a `Table`, a
    ``{column: values}`` dict, a pandas or polars `DataFrame`, and any list/tuple/generator
    of those. The frame and iterator forms matter more than they look — a `fn` written
    against ``batch_format="pandas"`` and then reused under the default ``"pyarrow"``
    returns a frame, and a **generator** `fn` (``yield`` one batch per decoded video / per
    LLM response) is the natural spelling of a row-expanding ML stage. Both used to reach
    the user as ``must return a pyarrow RecordBatch, Table, or dict; got DataFrame``, which
    names the type it got and nothing about the fix.

    A generator is materialized here rather than streamed: the caller's contract is a
    ``list[RecordBatch]`` per input batch, so the memory bound is the same as returning one
    concatenated `Table`, and `batch_size` remains the knob that bounds it.
    """
    if isinstance(result, pa.RecordBatch):
        return [result]
    if isinstance(result, pa.Table):
        # A 0-row Table yields *no* batches, which would drop the stage's output schema (the
        # parent falls back to the input schema and a downstream ref to a UDF-added column
        # fails). Keep one empty batch so the schema survives, like a 0-row RecordBatch does.
        batches = result.to_batches()
        if batches:
            return batches
        cols = [pa.array([], type=f.type) for f in result.schema]
        return [pa.RecordBatch.from_arrays(cols, schema=result.schema)]
    if isinstance(result, dict):
        columns = _tensorize_columns(result)
        try:
            return [pa.RecordBatch.from_pydict(columns)]
        except Exception as exc:  # a column Arrow cannot type — say which, and what to do
            _raise_unconvertible_column(columns, exc)
    framed = _frame_to_arrow(result)
    if framed is not None:
        return _coerce_parts(framed)
    parts = _iterable_parts(result)
    if parts is not None:
        out: list[pa.RecordBatch] = []
        for part in parts:
            out.extend(_coerce_parts(part))
        return out
    raise TypeError(
        "map_batches function must return a pyarrow RecordBatch or Table, a "
        "{column: values} dict, a pandas/polars DataFrame, or an iterable of those; "
        f"got {type(result).__name__}"
    )


def _frame_to_arrow(result: object) -> pa.Table | None:
    """A pandas or polars `DataFrame` converted to an Arrow `Table`, else ``None``.

    Detected by module and class name rather than by importing pandas/polars, so a UDF
    returning one costs nothing for the users who have neither installed.
    """
    cls = type(result)
    if cls.__name__ != "DataFrame":
        return None
    root = cls.__module__.split(".")[0]
    if root == "pandas":
        return pa.Table.from_pandas(result, preserve_index=False)
    if root == "polars":
        return result.to_arrow()  # type: ignore[attr-defined]
    return None


def _iterable_parts(result: object) -> list | None:
    """`result` as a list of parts when it is a list/tuple/iterator of batch-likes.

    Returns ``None`` for anything that is not a batch container, so a NumPy array or a
    string still falls through to the type error rather than being walked element by
    element. A list of row dicts is rejected with its own message: it is a plausible
    mistake with a different fix (`flat_map`, or one column dict), and silently treating
    each row as a one-row batch would be catastrophically slow rather than wrong.
    """
    from collections.abc import Iterator

    if not isinstance(result, list | tuple | Iterator):
        return None
    parts = list(result)
    if parts and all(isinstance(p, dict) for p in parts) and _looks_row_oriented(parts):
        raise TypeError(
            "map_batches function returned a list of row dicts. Return one batch instead — "
            "a {column: list_of_values} dict, a pyarrow RecordBatch, or a DataFrame — or use "
            "`ds.ml.flat_map` if the function is genuinely row-at-a-time."
        )
    return parts


def _looks_row_oriented(parts: list) -> bool:
    """Whether `parts` are row dicts (every value a scalar) rather than column dicts.

    A column dict's values are sequences (list, ndarray, Arrow array, Series); a row dict's
    are scalars. Strings and bytes count as scalars despite having ``__len__``, which is the
    whole reason this is a named predicate rather than a `hasattr` inline.
    """
    return not any(_column_like(value) for value in parts[0].values())


def _column_like(value: object) -> bool:
    """Whether `value` is a sequence of row values rather than one row's scalar."""
    return not isinstance(value, str | bytes) and hasattr(value, "__len__")


def _check_declared_columns(batches: list[pa.RecordBatch], op: Any) -> None:
    """Fail when a stage's output does not have the columns its `output_columns` declared.

    `output_columns` is a promise to the optimizer: `MapBatches.available_columns()` returns it
    verbatim, so every operator above the stage plans against it. Nothing checked it, so a
    typo'd or stale declaration produced a plan that believed in a column the `fn` never
    emitted. The symptom lands far away and reads like an engine bug — a downstream `select`
    resolving a column that "exists" in the plan, or a schema that unifies to null — rather
    than at the one line that made the false claim.

    Compared as sets on the first non-empty batch: order is not what the plan relies on, and a
    single name comparison per stage costs nothing. An all-empty result is not evidence of
    anything and is left alone, which also keeps a filtered-to-zero upstream from failing here.

    **A declared `input_columns` narrows the check to extra columns only.** That declaration
    is what lets projection pushdown prune the scan, and what it prunes is precisely the
    columns the `fn` merely passes through — so a pass-through name in `output_columns` is
    legitimately absent from the result whenever nothing above the stage still wanted it. The
    canonical example in the user guide does exactly this, and treating it as a mismatch would
    reject the idiom the same guide recommends. With `input_columns` unset the optimizer must
    keep every input column alive, so there is nothing to explain away and both directions are
    checked.
    """
    declared = getattr(op, "output_columns", None)
    if not declared:
        return
    produced = next((b.schema.names for b in batches if b.num_rows), None)
    if produced is None:
        return
    prunable = getattr(op, "input_columns", None) is not None
    missing = [] if prunable else sorted(set(declared) - set(produced))
    extra = sorted(set(produced) - set(declared))
    if not missing and not extra:
        return
    from batcher._internal.errors import PlanError

    parts = []
    if missing:
        parts.append(f"declared but not returned: {missing}")
    if extra:
        parts.append(f"returned but not declared: {extra}")
    raise PlanError(
        f"map_batches output_columns does not match what the function returned "
        f"({'; '.join(parts)}). Every operator above this stage plans against the declared "
        f"names, so the mismatch would surface far from here. Fix the declaration, or drop "
        f"output_columns to keep the input schema."
    )


def _tensorize_columns(result: dict) -> dict:
    """Turn a multi-dimensional NumPy value into a tensor column, fixed- or variable-shape.

    A `map_batches` `fn` (image decode, embedding, feature-map) commonly returns a
    ``(B, *shape)`` NumPy array per column — the Ray Data tensor-block shape.
    ``from_pydict`` can't build a column from a >1-D array, so multi-dim values are
    converted to the canonical ``arrow.fixed_shape_tensor`` column (`to_tensor_column`),
    which round-trips zero-copy through the FFI with its shape intact.

    A **list of arrays of differing shape** — the mixed-resolution decode, and the one shape
    the canonical type cannot express — becomes a variable-shape tensor column instead
    (`ragged_from_values`). That case used to be the multimodal path's first hard stop: it
    could not be typed at all, and the only advice was to resize before the engine saw it.

    1-D arrays, lists, and Arrow arrays pass through untouched, so scalar/label columns are
    unchanged. This keeps the tensor path identical single-node and distributed, for every
    modality.
    """
    import numpy as np

    from batcher.io.formats.ml.ragged import ragged_from_values
    from batcher.io.formats.ml.tensor import tensor_from_values, to_tensor_column

    converted: dict = {}
    for name, value in result.items():
        if isinstance(value, np.ndarray) and value.ndim >= 2:
            converted[name] = to_tensor_column(value)
            continue
        # A NumPy *object* array counts as a sequence of arrays here, not as one array. It is
        # exactly what `batch_format="numpy"` hands a `fn` for a variable-shape tensor column,
        # so an identity `fn` over one returned a value the tensor path did not recognize and
        # Arrow could not type — the round trip failed on the column the round trip is for.
        listed = list(value) if isinstance(value, np.ndarray) and value.dtype == object else value
        if isinstance(listed, list | tuple):
            tensor = tensor_from_values(listed) or ragged_from_values(listed)
            converted[name] = value if tensor is None else tensor
            continue
        converted[name] = value
    return converted


def _raise_unconvertible_column(columns: dict, cause: Exception) -> None:
    """Re-raise a failed batch conversion as a typed error naming the column and the fix.

    A UDF that returns PIL Images, torch tensors, UUIDs, enum members, or any custom object
    hands Arrow a value it cannot type. pyarrow's own message quotes the offending *value*
    and its class, but names neither the column it came from nor `map_batches` nor what to do
    — and for a multimodal or Python-object pipeline that is exactly the error a user hits
    first.

    Ray Data has the opposite failure here and it is worse: it silently falls back to a
    pickle-backed object column, which the field guides flag as a 10-100x slowdown on every
    downstream transfer and expect the user to catch by eyeballing `ds.schema()`. Failing
    loudly is the right call; failing loudly *and* unhelpfully is not.

    The diagnosis itself lives in `interop.diagnostics`, shared with the constructors on the
    other side of the boundary — the same value is just as unconvertible on the way in.
    """
    from batcher._internal.errors import PlanError
    from batcher.interop.diagnostics import describe_unconvertible, find_unconvertible_column

    name = find_unconvertible_column(columns)
    if name is None:
        ragged = _uneven_lengths(columns)
        if ragged is not None:
            raise PlanError(
                f"map_batches returned columns of different lengths ({ragged}). Every column "
                f"of a batch has one row count; a column built from a filtered or partial "
                f"result is the usual cause."
            ) from cause
        raise PlanError(f"map_batches returned a batch Arrow cannot represent: {cause}") from cause
    raise PlanError(
        f"map_batches returned a batch where {describe_unconvertible(name, columns[name])} "
        f"Every column crossing the engine boundary must be an Arrow type; an opaque Python "
        f"object would otherwise be pickled, which is far slower for every stage downstream."
    ) from cause


def _uneven_lengths(columns: dict) -> str | None:
    """``"a=2, b=1"`` when a returned dict's columns disagree on length, else `None`.

    pyarrow reports the mismatch as ``Arrays were not all the same length: 1 vs 2``, which
    for a twenty-column result says only that two of them differ. Naming each column and its
    length points at the one that is short — usually the one built from a filtered or partial
    result while its siblings came straight off the input.
    """
    lengths: dict[str, int] = {}
    for name, value in columns.items():
        try:
            lengths[name] = len(value)
        except TypeError:
            return None  # a scalar among the columns is a different mistake
    if len(set(lengths.values())) < 2:
        return None
    return ", ".join(f"{name}={n}" for name, n in lengths.items())
