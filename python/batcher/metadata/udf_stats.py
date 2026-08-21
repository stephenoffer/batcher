"""Measured per-UDF execution cost — Core measures it, and two subsystems spend it.

Core already times a `map_batches` `fn` on a sample to decide how coarsely to batch its
calls (`core.udf.strategy`). That measurement — seconds of compute per row — answers a
second question nobody was asking it: **how expensive is this operator?** Kyber's cost model
priced every CPU `map_batches` as `map_row x rows`, the same as a trivial column map, so a
UDF running a hundred microseconds a row was the *cheapest* node in its plan and there was
no reason to push a selective filter below it. That is the exact optimization the GPU factor
beside it exists to produce, withheld from every stage that is expensive without being on an
accelerator.

This module is the neutral home the two subsystems meet in, following `io_stats`: `core`
records, `kyber` consumes, and neither imports the other (they must not — see
`.claude/rules/architecture.md`). The **key function lives here too**, because the value is
worthless if the writer and the reader spell the identity differently, and they nearly did:
Core keys a local/lambda `fn` by its defining line precisely because `module.qualname` is not
unique for one, while Kyber's cardinality identity omits the line and accepts the collision.
Reading across those two spellings would hand one lambda's measured cost to another.

## What actually seeds this table, and the size floor nobody chose for Kyber

The only writer is `core.udf.strategy._fn_row_seconds`, and it is reached from exactly two
places, both inside Core's *batch-sizing* decision: `thread_batch_target`, gated on
``total_rows > _PROBE_MIN_ROWS`` (262,144), and the process-vs-thread probe. So the per-row
cost Kyber's cost model reads is a **by-product of a decision about how coarsely to batch**,
and it inherits that decision's threshold.

The two want different things from it. Core's floor is right for Core: below it, coarsening
cannot change the batch count enough to matter, so a small query must not pay the probe's
latency. Kyber's question is not about batch counts at all — it is whether a `map_batches` is
a trivial column map or the most expensive node in the plan, which decides whether a selective
filter is pushed below it, and that is worth knowing at *every* size.

The consequence is narrow but real, and it is the shape this module's opening paragraph says
it closed: a workload whose inputs are always under ~262k rows never seeds this table, so its
`map_batches` stays priced as `map_row x rows` forever. Measured: three runs of a 200,000-row
`map_batches` record nothing; the same `fn` at 400,000 rows records on the first run, and the
value then serves queries of any size, because it is keyed by `fn` identity rather than by
input size.

Stated rather than changed. Lowering the floor means probing on small queries — running the
user's `fn` for real, several times, before the query has produced a row — and that latency is
exactly what `_PROBE_MIN_ROWS` exists to avoid. Which side of the trade is right is a
measurement (`python benchmarks/run.py` over a small-UDF workload), not an argument.
"""

from __future__ import annotations

from batcher._internal.logging import note_suppressed
from batcher.metadata.hardware_scope import scoped
from batcher.metadata.hub import MetadataHub
from batcher.metadata.smoothed import load_scalar, record_smoothed_scalar

__all__ = [
    "load_udf_row_seconds",
    "load_udf_row_seconds_table",
    "record_udf_row_seconds",
    "udf_cost_key",
]

# Scoped to the hardware fingerprint: seconds-per-row is a machine measurement in the sense
# `hardware_scope` defines — the same `fn` over the same rows costs differently on a different
# core count, clock, or memory system. Blending a laptop's figure with a worker node's would
# be wrong for both.
_NAMESPACE = "udf.row_seconds"


def udf_cost_key(fn: object) -> str | None:
    """A cross-run stable identity for a `map_batches` callable, or `None` if it has none.

    ``module.qualname`` for a named function or factory class; a callable *instance* is keyed
    by its class, since that is what determines its behaviour. A locally-defined callable —
    every lambda and closure — additionally carries its defining line, because
    ``mod.outer.<locals>.<lambda>`` is shared by every lambda in one enclosing function, and
    without the line two different UDFs written the ordinary way collide and inherit each
    other's measured cost.

    Args:
        fn: The `map_batches` callable (or factory class / callable instance).

    Returns:
        The identity string, or ``None`` when the callable carries no usable name.
    """
    qual = getattr(fn, "__qualname__", None)
    if qual is None:  # a callable instance: identify it by its type
        cls = type(fn)
        qual = getattr(cls, "__qualname__", None) or getattr(cls, "__name__", None)
        module = getattr(cls, "__module__", "") or ""
        code = None
    else:
        module = getattr(fn, "__module__", "") or ""
        code = getattr(fn, "__code__", None)
    if not qual or not module:
        return None
    key = f"{module}.{qual}"
    if "<locals>" not in qual and "<lambda>" not in qual:
        return key
    # A row adapter (`api.dataset.callbacks`) wraps a user callback and carries its identity
    # rather than being one, so it has no code object and supplies the line itself.
    line = code.co_firstlineno if code is not None else getattr(fn, "_batcher_def_line", None)
    return f"{key}:{line}" if line is not None else key


def record_udf_row_seconds(hub: MetadataHub | None, key: str | None, seconds: float) -> None:
    """Record one measurement of `key`'s per-row compute cost.

    Smoothed across runs, so one contended sample cannot jerk the figure. Best-effort: a
    failure to record never breaks the query that produced the measurement.

    Args:
        hub: The metadata hub to record into, or `None` to skip.
        key: The UDF identity from `udf_cost_key`, or `None` to skip.
        seconds: Measured seconds of compute per input row.
    """
    if hub is None or key is None or not (seconds > 0.0):
        return
    record_smoothed_scalar(hub, scoped(_NAMESPACE), key, float(seconds))


def load_udf_row_seconds(hub: MetadataHub | None, key: str | None) -> float | None:
    """The smoothed per-row compute cost recorded for `key`, or `None` when unmeasured.

    Args:
        hub: The metadata hub to read from, or `None`.
        key: The UDF identity from `udf_cost_key`, or `None`.

    Returns:
        Seconds of compute per input row, or ``None`` if nothing has been measured.
    """
    if hub is None or key is None:
        return None
    return load_scalar(hub, scoped(_NAMESPACE), key)


def load_udf_row_seconds_table(hub: MetadataHub | None) -> dict[str, float]:
    """Every measured per-row cost, keyed by UDF identity.

    The bulk form, for a consumer that folds the whole store into a planning bundle once
    rather than reading it per node.

    Args:
        hub: The metadata hub to read from, or `None`.

    Returns:
        A mapping of UDF identity to seconds per row; empty when nothing is measured.
    """
    if hub is None:
        return {}
    try:
        params = hub.load_keyed_params(scoped(_NAMESPACE))
    except Exception as exc:  # pragma: no cover - learning is best-effort
        # Learning must never break a query, and this is the read half of the loop that makes
        # plans improve across runs — so a failure here is exactly the one that can persist
        # for months with every gate green. Best-effort, not silent.
        note_suppressed("metadata", "read the measured UDF per-row costs", exc)
        return {}
    out: dict[str, float] = {}
    for key, entry in (params or {}).items():
        value = entry.get("value") if isinstance(entry, dict) else entry
        if isinstance(value, (int, float)) and value > 0:
            out[key] = float(value)
    return out
