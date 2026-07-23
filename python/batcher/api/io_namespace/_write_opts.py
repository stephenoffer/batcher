"""The save-mode and keyword vocabulary `ds.write` accepts, normalized in one place.

Users arrive at `ds.write` from Spark and from pandas, and both spell the same three
concepts differently: the save mode, the partition columns, and "one file please". Every
one of those spellings used to be rejected — ``mode="errorIfExists"`` is *Spark's own
documented name* for a mode Batcher has, and it raised — so a mechanical port failed at
the last line of the pipeline, which is the most expensive place to fail.

Normalizing here rather than in `Writer.__call__` keeps one table for the whole surface:
`.parquet`, `.csv`, `.delta` and the rest all funnel through `__call__`, so a spelling
accepted here is accepted by every typed method, and there is no second list to update.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MODE_AWARE_SINKS",
    "SAVE_MODES",
    "normalize_partition_by",
    "normalize_save_mode",
    "reject_row_index",
]

# Save modes (Spark `SaveMode` parity). `append` is only meaningful for the sinks that
# can add to an existing target — the transactional lakehouse tables and the warehouse
# tables — which consume `mode` as a constructor option; the file sinks always overwrite,
# so for them `mode` only drives the existence gate.
#
# A sink that honors `mode` MUST be listed here. `snowflake` was not, and the result was
# the worst of both: `mode="append"` was rejected even though `SnowflakeSink` implements
# it, and `mode="overwrite"` passed the gate but never reached the sink, so the write
# quietly appended instead. A save mode that silently does the opposite of what it says is
# a data-corruption bug, not a missing feature.
SAVE_MODES = ("overwrite", "error", "ignore", "append")
MODE_AWARE_SINKS = frozenset({"delta", "iceberg", "hudi", "snowflake"})

#: The spellings other engines use for the same four modes. `errorifexists` is the one
#: that matters most: it is what Spark's own documentation calls the mode, so a ported job
#: hits it before it hits anything else. The single letters are Python's file modes, which
#: is what a pandas user reaches for.
_MODE_ALIASES = {
    "errorifexists": "error",
    "error_if_exists": "error",
    "w": "overwrite",
    "write": "overwrite",
    "a": "append",
    "x": "error",
}

#: The keywords that mean `partition_by`: Spark's camelCase and pandas' `to_parquet` name.
_PARTITION_ALIASES = ("partitionBy", "partition_cols")


def normalize_save_mode(mode: str) -> str:
    """Resolve a save mode, accepting the Spark and pandas spellings of the same four.

    Matching is case-insensitive, as Spark's is, so ``"ErrorIfExists"`` resolves like
    ``"errorifexists"``.

    Args:
        mode: The save mode the caller named.

    Returns:
        One of `SAVE_MODES`.

    Raises:
        PlanError: If `mode` is not a mode or a known alias for one.
    """
    from batcher._internal.errors import PlanError, unknown_value

    if not isinstance(mode, str):
        raise unknown_value(PlanError, "save mode", mode, SAVE_MODES, label="Accepted save modes")
    folded = mode.strip().lower()
    if folded in SAVE_MODES:
        return folded
    resolved = _MODE_ALIASES.get(folded)
    if resolved is not None:
        return resolved
    raise unknown_value(
        PlanError,
        "save mode",
        mode,
        (*SAVE_MODES, *_MODE_ALIASES),
        label="Accepted save modes (aliases included)",
        hint="'overwrite', 'error', 'ignore', and 'append' are the canonical four.",
    )


def normalize_partition_by(
    opts: dict[str, Any], partition_by: list[str] | None
) -> list[str] | None:
    """Fold the `partitionBy` / `partition_cols` aliases into `partition_by`, popping them.

    Args:
        opts: The write's keyword options. Any alias found is removed, so it cannot also
            reach the sink as an unknown option.
        partition_by: The value passed under the canonical name, if any.

    Returns:
        The partition columns, or None when none were named.

    Raises:
        PlanError: If two spellings are passed at once with different values.
    """
    from batcher._internal.errors import PlanError

    found: dict[str, list[str]] = {}
    for alias in _PARTITION_ALIASES:
        if alias in opts:
            value = opts.pop(alias)
            found[alias] = [value] if isinstance(value, str) else list(value)
    if partition_by is not None:
        found["partition_by"] = [partition_by] if isinstance(partition_by, str) else partition_by
    if not found:
        return None
    distinct = {tuple(v) for v in found.values()}
    if len(distinct) > 1:
        rendered = ", ".join(f"{k}={v!r}" for k, v in found.items())
        raise PlanError(
            f"write(): {rendered} name different partition columns. They are three "
            "spellings of one argument — pass whichever you like, but only one."
        )
    return list(next(iter(distinct)))


def reject_row_index(opts: dict[str, Any]) -> None:
    """Accept and drop pandas' ``index=False``; refuse ``index=True`` with the reason.

    ``to_parquet(path, index=False)`` is reflex for a pandas user, and it asks for exactly
    what Batcher already does — there is no row index to write — so it is dropped rather
    than raised on. ``index=True`` asks for something that does not exist, and silently
    ignoring *that* would hand back a file missing a column the caller believes is in it.

    Only a **boolean** ``index`` is claimed. A connector whose own option happens to be
    called ``index`` (an Elasticsearch index name, say) passes a string, and swallowing
    that here would break the connector to serve a pandas habit.

    Args:
        opts: The write's keyword options. A boolean ``index`` is removed.

    Raises:
        PlanError: If ``index=True`` was passed.
    """
    if not isinstance(opts.get("index"), bool):
        return
    from batcher._internal.errors import PlanError

    if opts.pop("index"):
        raise PlanError(
            "write(index=True): Batcher has no row index, so there is nothing to write. "
            "Drop index=, or add the numbering as a real column before writing so it is "
            "written like every other column."
        )
