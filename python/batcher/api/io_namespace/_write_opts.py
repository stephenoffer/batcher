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
    "DATABASE_SINKS",
    "MODE_AWARE_SINKS",
    "SAVE_MODES",
    "derive_partition_columns",
    "dml_write_modes",
    "normalize_partition_by",
    "normalize_save_mode",
    "one_or_many",
    "partition_key_name",
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
SAVE_MODES = ("overwrite", "overwrite_partitions", "error", "ignore", "append")
MODE_AWARE_SINKS = frozenset(
    {
        "delta",
        "iceberg",
        "hudi",
        "snowflake",
        "adbc",
        "dbapi",
        "mongo",
        "dynamodb",
        "cassandra",
        "redis",
        "elasticsearch",
        "hbase",
    }
)

#: Sinks whose destination is a database table rather than a path in a filesystem.
#:
#: Two of `Writer.__call__`'s steps are filesystem operations dressed as mode handling: the
#: ``error``/``ignore`` gate asks whether the destination *path* exists, and the overwrite
#: cleanup deletes stale files under it. Neither means anything for a table name, and the
#: first was actively wrong — ``mode="error"`` against a table that already held rows asked
#: the local filesystem whether a file called ``orders`` existed, got False, and wrote
#: anyway. The gate that exists to refuse an overwrite silently permitted one.
DATABASE_SINKS = frozenset(
    {
        "adbc",
        "dbapi",
        "mongo",
        "snowflake",
        "dynamodb",
        "cassandra",
        "redis",
        "elasticsearch",
        "hbase",
    }
)

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
    # Dynamic partition overwrite. Spark spells it as a session conf
    # (`partitionOverwriteMode=dynamic`) rather than a save mode, and Hive/Trino as
    # `INSERT OVERWRITE`, so all three spellings resolve to the one mode here.
    "dynamic": "overwrite_partitions",
    "overwritepartitions": "overwrite_partitions",
    "insert_overwrite": "overwrite_partitions",
}

#: The keywords that mean `partition_by`: Spark's camelCase and pandas' `to_parquet` name.
_PARTITION_ALIASES = ("partitionBy", "partition_cols")


def normalize_save_mode(mode: str) -> str:
    """Resolve a save mode, accepting the Spark and pandas spellings of each.

    Matching is case-insensitive, as Spark's is, so ``"ErrorIfExists"`` resolves like
    ``"errorifexists"``, and Spark's ``partitionOverwriteMode="dynamic"`` and Hive's
    ``INSERT OVERWRITE`` both name ``"overwrite_partitions"``.

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
        hint=(
            "'overwrite', 'error', 'ignore', and 'append' are the canonical four; "
            "'overwrite_partitions' is the dynamic partition overwrite."
        ),
    )


def partition_key_name(key: Any) -> str:
    """The output column name a partition key contributes to the Hive path.

    A key is a column name, a bare column expression, or any expression carrying an
    `alias`. The alias is what makes a *derived* key usable: a Hive path segment is
    ``name=value``, so an expression with no name has no segment to write into, and
    guessing one from the expression's shape would make ``dt.year()`` and
    ``dt.year() + 1`` fight over the same directory.

    Args:
        key: One entry of a `partition_by` list.

    Returns:
        The column name this key partitions by.

    Raises:
        PlanError: If the key is an expression with no name to write into the path.
    """
    from batcher._internal.errors import PlanError

    if isinstance(key, str):
        return key
    name = getattr(key, "name", None)
    if isinstance(name, str):
        return name
    raise PlanError(
        "write(partition_by=...): a derived partition key needs a name, because it "
        "becomes a 'name=value' directory. Add one with .alias(), e.g. "
        "partition_by=[bt.col('ts').dt.year().alias('year')]."
    )


def derive_partition_columns(dataset: Any, partition_by: list[Any]) -> tuple[Any, list[str]]:
    """Add any expression-valued partition keys as columns, returning plain names.

    This is how Batcher spells a *partition transform* — Iceberg's ``days(ts)`` /
    ``bucket(16, id)`` and Spark's generated partition column. The transform is an
    ordinary expression evaluated once by the engine, and the resulting column is
    partitioned on by name like any other. Nothing new is needed on the read side: the
    Hive writer already keeps a partition column in the *path* rather than the file, so
    the derived column costs no bytes and comes back as a real column on read, prunable
    by a filter like any other partition key.

    Args:
        dataset: The `Dataset` about to be written.
        partition_by: The partition keys, a mix of names and expressions.

    Returns:
        A ``(dataset, names)`` pair. `dataset` carries the derived columns; `names` is
        the plain-string key list every layer below the API works in.
    """
    names = [partition_key_name(k) for k in partition_by]
    derived = {
        name: key for name, key in zip(names, partition_by, strict=True) if not isinstance(key, str)
    }
    return (dataset.with_columns(**derived) if derived else dataset), names


def one_or_many(value: Any) -> list[Any]:
    """Read a key option as a list, treating a bare string or expression as a single key.

    Every "which columns" option on `write` takes a list, and users pass one key without
    the brackets. Iterating it then splits it into *characters*, which for `sort_by` was
    not an error but a wrong answer: ``sort_by="ab"`` on a frame holding columns `a`, `b`
    and `ab` sorted by `a` then `b`, silently clustering the output on keys nobody named
    and leaving every zonemap the option exists to tighten pointing the wrong way.

    Args:
        value: One key, or a list/tuple of them.

    Returns:
        The keys as a list.

    Examples:
        .. doctest::

            >>> from batcher.api.io_namespace._write_opts import one_or_many
            >>> one_or_many("dt"), one_or_many(["a", "b"])
            (['dt'], ['a', 'b'])
    """
    # A bare str or a lone expression is one key, not an iterable of them: `list(expr)`
    # would either raise or, worse, iterate something expression-shaped.
    one = isinstance(value, str) or not isinstance(value, list | tuple)
    return [value] if one else [*value]


def normalize_partition_by(
    opts: dict[str, Any], partition_by: list[str] | None
) -> list[str] | None:
    """Fold the `partitionBy` / `partition_cols` aliases into `partition_by`, popping them.

    A key may be an expression rather than a column name (a partition transform), so the
    spellings are compared by the *names* they resolve to rather than by value: an `Expr`
    is deliberately unhashable — `==` on one builds a comparison rather than answering a
    question — so a set of raw key tuples would raise instead of comparing.

    Args:
        opts: The write's keyword options. Any alias found is removed, so it cannot also
            reach the sink as an unknown option.
        partition_by: The value passed under the canonical name, if any.

    Returns:
        The partition keys, or None when none were named.

    Raises:
        PlanError: If two spellings are passed at once with different values.
    """
    from batcher._internal.errors import PlanError

    found: dict[str, list[Any]] = {}
    for alias in _PARTITION_ALIASES:
        if alias in opts:
            found[alias] = one_or_many(opts.pop(alias))
    if partition_by is not None:
        found["partition_by"] = one_or_many(partition_by)
    if not found:
        return None
    distinct = {tuple(partition_key_name(k) for k in v) for v in found.values()}
    if len(distinct) > 1:
        rendered = ", ".join(f"{k}={v!r}" for k, v in found.items())
        raise PlanError(
            f"write(): {rendered} name different partition columns. They are three "
            "spellings of one argument — pass whichever you like, but only one."
        )
    return next(iter(found.values()))


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


def dml_write_modes(fmt: str | None) -> tuple[str, ...]:
    """The row-level DML verbs `fmt`'s sink accepts as its `mode`, or empty for none.

    A save mode says what to do with the *table* — replace it, add to it, refuse. A
    database table is also maintained one row at a time, and ``upsert`` / ``update`` /
    ``delete`` are not save modes in any sense: they name what happens to the rows the
    write's keys match, and leave every other row alone. Spark has no spelling for them at
    all, which is why a JDBC upsert is written by hand there.

    A sink that implements them declares them in a ``dml_modes`` class attribute, the same
    way one declares its keyword vocabulary in ``write_spec``. `Writer.__call__` then passes
    such a mode through verbatim rather than running it past `normalize_save_mode`, which
    would reject it as a misspelled save mode.

    Args:
        fmt: The sink format name, or None when the format is still being detected from
            the path — in which case there is no sink to ask and no DML mode to honor.

    Returns:
        The accepted DML modes, or an empty tuple.

    Examples:
        .. doctest::

            >>> from batcher.api.io_namespace._write_opts import dml_write_modes
            >>> "upsert" in dml_write_modes("dbapi")
            True
            >>> dml_write_modes("parquet")
            ()
    """
    if not fmt:
        return ()
    from batcher.io.formats import SINKS

    sink_cls = SINKS.get(fmt) if fmt in SINKS else None
    modes = getattr(sink_cls, "dml_modes", ())
    return tuple(modes) if isinstance(modes, tuple | list) else ()
