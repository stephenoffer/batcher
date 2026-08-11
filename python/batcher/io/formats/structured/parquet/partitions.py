"""What a Hive ``col=value`` directory segment means, and what it proves.

The directory layer of a partitioned Parquet tree, separated from the source that reads it
because it is a different question. `dataset` answers "how do I read these files"; this
answers "what does this path segment stand for, and what can be stated about a column whose
values are path segments rather than data".

That second question is worth its own module because the answer is used three times over and
must agree with itself every time. The value a segment decodes to has to be the same value a
worker appends to its rows, the same value a predicate is pruned against, and the same value
that bounds the column for the optimizer — a segment that decodes one way for reading and
another for pruning does not read too much, it drops rows.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from batcher._internal.logging import note_suppressed

if TYPE_CHECKING:
    from batcher.plan.stats import ColumnStat

__all__ = [
    "HIVE_NULL",
    "date_typed_partitioning",
    "partition_bounds",
    "partitioning_arg",
    "typed_partition_value",
]

#: A Hive segment value that is a calendar date. Anchored and fixed-width, so ``2024-1-1``
#: and ``2024-01-01T00:00`` are both left as the strings they are.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: The value a Hive writer puts in the path for a NULL key. It is not a date, and it must
#: not veto the promotion — pyarrow maps it back to null for whatever type the field has.
HIVE_NULL = "__HIVE_DEFAULT_PARTITION__"


def _all_dates(values: Any) -> bool:
    """Whether every observed value of one partition key is a calendar date.

    Args:
        values: The key's observed segment values, as pyarrow discovery reports them.

    Returns:
        True when there is at least one real date and nothing that is not one.
    """
    if values is None:
        return False
    seen = [v for v in values.to_pylist() if v is not None and v != HIVE_NULL]
    if not seen:
        return False
    from datetime import date

    for value in seen:
        if not _ISO_DATE.match(value):
            return False
        try:
            date.fromisoformat(value)
        except ValueError:
            return False
    return True


def date_typed_partitioning(dataset: Any) -> bytes | None:
    """A partitioning schema that reads date-valued keys as dates, or None to keep discovery.

    ``partition_by=["day"]`` on a date column is the most common Hive layout there is, and
    reading it back gave a **string**, because pyarrow's discovery infers integers and
    nothing else. The cost was not cosmetic: ``filter(col("day") == date(2024, 2, 10))`` —
    the obvious query against that layout — died with
    ``Function 'equal' has no kernel matching input types (string, date32[day])``, an Arrow
    kernel error that names neither the partition column nor the cast that fixes it. DuckDB
    returns DATE for the same tree, and Spark infers partition column types by default, so
    this was also the one place a ported job silently needed a rewrite.

    The values come from `dataset.partitioning.dictionaries`, which discovery has **already**
    collected while walking the tree — so deciding costs no extra listing, and a tree with no
    date-valued key pays nothing at all, since it returns None and the caller keeps the
    dataset it already built.

    Promotion requires *every* observed value of the key to be a calendar date, so a key that
    is date-like in one branch and not in another stays a string rather than failing to parse
    later. `HIVE_NULL` is excluded from the vote, not counted against it.

    Args:
        dataset: A `pyarrow.dataset.Dataset` built with Hive discovery.

    Returns:
        The serialized partitioning schema to rebuild with, or None to keep what discovery
        produced. Serialized rather than live because it travels on a split to a worker and
        is part of a cache key, so it has to be both picklable and hashable.
    """
    part = getattr(dataset, "partitioning", None)
    schema = getattr(part, "schema", None)
    if schema is None or not len(schema):
        return None
    dictionaries = list(getattr(part, "dictionaries", None) or [])
    fields, promoted = [], False
    for index, field in enumerate(schema):
        values = dictionaries[index] if index < len(dictionaries) else None
        if field.type == pa.string() and _all_dates(values):
            fields.append(pa.field(field.name, pa.date32()))
            promoted = True
        else:
            fields.append(field)
    return pa.schema(fields).serialize().to_pybytes() if promoted else None


def partitioning_arg(partitioning: Any) -> Any:
    """The ``partitioning=`` value for `pyarrow.dataset`, from a string or a serialized schema.

    Args:
        partitioning: Either a discovery name such as ``"hive"``, or the bytes
            `date_typed_partitioning` returned.

    Returns:
        The value to pass to `pyarrow.dataset.dataset`.
    """
    if not isinstance(partitioning, bytes):
        return partitioning
    import pyarrow.dataset as pads

    return pads.HivePartitioning(pa.ipc.read_schema(pa.BufferReader(partitioning)))


def typed_partition_value(raw: str, target: pa.DataType) -> Any:
    """The Python value a Hive directory segment stands for, typed by the dataset schema.

    `raw` is the segment exactly as it appears in the path, which the writer URL-encoded
    (``x/y`` → ``x%2Fy``); the `pyarrow.dataset` read path URI-decodes it, so every other
    reader of the same tree must too or a distributed read returns the encoded spelling
    where a single-node read returns the real one.

    Shared by the split that *carries* a partition value to a worker, the planner that
    *prunes* on it, and the bounds that *describe* it, because a value that decodes one way
    for reading and another for pruning would drop rows rather than merely read too many.

    Args:
        raw: The ``value`` half of a ``col=value`` path segment.
        target: The type the dataset schema gives the partition column.

    Returns:
        The decoded, typed value, or None for the Hive NULL sentinel.
    """
    if raw == HIVE_NULL:
        return None
    from urllib.parse import unquote

    return pa.scalar(unquote(raw), pa.string()).cast(target).as_py()


def partition_bounds(
    dirs: list[tuple[str, tuple[str, str]]], schema: pa.Schema
) -> dict[str, ColumnStat]:
    """``min``/``max``/``ndv`` for the top-level partition column, or ``{}``.

    The column a table is partitioned by is the one a query most wants to prune on, and it
    was the one column with **no statistics at all** — its values live in directory names
    rather than in any file footer, so the footer sweep that stats every other column cannot
    see it. The consequence was not a worse estimate but a missing optimization:
    `kyber.rules.joins.runtime_join_filter` pushes a ``key BETWEEN other_min AND other_max``
    onto a join's prunable side *only when both sides' ranges are known*, so a star join
    against a partitioned fact table — the shape the rule exists for, and the shape its own
    docstring calls "dynamic partition pruning" — could never fire. A ten-day fact table
    joined to a two-day dimension read all ten.

    These are **bounds, not attained values**, and are tagged `DEFAULT` for that reason — the
    same downgrade a `Filter` applies for the same cause. A partition *directory* can outlive
    its rows: a rewrite deletes the data files and leaves ``dt=x`` standing, which is exactly
    what `io.filesystem.prune_empty_dirs` exists to clean up. So the true minimum may sit
    inside this range rather than on its edge. Over-approximating is sound for everything
    that matters here — pruning may only ever keep too much, and a join filter derived from a
    wider range is still a superset filter — while claiming `EXACT` would let a metadata
    shortcut answer ``MIN(day)`` with a day that holds no rows.

    `ndv` is the directory count for the same reason: an upper bound on the distinct values,
    not a measurement of them.

    Only the top level is described. Deeper levels would each cost their own listing, and the
    top level is both the cheapest and the one plan-time directory pruning acts on.

    Args:
        dirs: ``(dir, (key, raw_value))`` for each top-level partition directory.
        schema: The dataset schema, which types the partition values.

    Returns:
        ``{column: ColumnStat}`` for the top-level partition key, or ``{}`` when the tree is
        not partitioned or its values will not type.
    """
    from batcher.plan.stats import ColumnStat, Provenance

    if not dirs:
        return {}
    key = dirs[0][1][0]
    try:
        target = schema.field(key).type
        values = [typed_partition_value(raw, target) for _, (_, raw) in dirs]
    except Exception as exc:
        note_suppressed("io", "type the partition values for their bounds", exc)
        return {}
    present = [v for v in values if v is not None]
    if not present:
        return {}
    try:
        low, high = min(present), max(present)
    except TypeError as exc:  # values that do not order against each other
        note_suppressed("io", "order the partition values for their bounds", exc)
        return {}
    return {
        key: ColumnStat(
            min=low, max=high, ndv=float(len(set(values))), provenance=Provenance.DEFAULT
        )
    }
