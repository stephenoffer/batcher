"""Lazy EXACT column statistics over an immutable in-memory Arrow relation.

Split out of `io/source.py` on a responsibility seam: an in-memory source is immutable, so
its per-column min/max/null-count, distinct count, mean, sum, and per-value counts are
exact and constant. Computed once and reused, they are batcher's learned-metadata moat — an
unfiltered ``MIN``/``MAX``/``COUNT(DISTINCT)``/``AVG``/``SUM``/``COUNT(*) WHERE col = v`` is
answered from metadata on every *subsequent* run instead of re-scanning, which a static
optimizer cannot match.

These are pure compute helpers over a `build` callable that materializes one (narrow-int
widened) column; `InMemorySource` owns the batches, supplies `build`, and memoizes each
result per instance (never under its shape-based `identity`, which different data can
share). `pyarrow.compute` is imported at module top, so a source imports this module lazily
(inside the delegating method) to keep `pc`'s load cost off the cheap non-stats path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

if TYPE_CHECKING:
    from batcher.plan.source_stats import SourceStatistics
    from batcher.plan.stats import ColumnStat

# Materializes column `name` as one (widened) chunked array/array, or raises if absent.
ColumnBuilder = Callable[[str], "pa.ChunkedArray | pa.Array"]

# The Arrow errors a stat kernel can raise on an unsupported column — treated as
# "not derivable" (None / skip), never propagated: the metadata answer is optional.
_ARROW_ERRORS = (pa.ArrowInvalid, pa.ArrowNotImplementedError, KeyError)

# Types with a total order, so min/max is meaningful and `pc.min_max` has a kernel. Decimal
# belongs here and was missing: it is the type money is stored in, and without bounds a range
# predicate on a price column falls back to a prior instead of interpolating. `duration` is
# admitted by `is_temporal` but has no `min_max` kernel, so it still lands in the None path.
_ORDERED_TYPES = (
    pa.types.is_integer,
    pa.types.is_floating,
    pa.types.is_temporal,
    pa.types.is_decimal,
)


def _decoded(col: pa.ChunkedArray | pa.Array) -> pa.ChunkedArray | pa.Array:
    """A dictionary column's values as a plain array; anything else passes through.

    Every statistic in this module describes the relation the **engine** will execute over,
    and the engine decodes dictionary encoding at the FFI boundary: a `dictionary<string>`
    column collects back as plain `string`, 9.2 bytes per row where the encoded form was 4.0.
    A statistic computed on the encoded form would therefore describe a relation that never
    exists at runtime.

    Arrow's compute kernels agree by omission. `count_distinct`, `min_max`, `mean` and `sum`
    have no dictionary kernel and raise `ArrowNotImplementedError`, which `_ARROW_ERRORS`
    swallows -- so a categorical column silently lost its distinct count, its mean, its sum
    and its bounds together, all four at once. Dictionary encoding is the default for a
    low-cardinality column in Parquet and ORC and is what a pandas categorical arrives as, so
    this was most of the exact-statistics surface on a very common column shape.

    Args:
        col: The materialized column.

    Returns:
        `col` decoded when it is dictionary-encoded, otherwise `col` unchanged.
    """
    return col.cast(col.type.value_type) if pa.types.is_dictionary(col.type) else col


def _value_dtype(dtype: pa.DataType) -> pa.DataType:
    """The type a column's values have once decoded, seeing through dictionary encoding.

    `pa.types.is_integer(dictionary<values=int64, indices=int32>)` is `False`, so a type
    predicate applied to the encoded label rejects a column whose values are perfectly
    ordered integers. This is the same shape as an extension type hiding its storage type
    from `plan.types.widths` -- a predicate asked about the label rather than the values.

    Args:
        dtype: The declared column type.

    Returns:
        `dtype.value_type` for a dictionary type, otherwise `dtype` unchanged.
    """
    return dtype.value_type if pa.types.is_dictionary(dtype) else dtype


def column_bounds(build: ColumnBuilder, dtype: pa.DataType, name: str):
    """EXACT `ColumnStat` (min/max/null-count) for one column, or `None` if not derivable.

    A single vectorized ``min_max`` pass over an ordered type ([`_ORDERED_TYPES`]). Returns
    `None` for a non-ordered type (string/nested), an all-null column (its SQL ``MIN``/``MAX``
    is NULL — let a run return it), or an unsupported kernel — the same skips [`statistics`]
    makes per column.
    Float columns take [`_float_bounds`], whose NaN handling `pc.min_max` does not give us.
    """
    dtype = _value_dtype(dtype)
    if not any(ordered(dtype) for ordered in _ORDERED_TYPES):
        return None
    from batcher.plan.stats import ColumnStat, Provenance

    try:
        col = _decoded(build(name))
        if pa.types.is_floating(dtype):
            return _float_bounds(col)
        mm = pc.min_max(col, skip_nulls=True)
        lo, hi = mm["min"].as_py(), mm["max"].as_py()
    except _ARROW_ERRORS:
        return None
    if lo is None:  # all-null column
        return None
    return ColumnStat(min=lo, max=hi, null_count=col.null_count, provenance=Provenance.EXACT)


def _float_bounds(col: pa.Array):
    """Truthful float bounds under SQL's total order, where NaN is the **greatest** value.

    `pc.min_max` has no NaN policy we can rely on: over an all-NaN column it hands back the
    kernel's identity element (`+inf`/`-inf`), which is not a value in the column at all —
    so `min(f)` was answered from metadata as `inf` while executing the same query returned
    `nan`. A bound that is not a fact about the data is worse than no bound.

    So: NaN is excluded when computing the *minimum* (it can never be the smallest under a
    total order that makes it the largest), and the *maximum* is reported as NaN whenever any
    NaN is present, which is what it actually is. A column with no non-NaN value has no usable
    bound at all — return `None` and let a real run produce the answer.

    Args:
        col: The column to bound.

    Returns:
        An EXACT `ColumnStat`, or `None` when no sound bound exists.
    """
    from batcher.plan.stats import ColumnStat, Provenance

    non_null = col.drop_null()
    if len(non_null) == 0:  # all-null column — SQL MIN/MAX is NULL; let a run return it
        return None
    nan_mask = pc.is_nan(non_null)
    has_nan = bool(pc.any(nan_mask).as_py())
    finite = pc.filter(non_null, pc.invert(nan_mask))
    if len(finite) == 0:  # every value is NaN — no usable bound
        return None
    mm = pc.min_max(finite)
    return ColumnStat(
        min=mm["min"].as_py(),
        max=float("nan") if has_nan else mm["max"].as_py(),
        null_count=col.null_count,
        provenance=Provenance.EXACT,
    )


def statistics(build: ColumnBuilder, schema: pa.Schema, rows: int) -> SourceStatistics:
    """EXACT bounds where the type is ordered — and an EXACT null count for *every* column.

    A vectorized ``min_max`` per ordered column; strings and nested types have no bound worth
    recording (and an all-null column has none at all — a run returns its SQL-NULL ``MIN``).

    But every column, of every type, still gets its **null count**, and that is deliberate.
    Bounds and null counts are different facts with different requirements, and tying them
    together is what made `n_null("name")` scan a table whose null count was sitting in front of
    us: a string column produced no `ColumnStat` at all, so the one exact thing we knew about it
    was thrown away with the one we didn't. Arrow tracks `null_count` on the array — it is a
    field read, not a pass — so this costs nothing and makes `null_count()`, `count(name)`,
    `has_nulls`, and `dq.not_null` free on the columns most tables are actually made of.
    """
    from batcher.plan.source_stats import SourceStatistics

    columns: dict[str, ColumnStat] = {}
    for f in schema:
        stat = column_stat(build, f.type, f.name)
        if stat is not None:
            columns[f.name] = stat
    # `column_bounds` records NaN as the max when the column holds one (SQL ranks NaN
    # greatest), so unlike a Parquet footer these bounds *are* sound for `max(f)` and
    # for every float fact derived from them — declare it.
    return SourceStatistics(row_count=rows, columns=columns, bounds_include_nan=True)


def column_stat(build: ColumnBuilder, dtype: pa.DataType, name: str) -> ColumnStat | None:
    """Everything exactly known about one column: its bounds, or failing those its nulls.

    The single definition both callers share, and the reason it exists is that they had
    drifted. `statistics()` fell back to a null-count-only stat when a column had no
    trustworthy bounds — a string, a nested type, an all-null column — while
    `InMemorySource.column_bounds` returned `None` for exactly those columns and its
    docstring claimed that was "exactly as `statistics()` skips it". It is not what
    `statistics()` does, and `column_bounds` is the form the conductor actually calls: every
    real query narrows to the columns its predicates name (`_resident_subset_stats`), so the
    whole-relation path that got this right was the one nothing took.

    The cost was the exact fact, thrown away with the inexact one it sat beside. A string
    column of 1,000 rows with 100 nulls reported its null count on the raw path and nothing
    on the narrowed one, so `IS NULL` fell back to the 0.05 prior, and `n_null`, `count(col)`
    and `dq.not_null` lost the answer that was already in hand — on the column types most
    tables are made of.

    Args:
        build: The per-column array builder.
        dtype: The column's Arrow type.
        name: The column to describe.

    Returns:
        A `ColumnStat`, or `None` when nothing exact is derivable.
    """
    import dataclasses

    from batcher.plan.stats import ColumnStat, Provenance

    width = column_avg_bytes(build, name)
    stat = column_bounds(build, dtype, name)
    if stat is not None:
        return stat if width is None else dataclasses.replace(stat, avg_bytes=width)
    nulls = column_null_count(build, name)
    if nulls is None and width is None:
        return None
    # No trustworthy bounds, but an exact null count and/or an exact width.
    # `null_count_provenance` is what lets the second ride without the first.
    return ColumnStat(
        null_count=None if nulls is None else float(nulls),
        avg_bytes=width,
        provenance=Provenance.DEFAULT,
        null_count_provenance=Provenance.DEFAULT if nulls is None else Provenance.EXACT,
    )


def _dictionary_width(col: pa.ChunkedArray | pa.Array) -> float | None:
    """Bytes per row a dictionary column will occupy **once the engine decodes it**.

    The encoded buffers are the wrong number to report. The engine decodes at the FFI
    boundary, so a 12-value string dictionary that measures 4.0 bytes per row encoded runs at
    9.2 -- and reporting 4.0 under-sizes the memory envelope by 2.3x, which is the direction
    that OOMs. Dictionary encoding exists precisely because the column is low-cardinality, so
    this under-report lands on exactly the columns a categorical schema is made of.

    Measured off the dictionary's own values, whose count is the distinct count rather than
    the row count, so this stays O(distinct) and the cheap path stays cheap. That makes it an
    *estimate*: it takes the mean dictionary entry to be the mean decoded value, which is
    exact only when every entry appears equally often. Getting it exact needs the index
    distribution, which is an O(rows) pass -- the very thing the cheap path exists to avoid.

    The assumption is sound under *value* skew and breaks under **width** skew, where the
    frequent values and the wide values are different values: it reports 9.16667 against a
    9.16664 truth on a uniform dictionary but is 25.7x high on one shape and 0.51x low on its
    mirror image. The prior it replaces is 125x low on that same worst case, so it is better
    exactly where being wrong costs most. Measurements in the decision-quality ledger.

    Args:
        col: A dictionary-encoded column.

    Returns:
        Estimated decoded bytes per row, or `None` for an empty dictionary.
    """
    chunks = col.chunks if isinstance(col, pa.ChunkedArray) else [col]
    total = sum(c.dictionary.nbytes for c in chunks)
    entries = sum(len(c.dictionary) for c in chunks)
    return total / entries if entries else None


def _avg_bytes_of(col: pa.ChunkedArray | pa.Array) -> float | None:
    """Bytes per row `col` occupies as the engine will hold it. The one width computation.

    Args:
        col: The materialized column.

    Returns:
        Average bytes per row, or `None` for an empty column.
    """
    if pa.types.is_dictionary(col.type):
        return _dictionary_width(col)
    rows = len(col)
    return float(col.nbytes) / rows if rows else None


def column_cheap_stat(build: ColumnBuilder, name: str) -> ColumnStat | None:
    """Only what costs O(1): the null count and the average width. No bounds pass.

    The conductor narrows per-column statistics to the columns a plan's predicates name,
    because computing *bounds* is an O(rows) pass per column and a wide relation should not
    pay it for columns nothing reads. But that narrowing is keyed on `column_bounds_needed`
    and gates **everything**, so a plan with no predicate at all — a `group_by`, a plain scan
    — received no column statistics whatsoever.

    The width is not a bounds pass. Arrow tracks its buffer sizes, so it is a field read, and
    it is the number the memory envelope, the morsel row cap, broadcast eligibility and the
    task fan-out are all derived from. Measured: a `group_by` over a column of 2 KB documents
    priced its rows at the 36-byte string prior while the identical source under a *filter*
    reported the true 2,004.

    Both facts come off one materialization, because building the column twice more than
    doubled the cost of a call a wide relation makes once per column. A dictionary-encoded
    column is measured through [`_avg_bytes_of`], since the engine decodes it and the encoded
    size is not what runs.

    Args:
        build: The per-column array builder.
        name: The column to describe.

    Returns:
        A bounds-free `ColumnStat`, or `None` when neither fact is derivable.
    """
    from batcher.plan.stats import ColumnStat, Provenance

    try:
        col = build(name)
        nulls = float(col.null_count)
        width = _avg_bytes_of(col)
    except _ARROW_ERRORS:
        return None
    return ColumnStat(
        null_count=nulls,
        avg_bytes=width,
        provenance=Provenance.DEFAULT,
        null_count_provenance=Provenance.EXACT,
    )


def column_avg_bytes(build: ColumnBuilder, name: str) -> float | None:
    """EXACT average bytes per row of `name`, or None if not derivable.

    Arrow tracks its buffer sizes, so `nbytes / len` is a field read rather than a pass —
    the same argument `column_null_count` makes, and it costs microseconds on a
    half-megabyte column.

    It matters because the alternative is a *prior*. `plan.types.column_bytes` returns a
    flat 32-byte guess for any variable-length column, which is the only thing available
    before a row is read — and on text it is wrong by whatever the text actually is. Measured
    on a corpus of 2 KB documents: 36 B/row estimated against a true 2,004, a **56x**
    under-estimate of the number the memory envelope, the morsel row cap, broadcast
    eligibility and the task fan-out are all derived from. The same shape as sizing a decoded
    image column at 32 bytes, arriving through the other modality.

    A resident source is the one place this is free, which is exactly the argument that
    already justifies seeding distinct counts and heavy hitters there. `learn_column_stats`
    measures it after a run; this makes the *first* run right.

    Args:
        build: The per-column array builder.
        name: The column to measure.

    Returns:
        Average bytes per row, or `None` for an empty or unreadable column.
    """
    try:
        return _avg_bytes_of(build(name))
    except _ARROW_ERRORS:
        return None


def column_null_count(build: ColumnBuilder, name: str) -> int | None:
    """EXACT number of nulls in `name`, for a column of any type, or None if not derivable.

    Arrow maintains `null_count` on the array itself, so this is a field read per chunk rather
    than a pass over the values — free for any type, including the string and nested columns
    that have no bounds worth recording.
    """
    try:
        return int(build(name).null_count)
    except _ARROW_ERRORS:
        return None
