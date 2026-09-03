"""Persisted source statistics — remember what Batcher wrote, for the next read.

When Batcher writes a dataset it knows the result exactly: the row count and byte
size, and (cheaply, from the result already in memory) per-column distinct counts.
A later read of that path can then be answered from metadata even for formats with
no footer (CSV, JSON), instead of re-scanning. This module persists a
`SourceStatistics` into the `MetadataHub`, keyed by the source's stable
`identity()`, and loads it back on read.

It stores only JSON-safe scalars and tags row counts `exact_rows=True` (the write
counted every row) while distinct counts stay `SKETCH` (HLL-derived). Best-effort
throughout: persistence never breaks a write, and a missing/garbled entry simply
falls through to a normal read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from batcher._internal.logging import note_suppressed
from batcher.metadata.hub import MetadataHub
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance, SortOrder, as_sort_orders

__all__ = ["load_source_stats", "save_source_stats"]

_NAMESPACE = "io.source_stats"
_JSON_SCALARS = (int, float, str, bool)

# Ceilings on the two variable-length descriptive stats, so one pathological column cannot
# grow the hub without bound. The two are capped in *opposite* ways, and the asymmetry is the
# point: dropping the tail of an MCV table only removes the least common of the most common
# values, which weakens the estimate and cannot make it wrong, so an oversized table is
# truncated. A quantile grid has no such property — truncating it silently misdescribes the
# CDF's upper tail, and a range predicate above the cut then interpolates against a maximum
# that is not the column's. An oversized grid is therefore dropped whole, which costs the
# cold-start estimate and keeps every retained grid honest.
_MAX_QUANTILE_POINTS = 1024
_MAX_MCV_ENTRIES = 256


def save_source_stats(
    hub: MetadataHub, identity: str, stats: SourceStatistics, *, version: str | None = None
) -> None:
    """Persist `stats` for a source `identity`. Best-effort; never raises.

    `version` is the source's content token at the moment of writing (see
    `io.stats.file_identity.files_version`). It is what lets a later read tell "these
    statistics describe the bytes I am about to read" from "these statistics describe
    whatever used to be at this path", which for a *membership index* is the difference
    between skipping data and deleting rows. An entry saved without one is still readable;
    it simply cannot satisfy a versioned load.

    Args:
        hub: The metadata hub to persist into.
        identity: The source's stable identity, the key the statistics are filed under.
        stats: The statistics to persist.
        version: The source's content version, when the caller can compute one.
    """
    try:
        hub.save_params(f"{_NAMESPACE}:{identity}", _encode(stats, version))
    except Exception as exc:  # persistence must never break a write
        # Noted, not swallowed — the read half of this module already does, and the write half
        # is where silence lasts. A failed save is indistinguishable from a source Batcher has
        # never written: `load_source_stats` returns `None`, the reader falls through to a full
        # scan, and every later read does the same. That is the difference `note_suppressed`
        # exists to make visible, and it matters most for the formats this module was written
        # for — CSV and JSON have no footer, so a lost entry is a re-scan of the whole file
        # rather than a slightly worse estimate.
        note_suppressed("metadata", "persist source statistics", exc)


def load_source_stats(
    hub: MetadataHub, identity: str, *, require_version: str | None = None
) -> SourceStatistics | None:
    """Load persisted statistics for a source `identity`, or None if absent.

    With `require_version`, an entry is returned only when it was saved under exactly that
    content version — so an entry describing a path some *other* writer has since rewritten
    is not returned at all, rather than returned and trusted. An entry saved before versions
    were recorded carries none and therefore never matches, which is the safe direction: it
    falls back to whatever the caller does without persisted statistics.

    Args:
        hub: The metadata hub to read from.
        identity: The source's stable identity.
        require_version: The content version the entry must have been saved under. Omit to
            accept any entry, which is sound only for facets that merely inform cost.

    Returns:
        The persisted statistics, or None when absent, unreadable, or version-mismatched.
    """
    try:
        blob = hub.load_params(f"{_NAMESPACE}:{identity}")
    except Exception as exc:
        note_suppressed("metadata", "load source stats", exc)
        return None
    if not blob:
        return None
    if require_version is not None and blob.get("version") != require_version:
        return None
    return _decode(blob)


def _encode(stats: SourceStatistics, version: str | None = None) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for name, col in stats.columns.items():
        encoded = _encode_column(col)
        if encoded:
            columns[name] = encoded
    return {
        # The content version these statistics describe, when the writer could compute one.
        # `_decode` ignores it; only `load_source_stats(require_version=...)` reads it, and
        # its absence simply means a versioned load cannot be satisfied from this entry.
        "version": version,
        "row_count": stats.row_count,
        "byte_size": stats.byte_size,
        "exact_rows": stats.exact_rows,
        # The three qualifiers that say how the figures above may be *used*. Each defaults
        # to the conservative answer, so dropping one from the round trip does not produce a
        # wrong result — it silently produces a worse plan on every reload, which is harder
        # to notice: `content_byte_size` decides whether the width estimator may trust
        # `byte_size` at all (a media corpus reverts to a 36-byte type prior without it),
        # `bounds_include_nan` decides whether a float `max()` can be answered from bounds
        # rather than executed, and `row_group_count` is what lets a prune report what it
        # skipped without reading.
        "content_byte_size": stats.content_byte_size,
        "bounds_include_nan": stats.bounds_include_nan,
        "row_group_count": stats.row_group_count,
        # `sorted_by` drives redundant-sort removal: it reaches `RelStats.sorted_by` through
        # `to_relstats`, and without it a Batcher-written footerless source (CSV/JSON) loses
        # its ordering on reload and re-sorts data that is already sorted.
        "sorted_by": [_encode_sort_key(k) for k in as_sort_orders(stats.sorted_by)],
        # `partition_keys` is round-tripped for `ds.meta.storage`, and **not** for pruning —
        # a claim this comment used to make. Hive partition pruning happens in the reader:
        # the pushed predicate goes to `pyarrow.dataset`, which skips fragments by partition
        # value before Kyber sees a row. No optimizer rule reads this field, so if one is
        # ever written, that is the moment to check whether the reader already did the job.
        "partition_keys": list(stats.partition_keys),
        "columns": columns,
    }


def _encode_sort_key(key: SortOrder) -> str | dict[str, Any]:
    """One ordering key, as the bare column name when it is plain ascending, nulls-last.

    The compact form is the overwhelmingly common one and is exactly what earlier stores
    wrote, so a store written now stays readable by the same reader that read those — and
    the round trip does not inflate every entry to carry two defaults.
    """
    if not key.descending and not key.nulls_first:
        return key.column
    return {"column": key.column, "descending": key.descending, "nulls_first": key.nulls_first}


def _decode_sort_key(blob: str | dict[str, Any]) -> SortOrder:
    """One ordering key from either encoding — a bare name, or the explicit object."""
    if isinstance(blob, str):
        return SortOrder(blob)
    return SortOrder(
        str(blob["column"]),
        bool(blob.get("descending", False)),
        bool(blob.get("nulls_first", False)),
    )


def _encode_column(col: ColumnStat) -> dict[str, Any]:
    out: dict[str, Any] = {"provenance": col.provenance.name}
    for field in ("min", "max", "null_count", "ndv", "total_sum", "mean", "avg_bytes"):
        value = getattr(col, field)
        if isinstance(value, _JSON_SCALARS):
            out[field] = value
    if col.bloom is not None:  # the data-skipping membership index, base64 for JSON
        import base64

        out["bloom"] = base64.b64encode(col.bloom).decode("ascii")
    # The distributional pair. They are what answers a *cold* range or equality predicate:
    # without them a first-ever read of a written path falls back to the Selinger range
    # constant and `1/ndv`, which is the regime `TPCH_FINDINGS` measured the largest
    # cold-start wins against. Both are already JSON-native, so persisting them costs the
    # bytes and nothing else.
    grid = _encode_quantiles(col.quantiles)
    if grid is not None:
        out["quantiles"] = grid
    mcv = _encode_mcv(col.mcv)
    if mcv is not None:
        out["mcv"] = mcv
    # Drop a bare provenance with no usable values.
    if len(out) == 1:
        return {}
    # Per-field provenance sub-tags. `ndv` and `null_count` each carry their OWN trust
    # tag, which the bundle's cannot express: a sketch distinct count rides beside exact
    # bounds (`ndv_provenance` weaker), and an exact null count rides beside byte-truncated
    # bounds (`null_count_provenance` stronger). Dropping them makes `ndv_is_exact` /
    # `null_count_is_exact` fall back to the *bundle* tag on reload — silently promoting a
    # sketch to EXACT (a wrong `count_distinct`) or demoting an exact count to a rescan. So
    # they must round-trip alongside the bundle provenance the values were stored with.
    # `moments_provenance` (the tag on `total_sum`/`mean`) is here for the same reason and
    # in the same direction as `null_count_provenance`: a sum an in-memory source computed
    # over its own values is exact while the bundle around it holds no bounds at all, so
    # dropping the tag on reload demotes the sum to a rescan of a relation whose total is
    # already on file.
    for field in ("ndv_provenance", "null_count_provenance", "moments_provenance"):
        sub = getattr(col, field)
        if sub is not None:
            out[field] = sub.name
    return out


def _encode_quantiles(grid: Any) -> dict[str, list[float]] | None:
    """The ascending quantile grid as JSON, or None when it is absent or unusable.

    Requires the two arrays to be present, numeric, the same length, and non-empty — the
    shape `plan.stats.ColumnStat` documents. A grid failing any of those is dropped rather
    than repaired: a half-decoded CDF produces a confident wrong selectivity, where a missing
    one produces the documented fallback.
    """
    if not isinstance(grid, Mapping):
        return None
    probs, values = grid.get("probs"), grid.get("values")
    if not isinstance(probs, (list, tuple)) or not isinstance(values, (list, tuple)):
        return None
    if not probs or len(probs) != len(values) or len(probs) > _MAX_QUANTILE_POINTS:
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (*probs, *values)):
        return None
    return {"probs": [float(p) for p in probs], "values": [float(v) for v in values]}


def _encode_mcv(mcv: Any) -> dict[str, float] | None:
    """Most-common values as JSON, keeping the most frequent when there are too many.

    Truncation is safe here in a way it is not for a quantile grid: the entries dropped are
    the least frequent of the table, and every retained frequency is still that value's own.
    """
    if not isinstance(mcv, Mapping) or not mcv:
        return None
    usable = {
        str(k): float(v)
        for k, v in mcv.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not usable:
        return None
    if len(usable) > _MAX_MCV_ENTRIES:
        top = sorted(usable.items(), key=lambda kv: -kv[1])[:_MAX_MCV_ENTRIES]
        usable = dict(top)
    return usable


def _decode(blob: dict[str, Any]) -> SourceStatistics | None:
    try:
        columns = {name: _decode_column(c) for name, c in blob.get("columns", {}).items()}
        return SourceStatistics(
            row_count=blob.get("row_count"),
            byte_size=blob.get("byte_size"),
            columns=columns,
            sorted_by=tuple(_decode_sort_key(k) for k in blob.get("sorted_by", ())),
            partition_keys=tuple(blob.get("partition_keys", ())),
            exact_rows=bool(blob.get("exact_rows", True)),
            content_byte_size=bool(blob.get("content_byte_size", False)),
            bounds_include_nan=bool(blob.get("bounds_include_nan", False)),
            row_group_count=blob.get("row_group_count"),
        )
    except Exception as exc:
        note_suppressed("metadata", "decode source stats", exc)
        return None


def _decode_column(blob: dict[str, Any]) -> ColumnStat:
    prov = Provenance[blob.get("provenance", "DEFAULT")]
    bloom_b64 = blob.get("bloom")
    bloom = None
    if isinstance(bloom_b64, str):
        import base64

        bloom = base64.b64decode(bloom_b64)
    ndv_prov = blob.get("ndv_provenance")
    null_prov = blob.get("null_count_provenance")
    moments_prov = blob.get("moments_provenance")
    return ColumnStat(
        min=blob.get("min"),
        max=blob.get("max"),
        null_count=blob.get("null_count"),
        ndv=blob.get("ndv"),
        total_sum=blob.get("total_sum"),
        mean=blob.get("mean"),
        avg_bytes=blob.get("avg_bytes"),
        provenance=prov,
        bloom=bloom,
        # Re-validated on the way in, not trusted because this module wrote it: the hub is a
        # file on disk that another process, an older build, or a partial write can reach.
        quantiles=_encode_quantiles(blob.get("quantiles")),
        mcv=_encode_mcv(blob.get("mcv")),
        ndv_provenance=Provenance[ndv_prov] if isinstance(ndv_prov, str) else None,
        null_count_provenance=Provenance[null_prov] if isinstance(null_prov, str) else None,
        moments_provenance=Provenance[moments_prov] if isinstance(moments_prov, str) else None,
    )
