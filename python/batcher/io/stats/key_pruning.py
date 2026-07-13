"""Key-driven file pruning — the copy-on-write MERGE's "which files must I rewrite?".

A copy-on-write ``MERGE`` rewrites the target's data files. The naive version rewrites
*all* of them, which makes an upsert cost the whole table: merging a 1,000-row change
set into a 100M-row table reads and rewrites 100M rows. Every warehouse (Delta,
Iceberg, Snowflake) avoids that the same way — a data file only has to be rewritten if
it can actually **contain one of the source's keys**, and a file's per-column min/max
(the Parquet footer, or a manifest's add-actions) is enough to prove that it cannot.

This module is that proof. It reduces the source's join keys to a compact `KeyDigest`
(bounds + the sorted distinct values), then tests each target file's key zone-map
against it. Files that survive are read and rewritten; the rest are never opened, and
are carried into the new commit by reference.

## Soundness

The only unsafe answer is a false *prune* — dropping a file that did contain a matching
key would silently lose an update. So, exactly as in `file_skipping`, **a file is
dropped only when its statistics prove it cannot hold any source key; anything unknown
keeps it.** A missing footer, an absent statistic, a type that will not compare, a key
column the writer recorded no bounds for — all resolve to *keep*. `surviving_files`
returns `None` ("prune nothing, rewrite everything") rather than raising.

Two necessary conditions are tested per file, and a file must pass both to be rewritten:

1. **Bounding box** — for every key column, the file's ``[min, max]`` must intersect the
   source's ``[min, max]``. Cheap, and the whole win on a clustered/time-ordered key.
2. **Exact occupancy** — for a key column whose distinct source values were collected,
   at least one of them must lie inside the file's ``[min, max]``. Strictly sharper than
   the bounding box, and it is what saves the sparse-source case (keys ``{1, 9_999_999}``
   span the whole table under a bounding box alone, yet touch only two files).

A writer-*truncated* string bound only ever **widens** a file's interval, so both tests
stay sound on it: they prune only when even the widened interval admits no source key.

## Cost

Everything is vectorized over the **file** dimension — pyarrow compute for the interval
tests, one numpy `searchsorted` per key column for occupancy. No Python loop over files,
and no row is ever touched in the control plane. Pruning a 100,000-file table is a
handful of vector ops, not 100,000 comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "MAX_EXACT_KEYS",
    "KeyDigest",
    "key_digest",
    "surviving_files",
]

# How many distinct source keys we are willing to hold on the driver for the exact
# occupancy test. This is *metadata* — one key column's distinct values, not rows — and
# it stays a control-plane object. Beyond the cap the digest keeps only bounds: a source
# with millions of distinct keys is dense enough to touch most files anyway, so the
# sharper test would buy little and cost real memory.
MAX_EXACT_KEYS = 1_000_000


@dataclass(frozen=True, slots=True)
class KeyDigest:
    """The compact summary of a MERGE source's keys that each target file is tested against.

    `bounds` holds each key column's ``(min, max)``; `values` holds the *sorted distinct*
    values of the columns cheap enough to keep whole (see `MAX_EXACT_KEYS`). A column
    absent from either mapping simply contributes no pruning — the "unknown keeps the
    file" rule, encoded as a missing entry rather than as a special case.

    `null_keys` records key columns in which the source has a NULL. SQL's ``=`` never
    matches NULL, so such a source row matches no target row — but an engine whose hash
    join pairs null with null would match it, so any file recording a NULL in that column
    is kept. Conservative, and it costs at most one extra file per key column.
    """

    columns: tuple[str, ...]
    bounds: dict[str, tuple[Any, Any]]
    values: dict[str, Any]  # column -> sorted distinct values, as a numpy array
    null_keys: frozenset[str]
    empty: bool = False

    @property
    def prunes_nothing(self) -> bool:
        """True when the digest carries no usable statistic, so no file can be dropped."""
        return not self.bounds and not self.values


def key_digest(keys: pa.Table, max_exact: int = MAX_EXACT_KEYS) -> KeyDigest:
    """Reduce a source's distinct key columns to the digest a target file is pruned against.

    `keys` is a small control-plane table — the source's key columns, deduplicated — never
    the source itself. An empty `keys` yields an `empty` digest, which prunes *every*
    target file: a change set with no keys can match no target row.

    Args:
        keys: The source's distinct join-key columns.
        max_exact: Cap on distinct values kept per column for the exact occupancy test.

    Returns:
        The digest describing which target files could hold one of these keys.
    """
    import pyarrow.compute as pc

    columns = tuple(keys.column_names)
    if keys.num_rows == 0:
        return KeyDigest(columns, {}, {}, frozenset(), empty=True)

    bounds: dict[str, tuple[Any, Any]] = {}
    values: dict[str, Any] = {}
    null_keys: set[str] = set()

    for name in columns:
        column = keys.column(name)
        if column.null_count:
            null_keys.add(name)
        try:
            minmax = pc.min_max(column)  # null-skipping
            low, high = minmax["min"].as_py(), minmax["max"].as_py()
        except Exception:
            continue  # a type with no orderable min/max prunes nothing on this column
        if low is None or high is None:
            continue  # an all-null key column carries no bound to prune with
        bounds[name] = (low, high)
        if column.length() <= max_exact:
            sorted_values = _sorted_numpy(column)
            if sorted_values is not None:
                values[name] = sorted_values

    return KeyDigest(columns, bounds, values, frozenset(null_keys))


def _sorted_numpy(column: Any) -> Any | None:
    """`column`'s non-null values as a sorted numpy array, or None if not bisectable."""
    try:
        valid = column.drop_null().combine_chunks()
        if len(valid) == 0:
            return None
        array = valid.to_numpy(zero_copy_only=False)
        array.sort()
        return array
    except Exception:
        return None  # not representable as a sortable numpy array → bounds-only


def surviving_files(digest: KeyDigest, manifest: Any) -> list[str] | None:
    """Paths in `manifest` of the target files that could contain one of the source's keys.

    `manifest` is per-file statistics in the add-action layout `file_skipping` defines
    (``path | num_records | min.<col> | max.<col> | null_count.<col>``), so a Parquet
    footer scrape and a lakehouse transaction log prune through this one code path.

    Returns `None` when nothing could be decided — an unusable manifest, or no key column
    with a recorded bound — which the caller MUST read as **"rewrite every file"**. Never
    raises: a manifest it cannot interpret prunes nothing rather than failing the merge.

    Args:
        digest: The source keys' digest, from `key_digest`.
        manifest: Per-file statistics in the add-action layout.

    Returns:
        The surviving file paths, or None if no pruning could be proven.
    """
    try:
        return _surviving(digest, manifest)
    except Exception:
        return None  # an uninterpretable manifest prunes nothing, it does not fail


def _surviving(digest: KeyDigest, manifest: Any) -> list[str] | None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    from batcher.io.stats.file_skipping import MAX_PREFIX, MIN_PREFIX, NULL_PREFIX

    if manifest is None or manifest.num_rows == 0 or "path" not in manifest.column_names:
        return None
    if digest.empty:
        return []  # no source keys ⇒ no target file can hold one
    if digest.prunes_nothing:
        return None

    names = manifest.column_names
    keep: Any = None

    for name in digest.columns:
        low_name, high_name = f"{MIN_PREFIX}{name}", f"{MAX_PREFIX}{name}"
        if low_name not in names or high_name not in names or name not in digest.bounds:
            continue  # no recorded bound for this key column → it prunes nothing
        file_low = manifest.column(low_name).combine_chunks()
        file_high = manifest.column(high_name).combine_chunks()
        # A null bound is an *absent statistic*, never a proof of non-match.
        known = pc.and_(pc.is_valid(file_low), pc.is_valid(file_high))

        low, high = digest.bounds[name]
        try:
            # Intervals intersect: file.min <= source.max AND file.max >= source.min.
            matches = pc.and_(
                pc.less_equal(file_low, pa.scalar(high, file_low.type)),
                pc.greater_equal(file_high, pa.scalar(low, file_high.type)),
            )
        except Exception:
            continue  # a type that will not compare prunes nothing on this column

        exact = digest.values.get(name)
        if exact is not None and len(exact):
            occupied = _occupied(exact, file_low, file_high, np, pa)
            if occupied is not None:
                matches = pc.and_(matches, occupied)

        # A source NULL key keeps any file recording a NULL in that column, since a
        # null-matching hash join would pair them.
        if name in digest.null_keys:
            matches = pc.or_(matches, _has_nulls(manifest, f"{NULL_PREFIX}{name}", pc))

        # Unknown bounds always keep the file.
        survives = pc.or_(pc.fill_null(matches, True), pc.invert(known))
        keep = survives if keep is None else pc.and_(keep, survives)

    if keep is None:
        return None  # no key column was decidable → prune nothing
    return manifest.column("path").filter(pc.fill_null(keep, True)).to_pylist()


def _occupied(exact: Any, file_low: Any, file_high: Any, np: Any, pa: Any) -> Any | None:
    """Files holding at least one source key inside ``[min, max]`` — the sharp test.

    ``searchsorted`` finds, for each file, the first source key at or above the file's
    minimum; the file is occupied iff that key also falls at or below its maximum. One
    binary search per file, vectorized — the key set is never scanned.

    Returns None (no refinement, keep the bounding-box answer) for a key type numpy will
    not bisect, or where the bounds carry nulls we would have to guess at.
    """
    try:
        low = file_low.to_numpy(zero_copy_only=False)
        high = file_high.to_numpy(zero_copy_only=False)
        if low.dtype != exact.dtype or high.dtype != exact.dtype:
            return None  # mismatched representations → do not risk an unsound bisect
        position = np.searchsorted(exact, low, side="left")
        found = position < len(exact)
        candidate = exact[np.minimum(position, len(exact) - 1)]
        occupied = found & (candidate <= high)
        return pa.array(np.asarray(occupied, dtype=bool))
    except (TypeError, ValueError):
        return None


def _has_nulls(manifest: Any, column: str, pc: Any) -> Any:
    """Files recording at least one NULL in `column`. An absent count is unknown → True."""
    import pyarrow as pa

    if column not in manifest.column_names:
        return pa.scalar(True)
    return pc.fill_null(pc.greater(manifest.column(column), 0), True)
