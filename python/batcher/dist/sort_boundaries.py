"""Learned range-sort boundaries: persist the quantile grid the SAMPLE barrier measured,
keyed by sort shape, so a later run of the same shape skips the barrier entirely.

A distributed full sort runs its map plan **twice**. Once in `sample_quantiles`, which
executes the whole mapped prefix — scan, pushed predicate, projection — over every split
purely to return ~33 floats per worker; and once again in `range_publish`, which executes
the identical plan to bucketize the rows it just measured and threw away. On a
scan-dominated sort that duplicated prefix is close to half the job, and it buys a quantity
that barely moves between runs of the same query over the same data.

So measure it once. The merged per-worker grids are persisted under the sort's shape, and a
later run of that shape range-partitions straight from them: one pass over the input rather
than two.

**This is safe to act on even when the learned grid is stale, and that is a property of the
algorithm rather than a hope.** Boundaries decide only which reducer a row goes to. The
buckets are globally ordered for *any* monotone boundary list, because `bucketize` places
rows by `searchsorted(side="right")` against deduplicated boundaries and the reducers'
outputs are concatenated in bucket order. A grid that no longer describes the data
therefore costs *balance* — some reducer gets more rows than its share — and can never cost
a row, a duplicate, or an ordering. That is the same failure mode sampling error already
has, which is why the sample pass is allowed to sample rather than sort.

Stored as neutral learned params in the process-wide `MetadataHub`; `dist` reads and writes
them directly, as it does for learned join skew (see `dist/skew.py`, whose loop this
mirrors: measure a pre-pass once, key it by shape, never pay for it again).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from batcher._internal.logging import note_suppressed

__all__ = [
    "load_learned_grids",
    "persist_grids",
    "sort_key_identity",
    "sort_key_is_string",
    "sort_shape_key",
]

_SORT_NAMESPACE = "dist.sort_grid"

# Cap on the grids retained for one shape. A grid is one worker's sampled CDF, so the
# stored payload grows with the fleet that measured it; past this many the mixture is
# already pinned and the rest is storage. Kept well above a typical fleet so the common
# case stores everything it measured.
_MAX_GRIDS = 256


def sort_shape_key(map_ir: str, key_name: str, key_identity: str | None = None) -> str:
    """A stable key identifying this sort's shape, so a grid measured on one run is reused
    on the next run of the same shape.

    The mapped plan is part of the key and has to be: the grid describes the rows the sort
    actually partitions, which is the input *after* the pushed-down predicate and
    projection. Two sorts over the same table on the same column but behind different
    filters have genuinely different key distributions, and keying on the column alone
    would hand one of them the other's boundaries.

    **The plan alone does not identify the relation, and `map_ir` is where that breaks.** A
    mapped prefix that is a bare scan serializes to `{"op": "scan", "source_id": 0}` — a
    *positional* index into this plan's own source list, with no schema and no source
    identity. Every single-source sort in the process therefore produced the same digest,
    and shared one grid. Two ways that hurt, both measured:

    * **Wrong type — it raises.** `sort("k")` over a `float64` column and over a `string`
      column hash identically, so the second loads the first's grid and a float boundary
      list reaches the string range partitioner:
      ``TypeError: argument 'boundaries': 'float' object cannot be converted to 'PyString'``,
      from inside a Ray task, after two retries. The reverse fails in NumPy.
    * **Wrong relation — it silently serializes the reduce.** Two tables with the same
      schema and the same key column but disjoint ranges share a grid, so every key of the
      second falls past the last boundary of the first. Measured over 4,000 rows into 8
      buckets: `[547, 479, 481, 485, 502, 473, 487, 546]` with its own grid against
      `[0, 0, 0, 0, 0, 0, 0, 4000]` with the other table's — **seven of eight reducers
      idle and the whole relation on one**. Correct, and no longer distributed.

    The second is the one to keep in mind when reading this module's safety argument ("a
    grid that no longer describes the data costs *balance* and can never cost a row"). That
    is true, and "balance" can mean the entire fan-out.

    `kyber.signature` already fixed this exact defect for learned statistics by putting
    `Scan.source_key` in its scan token instead of the positional id; `key_identity` is the
    same correction for this store, and is built by [`sort_key_identity`] from the same
    `plan.source_stats` helper so the two cannot drift apart.

    Args:
        map_ir: The serialized mapped plan prefix each worker executes over its split.
        key_name: The leading sort key's column name.
        key_identity: The relation-and-type token from [`sort_key_identity`]. `None` keeps
            the pre-identity digest, so a caller that cannot see the source still works —
            and is still protected by the load-side check in [`load_learned_grids`].

    Returns:
        A short hex digest identifying the shape.
    """
    payload = json.dumps([map_ir, key_name, key_identity], sort_keys=True)
    # `usedforsecurity=False` because this is a cache key, not a security claim — on a
    # FIPS-enforcing host a bare `sha1()` raises, which would turn an unavailable
    # optimization into a failed sort.
    return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:16]


def sort_key_identity(source: object, key_name: str) -> str | None:
    """Which relation this grid describes and what type its key is, or `None` if unknown.

    Two facts that `map_ir` cannot carry, in the one token `sort_shape_key` hashes:

    * the source's own key from `plan.source_stats.source_stats_key` — the same helper
      `kyber.signature` uses, so a real table keeps its identity across runs (which is what
      makes the grid worth persisting) while an in-memory relation gets a per-instance
      serial rather than a shape-based key that two unrelated relations would share;
    * the leading key column's Arrow type, because a grid of the wrong type does not merely
      describe the wrong distribution, it cannot be passed to the range partitioner at all.

    `None` when the source can name neither, which leaves the digest exactly as it was and
    leans on the load-side check instead. Best-effort throughout: this is a cache key, and
    failing to build one must cost a sample pass rather than the query.

    Args:
        source: The bound input the sort reads.
        key_name: The leading sort key's column name.

    Returns:
        A token identifying the relation and the key's type, or `None`.
    """
    try:
        from batcher.plan.source_stats import source_stats_key

        key = source_stats_key(source) or ""
        field = source.schema().field(key_name)  # type: ignore[attr-defined]
    except Exception as exc:
        note_suppressed("dist", "identify the sort key's relation", exc)
        return None
    return f"{key}|{field.type}"


def sort_key_is_string(source: object, key_name: str) -> bool | None:
    """Whether `source`'s `key_name` column is a string key — `None` when it cannot be seen.

    The one question the grid's element type has to answer, asked of the schema rather than
    of the data, because `merge_boundaries` dispatches on exactly this distinction: a string
    grid is merged lexically and everything else numerically. Best-effort by construction —
    a source that cannot produce a schema yields `None`, which every caller reads as "no
    type opinion" and which leaves behavior exactly as it was.

    Args:
        source: The bound input the sort reads.
        key_name: The leading sort key's column name.

    Returns:
        True for a string/large-string key, False for any other type, `None` if unknown.
    """
    try:
        import pyarrow as pa

        field = source.schema().field(key_name)  # type: ignore[attr-defined]
    except Exception as exc:
        note_suppressed("dist", "read the sort key's type", exc)
        return None
    return bool(pa.types.is_string(field.type) or pa.types.is_large_string(field.type))


def load_learned_grids(
    shape_key: str, expect_strings: bool | None = None
) -> list[tuple[list[Any], int]] | None:
    """The per-worker quantile grids learned for this sort shape, or `None` if never
    measured.

    Returned in exactly the `(grid, row_count)` shape `merge_boundaries` consumes, so the
    caller re-cuts them for whatever bucket count *this* run resolved — the learned fan-out
    moves between runs, and boundaries sized for the wrong count would route rows past the
    last bucket.

    `expect_strings` is the load-side half of the type guard `sort_shape_key` describes. It
    is deliberately a *second* check rather than a restatement of the first: keying by type
    stops the two shapes from sharing an entry from now on, and this stops an entry written
    before that — by an older build, under the colliding digest — from reaching the range
    partitioner and raising inside a Ray task. A grid whose elements disagree with the key
    is discarded, which costs one ordinary sample pass and is what this module already
    promises for every other failure. `None` asks no question, which is what a caller that
    cannot see the schema gets.

    Best-effort: an unreachable hub means no learned grid and an ordinary sample pass.

    Args:
        shape_key: The digest from [`sort_shape_key`].
        expect_strings: Whether the leading key is a string, from [`sort_key_is_string`].

    Returns:
        The stored grids, or `None` when this shape has never been sampled or what was
        stored does not describe this key.
    """
    try:
        from batcher.core import default_hub

        stored = default_hub().get_keyed_param(_SORT_NAMESPACE, shape_key)
        if not stored:
            return None
        grids = [(list(grid), int(n)) for grid, n in stored if grid and n]
        if expect_strings is not None and any(
            isinstance(grid[0], str) is not expect_strings for grid, _n in grids
        ):
            note_suppressed(
                "dist",
                "reuse a learned sort grid",
                TypeError(
                    f"stored grid is {'not ' if expect_strings else ''}a string grid but the "
                    f"sort key is {'' if expect_strings else 'not '}a string; re-sampling"
                ),
            )
            return None
        return grids or None
    except Exception as exc:
        # Noted, not silent: a hub that cannot be read would otherwise re-run the sample
        # pass forever with nothing in the log to say why.
        note_suppressed("dist", "load learned sort boundaries", exc)
        return None


def persist_grids(shape_key: str, grids: list[tuple[list[Any], int]]) -> None:
    """Record the grids the SAMPLE barrier measured, so a later run of this sort shape
    range-partitions without re-reading its input.

    Grids describing no rows are dropped rather than stored: an empty split contributes
    nothing to the mixture, and storing it would only make a later run's payload larger.
    Best-effort; never breaks the sort.

    Args:
        shape_key: The digest from [`sort_shape_key`].
        grids: The `(sampled_cdf, row_count)` pairs the workers returned.
    """
    try:
        from batcher.core import default_hub

        kept = [[list(grid), int(n)] for grid, n in grids if grid and n][:_MAX_GRIDS]
        if not kept:
            return
        default_hub().put_keyed_param(_SORT_NAMESPACE, shape_key, kept)
    except Exception as exc:
        note_suppressed("dist", "persist learned sort boundaries", exc)
