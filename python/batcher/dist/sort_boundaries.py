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
    "sort_shape_key",
]

_SORT_NAMESPACE = "dist.sort_grid"

# Cap on the grids retained for one shape. A grid is one worker's sampled CDF, so the
# stored payload grows with the fleet that measured it; past this many the mixture is
# already pinned and the rest is storage. Kept well above a typical fleet so the common
# case stores everything it measured.
_MAX_GRIDS = 256


def sort_shape_key(map_ir: str, key_name: str) -> str:
    """A stable key identifying this sort's shape, so a grid measured on one run is reused
    on the next run of the same shape.

    The mapped plan is part of the key and has to be: the grid describes the rows the sort
    actually partitions, which is the input *after* the pushed-down predicate and
    projection. Two sorts over the same table on the same column but behind different
    filters have genuinely different key distributions, and keying on the column alone
    would hand one of them the other's boundaries.

    Args:
        map_ir: The serialized mapped plan prefix each worker executes over its split.
        key_name: The leading sort key's column name.

    Returns:
        A short hex digest identifying the shape.
    """
    payload = json.dumps([map_ir, key_name], sort_keys=True)
    # `usedforsecurity=False` because this is a cache key, not a security claim — on a
    # FIPS-enforcing host a bare `sha1()` raises, which would turn an unavailable
    # optimization into a failed sort.
    return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:16]


def load_learned_grids(shape_key: str) -> list[tuple[list[Any], int]] | None:
    """The per-worker quantile grids learned for this sort shape, or `None` if never
    measured.

    Returned in exactly the `(grid, row_count)` shape `merge_boundaries` consumes, so the
    caller re-cuts them for whatever bucket count *this* run resolved — the learned fan-out
    moves between runs, and boundaries sized for the wrong count would route rows past the
    last bucket.

    Best-effort: an unreachable hub means no learned grid and an ordinary sample pass.

    Args:
        shape_key: The digest from [`sort_shape_key`].

    Returns:
        The stored grids, or `None` when this shape has never been sampled.
    """
    try:
        from batcher.core import default_hub

        stored = default_hub().get_keyed_param(_SORT_NAMESPACE, shape_key)
        if not stored:
            return None
        grids = [(list(grid), int(n)) for grid, n in stored if grid and n]
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
