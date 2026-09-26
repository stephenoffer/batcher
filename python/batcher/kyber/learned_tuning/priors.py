"""Per-signature learned scalars — the priors that seed sizing and pre-aggregation.

Where `bandit` picks between algorithms and `crossover` learns a threshold, this module learns a
*number* per plan signature from what actually happened: how many rows a breaker shuffled and
how far an aggregate collapsed its input. Each is folded into O(1) sufficient statistics by a
`record_*` function and read back by a `learned_*` one.

Whether to re-optimize *between* stages is not here: it is a two-sided cost question, so it
lives with the other regret-minimizing choices in `bandit.learned_adaptive_route`.

Every one of them steers sizing, sharding or planning effort only — a partition count, whether to
pre-aggregate — so a wrong learned value costs throughput and never correctness. The family
contract is in the package docstring.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.kyber import plan_cache
from batcher.metadata.smoothed import convergent_blend

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "learned_partial_agg",
    "learned_partition_count",
    "learned_signature_rows",
    "record_group_reduction",
    "record_partition_rows",
]

_NS_PART = "tuning.partition_rows"  # per-signature measured shuffle rows
_NS_GROUP = "tuning.group_reduction"  # per-signature measured groups / input rows

# **Deliberately not hardware-scoped**, where the rest of `learned_tuning` is.
#
# `metadata.hardware_scope` draws the line at what the stored value *describes*: scope a
# machine-unit measurement (nanoseconds, bytes of RAM, a batch size chosen against them), and
# never scope a statement about data, because "scoping those would fragment the statistics that
# took the most work to collect, turning a well-calibrated fleet into N poorly-calibrated ones
# for no gain." Both values here are on the data side of that line and nothing else: a breaker's
# shuffled row count and an aggregate's `groups / input_rows` ratio. A relation has the same
# number of rows whichever machine counts them.
#
# Both are recorded on the *driver* from the whole query's figures (`api.tuning.decisions`),
# never per shard. Scoped, the same query planned single-node and then distributed would
# write two entries for identical data, neither run informing the other, and an autoscaling
# fleet would split them again on every instance type it moved through.


# Decision family — per-signature priors (partitions, pre-aggregation).
def _record_scalar(
    hub: MetadataHub | None, namespace: str, key: str, field: str, value: float
) -> None:
    # Non-finite observations are dropped: smoothing folds a NaN or an infinity into the stored
    # prior and from there into every later update, poisoning the entry for the life of the
    # store (`metadata.smoothed.record_smoothed_scalar` spells out the same argument).
    if hub is None or not math.isfinite(value) or value < 0.0:
        return
    try:
        entry = dict(hub.get_keyed_param(namespace, key) or {})
        n = int(entry.get("n_obs", 0))
        prior = entry.get(field)
        entry[field] = (
            float(value) if prior is None else convergent_blend(float(prior), float(value), n)
        )
        entry["n_obs"] = n + 1
        plan_cache.record_write(hub, namespace, key, entry)
    except Exception as exc:  # pragma: no cover - best-effort learned prior
        note_suppressed("kyber", "record scalar prior", exc)


def record_partition_rows(hub: MetadataHub | None, signature: str, rows: float) -> None:
    """Record a breaker's measured shuffle row count, keyed by signature."""
    _record_scalar(hub, _NS_PART, signature, "rows", rows)


def learned_partition_count(
    hub: MetadataHub | None, signature: str, target_rows: int
) -> int | None:
    """A partition prior from measured shuffle rows (`ceil(rows / target_rows)`), or `None`.

    Fan-out from the *measured* volume this breaker actually shuffled, not a cold estimate, so a
    recurring stage shards to fit memory on the first re-run. A partition count only shards data,
    so any value produces the identical result.
    """
    if hub is None or target_rows <= 0:
        return None
    try:
        entry = hub.get_keyed_param(_NS_PART, signature) or {}
        rows = entry.get("rows")
        if rows is None or float(rows) <= 0.0:
            return None
        return max(1, math.ceil(float(rows) / target_rows))
    except Exception as exc:  # pragma: no cover - best-effort learned prior
        note_suppressed("kyber", "read partition rows", exc)
        return None


def record_group_reduction(
    hub: MetadataHub | None, signature: str, groups: float, input_rows: float
) -> None:
    """Record an aggregate's measured cardinality reduction (`groups / input_rows`)."""
    if input_rows <= 0.0:
        return
    _record_scalar(hub, _NS_GROUP, signature, "ratio", max(0.0, min(1.0, groups / input_rows)))


def learned_partial_agg(
    hub: MetadataHub | None, signature: str, *, engage_below: float = 0.5
) -> bool | None:
    """Whether to engage partial pre-aggregation, from the group-reduction ratio, or `None`.

    Partial pre-aggregation pays off exactly when a group-by collapses many rows into few groups
    (a low measured `groups/input` ratio); when almost every row is its own group it is wasted
    work. Learning the ratio per signature beats DuckDB's static "always pre-aggregate" guess.
    Engaging or skipping the pre-agg is an algebraic identity — the final aggregate is unchanged.
    """
    if hub is None:
        return None
    try:
        entry = hub.get_keyed_param(_NS_GROUP, signature) or {}
        ratio = entry.get("ratio")
        return None if ratio is None else float(ratio) <= engage_below
    except Exception as exc:  # pragma: no cover - best-effort learned prior
        note_suppressed("kyber", "read group reduction", exc)
        return None


# Decision family — learned selectivity-primed estimate.
def learned_signature_rows(hub: MetadataHub | None, signature: str) -> float | None:
    """The measured output rows recorded for a (sub)plan signature, or `None` if never seen.

    Reads the same `kyber.stats` feedback `learning.record_execution` writes, so a recurring
    subplan's estimate starts from its measured size rather than a default — priming selectivity
    and join-order costing for the intermediate, not just the whole query. Estimate-only: it steers
    cost, never the result.
    """
    if hub is None:
        return None
    try:
        from batcher.kyber.learning import load_learned_stats

        rows = load_learned_stats(hub).get(signature, {}).get("rows")
        return float(rows) if rows is not None else None
    except Exception as exc:  # pragma: no cover - best-effort learned prior
        note_suppressed("kyber", "read signature rows", exc)
        return None
