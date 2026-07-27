"""Per-signature learned scalars — the priors that seed sizing and pre-aggregation.

Where `bandit` picks between algorithms and `crossover` learns a threshold, this module learns a
*number* per plan signature from what actually happened: how big each join side was, how many rows
a breaker shuffled, how far an aggregate collapsed its input. Each is folded into O(1)
sufficient statistics by a `record_*` function and read back by a `learned_*` one.

Whether to re-optimize *between* stages is not here: it is a two-sided cost question, so it
lives with the other regret-minimizing choices in `bandit.learned_adaptive_route`.

Every one of them steers sizing, sharding or planning effort only — a build orientation, a
partition count, whether to pre-aggregate — so a wrong learned value costs
throughput and never correctness. The family contract is in the package docstring.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config
from batcher.kyber import plan_cache

if TYPE_CHECKING:
    from batcher.metadata import MetadataHub

__all__ = [
    "learned_build_sides",
    "learned_partial_agg",
    "learned_partition_count",
    "learned_signature_rows",
    "record_group_reduction",
    "record_join_sides",
    "record_partition_rows",
]

_NS_SIDES = "tuning.join_sides"  # per-signature measured (left_rows, right_rows)
_NS_PART = "tuning.partition_rows"  # per-signature measured shuffle rows
_NS_GROUP = "tuning.group_reduction"  # per-signature measured groups / input rows


# Decision family — per-signature priors (build sides, partitions, pre-aggregation).
def _smooth(prior: float, observed: float, n_obs: int) -> float:
    """A running mean while evidence is thin, decaying into an EWMA.

    Step `max(OptimizerConfig.learned_scalar_alpha_floor, 1/(n_obs+1))`. The floor is *not*
    the static blend weight used elsewhere: at that 0.5 the newest run always carried half the
    weight, giving these priors a ~2-observation memory one anomalous run swung by half.
    """
    floor = active_config().optimizer.learned_scalar_alpha_floor
    alpha = max(floor, 1.0 / (n_obs + 1))
    return alpha * observed + (1.0 - alpha) * prior


def _record_scalar(
    hub: MetadataHub | None, namespace: str, key: str, field: str, value: float
) -> None:
    if hub is None or value < 0.0:
        return
    try:
        entry = dict(hub.get_keyed_param(namespace, key) or {})
        n = int(entry.get("n_obs", 0))
        prior = entry.get(field)
        entry[field] = float(value) if prior is None else _smooth(float(prior), float(value), n)
        entry["n_obs"] = n + 1
        plan_cache.record_write(hub, namespace, key, entry)
    except Exception as exc:  # pragma: no cover - best-effort learned prior
        note_suppressed("kyber", "record scalar prior", exc)


def record_join_sides(
    hub: MetadataHub | None, signature: str, left_rows: float, right_rows: float
) -> None:
    """Record a join's measured left/right input sizes, keyed by signature."""
    if hub is None:
        return
    try:
        entry = dict(hub.get_keyed_param(_NS_SIDES, signature) or {})
        n = int(entry.get("n_obs", 0))
        for field, value in (("left", left_rows), ("right", right_rows)):
            prior = entry.get(field)
            entry[field] = float(value) if prior is None else _smooth(float(prior), float(value), n)
        entry["n_obs"] = n + 1
        plan_cache.record_write(hub, _NS_SIDES, signature, entry)
    except Exception as exc:  # pragma: no cover - best-effort learned prior
        note_suppressed("kyber", "record join sides", exc)


def learned_build_sides(hub: MetadataHub | None, signature: str) -> tuple[float, float] | None:
    """The measured `(left_rows, right_rows)` for this join, or `None` cold.

    Seeds build-side selection from what the two sides *actually* were last time, so a join whose
    estimate is wrong (correlated predicates, skew) still builds the truly-smaller side. Only the
    build orientation changes — the relation does not.
    """
    if hub is None:
        return None
    try:
        entry = hub.get_keyed_param(_NS_SIDES, signature) or {}
        left, right = entry.get("left"), entry.get("right")
        if left is None or right is None:
            return None
        return float(left), float(right)
    except Exception as exc:  # pragma: no cover - best-effort learned prior
        note_suppressed("kyber", "read join sides", exc)
        return None


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
