"""Learned join-skew: persist the hot join-key values measured by the detection
pre-pass, keyed by join shape, so a later run of the same shape engages salting from
learned skew without re-running the pre-pass (and without the user re-opting in).

The loop is result-preserving — salting only moves a hot key's work between reducers,
never the joined relation — so a learned hot set is always safe to act on. Stored as
neutral learned params in the process-wide `MetadataHub`; `dist` reads/writes them
directly (it is outside the kyber/carbonite/core independence set).
"""

from __future__ import annotations

import hashlib
import json
import math

from batcher._internal.logging import note_suppressed
from batcher.plan.logical import Join

__all__ = [
    "DEFAULT_LEARNED_SALT",
    "join_skew_key",
    "load_learned_hot_keys",
    "persist_hot_keys",
    "salt_factor",
]

# Learned-skew namespace + the salt fan-out used when learned hot keys engage salting
# on a run that did not explicitly request it.
_SKEW_NAMESPACE = "dist.skew"
DEFAULT_LEARNED_SALT = 4
# Upper bound on the salt fan-out. Each sub-partition of a hot key costs a replicated copy
# of the matching build-side rows, so an unbounded fan-out trades one overloaded reducer for
# a network-wide broadcast. 64 covers a key holding a third of the data across a 200-way
# shuffle, which is already an extreme skew.
_MAX_SALT = 64


def salt_factor(hot_fraction: float, partitions: int) -> int:
    """How many sub-partitions one hot key should be fanned across.

    A fixed fan-out is the wrong shape for this problem, because the imbalance it has to
    repair depends on the shuffle's width. With `P` reducers the average reducer receives
    `N/P` rows, while the reducer owning a value of frequency `f` receives `f·N` — an
    overload factor of `f·P`, which grows with the cluster. Splitting that key across `s`
    sub-partitions divides its load by `s`, so levelling it needs

    ``s >= f · P``

    and nothing more: past that point the hot key's reducers are already below average and
    every extra sub-partition only replicates more of the build side. A constant 4 is
    therefore simultaneously too much on a 16-way shuffle (where a 10% key overloads by 1.6x)
    and far too little on a 200-way one (where the same key overloads by 20x and stays 5x
    over average after salting) — and it is the wide shuffle, the one that only appears at
    scale, that the constant serves worst.

    Args:
        hot_fraction: The hot value's share of the side's rows.
        partitions: The number of reduce partitions the shuffle fans into.

    Returns:
        The salt fan-out, at least 2 (salting at all means splitting in two) and capped.
    """
    if hot_fraction <= 0.0 or partitions <= 1:
        return 2
    needed = math.ceil(hot_fraction * partitions)
    return max(2, min(_MAX_SALT, needed))


def join_skew_key(left_ir: str, right_ir: str, join: Join) -> str:
    """A stable key identifying this join's shape (both sides + keys + type), so the
    hot values learned on one run are reused on the next run of the same shape."""
    payload = json.dumps(
        [left_ir, right_ir, list(join.left_keys), list(join.right_keys), join.join_type],
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def load_learned_hot_keys(shape_key: str) -> list[str] | None:
    """The hot join-key values learned for this shape, or `None` if never measured.

    A learned empty list means "measured, not skewed" — distinct from never-measured,
    so a non-skewed shape never re-runs the detection pre-pass. Best-effort; the hub
    being unavailable simply means no learned skew (fall back to the config behavior).
    """
    try:
        from batcher.core import default_hub

        val = default_hub().get_keyed_param(_SKEW_NAMESPACE, shape_key)
        return list(val) if val is not None else None
    except Exception:
        return None


def persist_hot_keys(shape_key: str, hot: list[str]) -> None:
    """Record the hot values measured by the detection pre-pass, so a later run of the
    same join shape engages salting from learned skew without re-running the pre-pass.
    Best-effort; never breaks the join."""
    try:
        from batcher.core import default_hub

        default_hub().put_keyed_param(_SKEW_NAMESPACE, shape_key, hot)
    except Exception as exc:
        note_suppressed("dist", "persist learned hot keys", exc)
