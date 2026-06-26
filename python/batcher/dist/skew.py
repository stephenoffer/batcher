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

from batcher.plan.logical import Join

__all__ = [
    "DEFAULT_LEARNED_SALT",
    "join_skew_key",
    "load_learned_hot_keys",
    "persist_hot_keys",
]

# Learned-skew namespace + the salt fan-out used when learned hot keys engage salting
# on a run that did not explicitly request it.
_SKEW_NAMESPACE = "dist.skew"
DEFAULT_LEARNED_SALT = 4


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
    except Exception:
        pass
