"""Aggregate rule families: an aggregate that is a cheaper aggregate in disguise.

`agg_algebra` and `extra/agg_rules` already rewrite aggregates against the *shape of the
plan* — a `count` over a non-nullable column, an aggregate over a group key, a `sum` of a
constant. This package covers the orthogonal case: an aggregate whose *arguments* make it
equal to a different, far cheaper aggregate.

Both families here collapse a stateful reduction into a streaming one, which is the whole
point. A `quantile` builds a sorted structure over the group and an `approx_quantile`
builds a KLL sketch; `min` and `max` carry a single scalar. On a large group that is the
difference between a spill and a register.
"""

from __future__ import annotations

from batcher.kyber.rules.aggregate_algebra import (
    extremes as _extremes,  # noqa: F401  (registers)
)

__all__: list[str] = []
