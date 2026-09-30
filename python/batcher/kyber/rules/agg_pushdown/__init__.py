"""Aggregate-through-join pushdown: pre-aggregate a join side to shrink its input.

`rules` holds the registered rules, in registration order; `gates` the cost gates they ask;
`reassociate` pre-aggregates a star's facts beneath its dimensions, for the aligned planner.
"""

from __future__ import annotations

from batcher.kyber.rules.agg_pushdown.reassociate import pre_aggregate_facts
from batcher.kyber.rules.agg_pushdown.rules import (
    count_distinct_to_distinct_count,
    eager_aggregation,
    pre_aggregate_join_measures,
    pre_aggregation_through_join,
    pre_aggregation_through_reordered_join,
)

__all__ = [
    "count_distinct_to_distinct_count",
    "eager_aggregation",
    "pre_aggregate_facts",
    "pre_aggregate_join_measures",
    "pre_aggregation_through_join",
    "pre_aggregation_through_reordered_join",
]
