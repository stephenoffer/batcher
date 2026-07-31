"""How a plan splits across workers — the neutral algebra both the optimizer and the backends read.

Distributing an operator is a scheduling concern, but *whether* an operator can be distributed
is a property of the plan, and two layers need the answer: `kyber` decides where to route a
plan (one that shards is bounded by its shard size, not by one machine's memory), and `dist`
builds the fan-out. Those layers cannot import each other, so the algebra lives here, stated
once. Two statements of it is the one way the routing and the execution could disagree, and a
disagreement there is a wrong answer rather than a slow one.
"""

from __future__ import annotations

from batcher.plan.distribution.mergeable import (
    BROADCAST_SAFE_JOINS,
    ROW_LOCAL_OPS,
    ShardSplit,
    decompose,
    flatten_ops,
    nest_ops,
    recombine,
    shard_plan,
)
from batcher.plan.distribution.tree_shard import (
    LEFT_DRIVEN_JOINS,
    RIGHT_DRIVEN_JOINS,
    shardable_leaves,
)

__all__ = [
    "BROADCAST_SAFE_JOINS",
    "LEFT_DRIVEN_JOINS",
    "RIGHT_DRIVEN_JOINS",
    "ROW_LOCAL_OPS",
    "ShardSplit",
    "decompose",
    "flatten_ops",
    "nest_ops",
    "recombine",
    "shard_plan",
    "shardable_leaves",
]
