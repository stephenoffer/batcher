"""Learned strategy + parameter tuning — self-tuning physical decisions from measured runs.

Kyber's physical choices (which join algorithm, the broadcast byte threshold, the sort-merge
row crossover, whether to pre-aggregate, how many partitions, a per-task CPU share) ship as
*static* constants tuned for one cluster. This module closes the same learning loop the GPU
crossover (`gpu/adaptive.py`) and cost calibration (`calibration.py`) close: every decision reads
a **measured/learned** signal from the `MetadataHub`, keyed by plan signature, and on a cold store
falls back to the current default so a first run is byte-identical.

Every decision here ranges over **semantically-equivalent** alternatives — hash vs broadcast vs
sort-merge all emit the same relation, a partition count only shards, a CPU share only schedules —
so the tuned choice changes *performance*, never the *result*. That invariance is what lets Kyber
learn aggressively: the worst a wrong learned value can do is cost throughput, never correctness.

Two reusable primitives back the family:

* a **UCB1 bandit** (`ucb1_best_arm` / `learned_arm`) — regret-minimizing selection over a fixed
  arm set from measured per-arm latencies; deterministic (no RNG, ties broken by arm name), so a
  plan is reproducible. It generalizes the two-arm GPU crossover to N discrete algorithm arms.
* an **OLS line-crossover** (`_fit` / `_solve_crossover`) — the exact machinery `gpu/adaptive.py`
  uses, fitting `t ≈ a + b·x` per algorithm and solving for the x (bytes or rows) where the
  cheaper-below algorithm is overtaken by the cheaper-above one, clamped to a band around the
  default so one noisy early fit can't send a threshold to an absurd value.

Everything is best-effort: a malformed bucket, a degenerate fit, or a cold store yields the
default (or `None`), never an exception into planning or execution. **Core measures, Kyber
consumes** — the `record_*` functions fold one observation into O(1) sufficient statistics; the
`learned_*` functions read them back and decide.
"""

from __future__ import annotations

# `_reward_scale` / `_smooth` / `_fold_ols` are re-exported (redundant alias = an explicit
# re-export) because they are the family's tested primitives, named as such in the audit ledger.
from batcher.kyber.learned_tuning.bandit import _reward_scale as _reward_scale
from batcher.kyber.learned_tuning.bandit import (
    learned_arm,
    learned_join_strategy,
    record_arm,
    record_join_strategy,
    ucb1_best_arm,
)
from batcher.kyber.learned_tuning.crossover import _fold_ols as _fold_ols
from batcher.kyber.learned_tuning.crossover import (
    learned_broadcast_max_bytes,
    learned_sort_merge_min_rows,
    record_broadcast_timing,
    record_sort_merge_timing,
)
from batcher.kyber.learned_tuning.priors import _smooth as _smooth
from batcher.kyber.learned_tuning.priors import (
    learned_adaptive_helps,
    learned_build_sides,
    learned_partial_agg,
    learned_partition_count,
    learned_signature_rows,
    record_adaptive_flip,
    record_group_reduction,
    record_join_sides,
    record_partition_rows,
)

__all__ = [
    "learned_adaptive_helps",
    "learned_arm",
    "learned_broadcast_max_bytes",
    "learned_build_sides",
    "learned_join_strategy",
    "learned_partial_agg",
    "learned_partition_count",
    "learned_signature_rows",
    "learned_sort_merge_min_rows",
    "record_adaptive_flip",
    "record_arm",
    "record_broadcast_timing",
    "record_group_reduction",
    "record_join_sides",
    "record_join_strategy",
    "record_partition_rows",
    "record_sort_merge_timing",
    "ucb1_best_arm",
]
