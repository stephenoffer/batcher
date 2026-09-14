#!/usr/bin/env python3
"""Draw `join_strategy_choice.svg` -- how a join strategy and a build side are picked.

Source of truth: `python/batcher/kyber/rules/selection.py` -- `adaptive_build_side` (:134),
the broadcast/cost branch (:365-412), the strategy assignment (:481-487),
`SORT_MERGE_MIN_ROWS = 50_000_000` (:61) scaled by the worker count (:212), the memory
share `_SORT_MERGE_MEMORY_SHARE = 6.0` (:92), and the bandit override (:313-319).
The ceiling itself is `OptimizerConfig.resolved_broadcast_max_bytes`
(`python/batcher/config/config.py:1038`): a quarter of L3, and on a cluster that times 16
with a 64 MiB floor. The run-time re-check is `python/batcher/dist/executors/join.py:504`,
which falls back to the shuffle join when the measured build side is over the ceiling.

Two corrections this diagram makes to the obvious three-way picture:
  * The three strategies are **broadcast**, **hash** (the co-partitioned shuffle) and
    **sort_merge**. There is no separate "bucketed join": the shuffle join *is* the
    co-partitioned one, and there is no pre-bucketed-table path.
  * "Choosing the build side" means "deciding whether to swap the inputs". The runtime
    always builds on the right, and only an inner join may be swapped.

Layout: the inputs, then the decision as the tree it is, then the two later points where
the same decision is made again on better information.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 632

body: list[str] = [
    band(20, 24, 940, 98, "WHAT THE DECISION IS MADE FROM", "grey"),
    label(200, 64, "estimated rows x row width", anchor="middle", size=12),
    note(200, 84, "per side, from Kyber's sketches", anchor="middle"),
    label(490, 64, "the join type", anchor="middle", size=12),
    note(490, 84, "only an inner join may swap sides", anchor="middle"),
    label(790, 64, "the broadcast ceiling", anchor="middle", size=12),
    note(790, 84, "a quarter of L3; times 16, at least 64 MiB, on a cluster", anchor="middle"),
]

# ---- The tree -------------------------------------------------------------------------
body += [
    band(20, 138, 940, 330, "THE DECISION", "blue"),
    card(340, 170, 300, 62, "smaller side under the ceiling?", "the right side, unless inner"),
    arrow(336, 201, 232, 201),
    label(284, 187, "yes", anchor="middle", size=11.5),
    card(44, 171, 184, 60, "broadcast", "the probe never moves"),
    arrow(490, 236, 490, 274),
    label(504, 262, "no", size=11.5),
    card(340, 280, 300, 62, "build side very large?", "50M rows per worker"),
    arrow(336, 311, 232, 311),
    label(284, 297, "yes", anchor="middle", size=11.5),
    card(44, 281, 184, 60, "sort_merge", "no hash table at all"),
    arrow(490, 346, 490, 384),
    label(504, 372, "no", size=11.5),
    card(340, 390, 300, 60, "hash: the shuffle join", "both sides partitioned by key"),
    note(800, 176, "The smaller side becomes the build side, and", anchor="middle"),
    note(800, 194, "the runtime always builds on the right, so", anchor="middle"),
    note(800, 212, "Kyber swaps the inputs when it is the left.", anchor="middle"),
    note(800, 230, "Only an inner join may be swapped.", anchor="middle"),
    note(800, 292, "The row floor stands down only when the byte", anchor="middle"),
    note(800, 310, "size is also large, or is a scan's bounded", anchor="middle"),
    note(800, 328, "ceiling rather than a compounding guess.", anchor="middle"),
    note(800, 404, "Broadcast is limited to inner, left, semi", anchor="middle"),
    note(800, 422, "and anti. Every other type shuffles.", anchor="middle"),
]

# ---- Re-decided twice more --------------------------------------------------------------
body += [
    arrow(400, 474, 268, 504),
    label(272, 488, "the chosen arm", size=11.5),
    band(20, 484, 940, 106, "AND THE SAME DECISION IS MADE TWICE MORE", "amber"),
    card(120, 512, 286, 58, "the bandit, before the run", "substitutes a learned arm"),
    arrow(412, 541, 464, 541),
    label(438, 529, "then", anchor="middle", size=11.5),
    card(470, 512, 300, 58, "the driver, at run time", "a broadcast that no longer fits shuffles"),
    note(866, 534, "measured bytes", anchor="middle"),
    note(866, 552, "beat estimates", anchor="middle"),
]

body.append(note(490, 614, "All three produce the same relation, so a wrong pick is slow rather than wrong. "
                           "That is what makes the choice safe to learn across runs.", anchor="middle"))

write("join_strategy_choice", svg(W, H, "".join(body)))
print("wrote join_strategy_choice.svg")
