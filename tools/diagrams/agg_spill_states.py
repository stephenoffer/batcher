#!/usr/bin/env python3
"""Draw `agg_spill_states.svg` -- what an aggregation holds, and what it spills.

Source of truth: `crates/bc-runtime/src/agg/group/assign.rs` (dense group ids: a
`hashbrown::HashTable<u32>` over first-seen rows, or a direct map, or nothing at all when
the key arrives sorted), `crates/bc-runtime/src/agg/mod.rs` (`Partial { group_columns,
states }`, `partial` at :444, `combine` at :515, `finalize` re-exported at :126, and
`AggFunc::state_arity` at :296), and `crates/bc-runtime/src/agg/spill/mod.rs`
(`combine_finalize_spilling` at :52, the routing loop at :60, `pack_partial` at :299, the
per-partition merge at :81 and the recursive split at :163).

Three claims the figure makes deliberately, because each is the opposite of the obvious
guess:
  * **What spills is partial state, not input rows.** `pack_partial` writes the group
    columns followed by each aggregate's state columns.
  * **Everything spills.** The routing loop sends every partial to a partition; there is
    no resident subset and this is not an eviction policy. Peak memory is one partition.
  * **A group is never split.** A key always hashes to the same partition, which is the
    whole reason merging one partition at a time is the global aggregate.

Layout: the three phases stacked, with the merge drawn with its re-split recursion.
"""

from __future__ import annotations

from _authoring import BLUE_MID, arrow, band, card, curve, label, note, svg, write

W, H = 980, 668


def chip(x: float, y: float, w: float, h: float) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{BLUE_MID}" '
        f'fill-opacity="0.28" stroke="{BLUE_MID}" stroke-width="1.4"/>'
    )


def row_of(x: float, y: float, n: int, w: float, h: float, step: float) -> str:
    return "".join(chip(x + i * step, y, w, h) for i in range(n))


def stack(x: float, y: float, w: float, n: int) -> str:
    return "".join(chip(x, y + i * 24, w, 18) for i in range(n))


body: list[str] = [
    band(20, 24, 940, 176, "PARTIAL: A STATE, NOT AN ANSWER", "blue"),
    note(126, 68, "morsels", anchor="middle"),
    row_of(44, 78, 6, 28, 40, 33),
    arrow(247, 98, 305, 98),
    label(278, 86, "group ids", anchor="middle", size=11.5),
    card(311, 70, 214, 56, "dense group ids", "a hash table, or none at all"),
    arrow(535, 98, 593, 98),
    label(564, 86, "scatter", anchor="middle", size=11.5),
    card(599, 70, 280, 56, "state columns per group", "Arrow, one row per group"),
    note(490, 152, "The state is not the answer. mean carries (sum, count); var carries (mean, M2, count);",
         anchor="middle"),
    note(490, 170, "median carries the group's values as a list. Only finalize turns one into a number.",
         anchor="middle"),

    arrow(490, 202, 490, 228),
    label(504, 220, "partials", size=11.5),

    band(20, 232, 940, 180, "SPILL: ROUTE EVERY PARTIAL BY A HASH OF THE GROUP KEY", "amber"),
    note(126, 274, "partials", anchor="middle"),
    row_of(44, 284, 6, 28, 36, 33),
    arrow(247, 302, 400, 302),
    label(324, 290, "hash the group key", anchor="middle", size=11.5),
    stack(410, 268, 150, 4),
    note(485, 380, "part-0 .. part-P, Arrow IPC", anchor="middle"),
    note(772, 274, "Every partial is spilled. Nothing is kept", anchor="middle"),
    note(772, 292, "resident, so this is not an eviction policy.", anchor="middle"),
    note(772, 326, "P is the state bytes over the budget, 2 to 256.", anchor="middle"),
    note(772, 360, "A key always hashes to the same partition,", anchor="middle"),
    note(772, 378, "so a group is never split across two of them.", anchor="middle"),

    arrow(485, 414, 485, 440),
    label(499, 432, "one partition at a time", size=11.5),

    band(20, 444, 940, 180, "MERGE ONE PARTITION AT A TIME", "blue"),
    note(109, 486, "partition i", anchor="middle"),
    stack(44, 494, 130, 2),
    arrow(184, 512, 242, 512),
    label(213, 500, "read it", anchor="middle", size=11.5),
    card(248, 484, 200, 58, "combine", "merge the states by key"),
    arrow(454, 513, 512, 513),
    label(483, 501, "one state", anchor="middle", size=11.5),
    card(518, 484, 200, 58, "finalize", "the answer at last"),
    arrow(724, 513, 782, 513),
    label(753, 501, "rows", anchor="middle", size=11.5),
    card(788, 484, 158, 58, "output rows", "this partition's groups"),
    curve(446, 548, 345, 590, 244, 548, "amber"),
    label(345, 606, "partition still over budget: re-partition it with a fresh salt", anchor="middle", size=11.5),
]

body.append(note(490, 644, "Peak memory is one partition, not one hash table.", anchor="middle"))
body.append(note(490, 662, "This is the algebra the distributed path runs, with combine reading from disk "
                           "instead of from the network.", anchor="middle"))

write("agg_spill_states", svg(W, H, "".join(body)))
print("wrote agg_spill_states.svg")
