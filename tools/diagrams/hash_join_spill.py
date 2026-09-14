#!/usr/bin/env python3
"""Draw `hash_join_spill.svg` -- the grace hash join, and what a bucket costs.

Source of truth: `crates/bc-interp/src/par.rs:250` (`admit`, the spill decision) and
`:1666` (the build-side estimate, Arrow bytes plus 12 B per build row from
`bc_runtime::join::estimate_build_bytes`); `crates/bc-interp/src/join_par/mod.rs` --
`spilling_hash_join_streaming` (:145), `grace_fanout` (:247), `join_bucket` (:314),
`stream_probe_bucket` (:337) and `split_and_join_bucket` (:389);
`crates/bc-interp/src/spill_split.rs` -- `grace_bucket_count` (2 to 256) and
`MAX_GRACE_SPLIT_DEPTH = 3`.

Four things the figure is careful to state, because each one is easy to draw wrongly:
  * **The build side is always the right input.** The planner swaps the inputs; the
    runtime never builds on the left.
  * **Both sides are partitioned and both are written to disk.** A picture that spills
    only the build side is describing a different algorithm.
  * **Only the build bucket is resident.** The probe bucket streams past it in chunks,
    so the pair's memory cost is one build bucket, not two buckets.
  * **The re-split cannot save a hot key.** It is a re-hash, and a re-hash cannot
    separate rows that share a key. The depth bound exists because of that, not in
    spite of it.

Layout: the admission branch, then the fan-out, then one bucket pair with its recursion.
"""

from __future__ import annotations

from _authoring import BLUE_MID, GREY, arrow, band, card, curve, label, note, svg, write

W, H = 980, 652


def chip(x: float, y: float, w: float, h: float, kind: str = "blue") -> str:
    fill = {"blue": BLUE_MID, "grey": GREY}[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
        f'fill-opacity="0.28" stroke="{fill}" stroke-width="1.4"/>'
    )


def batches(x: float, y: float, n: int = 4, kind: str = "blue") -> str:
    return "".join(chip(x + i * 35, y, 30, 32, kind) for i in range(n))


def stack(x: float, y: float, w: float, n: int, kind: str = "blue") -> str:
    return "".join(chip(x, y + i * 24, w, 18, kind) for i in range(n))


body: list[str] = []

# ---- Admission ----------------------------------------------------------------------
body += [
    band(20, 24, 940, 96, "DOES THE BUILD SIDE FIT?", "grey"),
    card(352, 54, 276, 54, "admit(build bytes)", "Arrow bytes + 12 B per build row"),
    arrow(348, 81, 262, 81),
    label(305, 67, "fits", anchor="middle", size=11.5),
    card(44, 54, 214, 54, "one hash table", "on the right, probed by the left"),
    arrow(490, 106, 490, 142),
    label(504, 130, "does not fit", size=11.5),
]

# ---- The grace fan-out ---------------------------------------------------------------
body += [
    band(20, 148, 940, 232, "PARTITION BOTH SIDES BY THE JOIN KEY", "blue"),
    note(114, 190, "left rows (probe)", anchor="middle"),
    batches(44, 200),
    arrow(189, 216, 366, 216),
    label(277, 204, "hash the join key", anchor="middle", size=11.5),
    stack(376, 190, 150, 3),
    note(451, 276, "join-left/part-i.arrow", anchor="middle"),
    note(114, 298, "right rows (build)", anchor="middle"),
    batches(44, 308),
    arrow(189, 324, 366, 324),
    label(277, 312, "the same hash, same buckets", anchor="middle", size=11.5),
    stack(376, 298, 150, 3),
    note(451, 366, "join-right/part-i.arrow", anchor="middle"),
    note(762, 196, "One batch at a time, so neither side", anchor="middle"),
    note(762, 214, "is ever fully materialized.", anchor="middle"),
    note(762, 300, "Bucket count is sized from the LARGER", anchor="middle"),
    note(762, 318, "side divided by the budget, from 2 to 256.", anchor="middle"),
    note(762, 336, "Equal keys co-partition, so the union of the", anchor="middle"),
    note(762, 354, "per-bucket joins is the full join, every type.", anchor="middle"),
]

# ---- One bucket pair ------------------------------------------------------------------
body += [
    arrow(451, 388, 451, 420),
    label(465, 410, "one pair at a time", size=11.5),
    band(20, 396, 940, 214, "JOIN ONE BUCKET PAIR", "amber"),
    note(84, 450, "build bucket i", anchor="middle"),
    chip(44, 458, 80, 22),
    note(84, 578, "probe bucket i", anchor="middle"),
    chip(44, 586, 80, 22),
    arrow(130, 470, 224, 494),
    label(140, 462, "read whole", size=11.5),
    arrow(130, 590, 224, 524),
    label(138, 566, "streamed in chunks", size=11.5),
    card(230, 478, 250, 62, "join the bucket pair", "index pairs per chunk"),
    arrow(486, 509, 544, 509),
    label(515, 497, "indices", anchor="middle", size=11.5),
    card(550, 479, 190, 60, "output rows", "one bucket's share"),
    note(858, 470, "Only the build bucket is", anchor="middle"),
    note(858, 488, "resident. The probe side", anchor="middle"),
    note(858, 506, "streams past it, so the cost", anchor="middle"),
    note(858, 524, "is one bucket, not two.", anchor="middle"),
    curve(478, 544, 355, 590, 232, 544, "amber"),
    label(
        355,
        602,
        "build bucket still over budget: re-partition it with a fresh salt, at most 3 deep",
        anchor="middle",
        size=11.5,
    ),
]

body.append(
    note(
        490,
        636,
        "A re-split is a re-hash, so it cannot separate rows that share a key. "
        "One hot key stays in one bucket at every level, which is why the depth is bounded.",
        anchor="middle",
    )
)

write("hash_join_spill", svg(W, H, "".join(body)))
print("wrote hash_join_spill.svg")
