#!/usr/bin/env python3
"""Draw `sort_run_merge.svg` -- the external merge sort, from admission to the last pass.

Source of truth: `crates/bc-interp/src/ops/external_sort.rs` (the run loop at :106,
`spill_run` at :244, the multi-pass merge at :135, the min-heap at :384, and the
`SpillTruncated` row check at :170), `crates/bc-interp/src/par.rs:1225` (`admit`, which
decides in-memory against spill, and the per-operator run target at :1241), and
`crates/bc-arrow/src/lib.rs:226` (`sort_merge_fanin`, default 16).

Three facts the obvious picture gets wrong, and this diagram is drawn around:
  * **A run is bounded in bytes, not rows** -- `min(operator budget / 4, 64 MiB)`, floored
    at 1 MiB. There is no run row count.
  * **The merge is multi-pass.** With a fan-in of 16, 300 runs do not merge at once; they
    merge to 19, then to 2, then to 1. The loop is the shape of the operator.
  * **The in-memory path is not a degenerate spill.** `admit` returns before any of this,
    and `parallel_sort_batch` touches no disk at all.

Layout: the admission branch first, then the two passes stacked, with the merge drawn as
the loop it is rather than as a single fan-in.
"""

from __future__ import annotations

from _authoring import BLUE_MID, GREY, arrow, band, card, curve, label, note, svg, write

W, H = 980, 600


def chip(x: float, y: float, w: float, h: float, kind: str = "blue") -> str:
    """A data rectangle: a morsel, or a run file on disk."""
    fill = {"blue": BLUE_MID, "grey": GREY}[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
        f'fill-opacity="0.28" stroke="{fill}" stroke-width="1.4"/>'
    )


def stack(x: float, y: float, w: float, n: int, kind: str = "blue") -> str:
    """`n` run files drawn as a stack, so a count reads without a number."""
    return "".join(chip(x, y + i * 24, w, 18, kind) for i in range(n))


body: list[str] = []

# ---- The admission branch ----------------------------------------------------------
body += [
    band(20, 24, 940, 108, "DOES THE SORT FIT IN ITS ENVELOPE?", "grey"),
    card(360, 54, 260, 54, "admit(op_id, bytes)", "the memory pool answers"),
    arrow(356, 81, 270, 81),
    label(313, 67, "fits", anchor="middle", size=11.5),
    card(48, 54, 216, 54, "parallel_sort_batch", "no disk at all"),
    arrow(490, 112, 490, 154),
    label(504, 140, "does not fit", size=11.5),
]

# ---- Pass 0: build and spill the runs -----------------------------------------------
body += [
    band(20, 160, 940, 184, "PASS 0: FILL A RUN, SORT IT, SPILL IT", "blue"),
    note(129, 202, "morsels", anchor="middle"),
]
x = 44
for _ in range(5):
    body.append(chip(x, 212, 30, 40))
    x += 35
body += [
    arrow(222, 232, 280, 232),
    label(251, 220, "accumulate", anchor="middle", size=11.5),
    card(286, 204, 176, 60, "one run", "budget/4, max 64 MiB"),
    arrow(472, 232, 530, 232),
    label(501, 220, "sort_batch", anchor="middle", size=11.5),
    card(536, 204, 176, 60, "write it out", "Arrow IPC stream"),
    arrow(722, 232, 780, 232),
    label(751, 220, "one file", anchor="middle", size=11.5),
    stack(790, 196, 150, 4),
    note(865, 300, "one run per file, closed at once", anchor="middle"),
    note(
        400,
        300,
        "each input batch is dropped as the run is written, so the relation is never all resident",
        anchor="middle",
    ),
]

# ---- Pass 1..n: merge, bounded fan-in, repeated ---------------------------------------
body += [
    arrow(865, 348, 865, 386),
    label(760, 372, "every run, on disk", size=11.5),
    band(20, 356, 940, 194, "PASS 1..n: MERGE 16 RUNS AT A TIME, AND REPEAT", "amber"),
    stack(44, 400, 130, 3),
    note(109, 484, "runs in", anchor="middle"),
    arrow(184, 424, 242, 424),
    label(213, 412, "fan-in 16", anchor="middle", size=11.5),
    card(248, 396, 240, 60, "min-heap of run heads", "one batch per reader"),
    arrow(498, 424, 556, 424),
    label(527, 412, "merged", anchor="middle", size=11.5),
    stack(562, 412, 130, 2),
    note(627, 484, "runs out", anchor="middle"),
    curve(627, 466, 368, 524, 109, 466, "amber"),
    note(368, 538, "repeat while more than one run remains", anchor="middle"),
    arrow(702, 424, 760, 424),
    label(731, 412, "last pass", anchor="middle", size=11.5),
    card(766, 396, 180, 60, "sorted rows", "16,384 at a time"),
]

body.append(
    note(
        490, 572, "The merged row count is checked against the rows that went in.", anchor="middle"
    )
)
body.append(
    note(
        490,
        590,
        "A truncated spill file reads back as a valid shorter stream, which is a sorted "
        "prefix rather than an error.",
        anchor="middle",
    )
)

write("sort_run_merge", svg(W, H, "".join(body)))
print("wrote sort_run_merge.svg")
