#!/usr/bin/env python3
"""Draw `topn_heap.svg` -- why a top-N is not a sort followed by a slice.

Source of truth: `crates/bc-interp/src/ops/mod.rs` -- `heap_select_k` (:1049, a
`BinaryHeap<(u64 rank, u32 row)>` capped at `k`), the per-row test at :1055, the
`TOP_K_SELECT_RATIO = 2` gate at :964, and `parallel_top_n` (:1336), which gathers the
wide payload exactly once through `arrow::compute::interleave` at :1462. The shared
morsel-level bound is `crates/bc-runtime/src/topn.rs::TopNBound`. The fusion that
creates `Sort { limit }` in the first place is Kyber's `fuse_topn`
(`python/batcher/kyber/rules/fusion.py:66`).

The comparison is what the diagram is for: both lanes return the same `k` rows, and the
difference is how much of the relation each one has to order and how much of it each one
has to copy. Drawn as two lanes because the claim is a before-and-after, not a sequence.

One honesty note kept in the figure: the heap is only reached when `k` is small relative
to the morsel (`k * 2 <= rows`). Above that the engine sorts the morsel and slices it,
because a linear sort of a morsel costs no more than selecting from it.
"""

from __future__ import annotations

from _authoring import BLUE_MID, GREY, arrow, band, card, curve, label, note, svg, write

W, H = 980, 560


def chip(x: float, y: float, w: float, h: float, kind: str = "blue") -> str:
    fill = {"blue": BLUE_MID, "grey": GREY}[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
        f'fill-opacity="0.28" stroke="{fill}" stroke-width="1.4"/>'
    )


def morsels(x: float, y: float, kind: str = "blue") -> str:
    return "".join(chip(x + i * 33, y, 28, 40, kind) for i in range(6))


body: list[str] = []

# ---- Lane 1: what sort-then-slice costs ---------------------------------------------
body += [
    band(20, 24, 940, 148, "SORT, THEN SLICE", "grey"),
    note(126, 66, "morsels", anchor="middle"),
    morsels(44, 76, "grey"),
    arrow(248, 96, 306, 96),
    label(277, 84, "concat", anchor="middle", size=11.5),
    card(312, 68, 214, 56, "the whole relation", "one giant batch"),
    arrow(536, 96, 594, 96),
    label(565, 84, "full sort", anchor="middle", size=11.5),
    card(600, 68, 214, 56, "every row ordered", "every column gathered"),
    arrow(824, 96, 882, 96),
    label(853, 84, "slice", anchor="middle", size=11.5),
    chip(888, 76, 54, 40, "grey"),
    note(915, 136, "k rows", anchor="middle"),
    note(
        490,
        158,
        "The relation is copied twice and ordered once, to keep k rows of it.",
        anchor="middle",
    ),
]

# ---- Lane 2: what the engine does ----------------------------------------------------
body += [
    band(20, 190, 940, 296, "TOP-N: ops::parallel_top_n", "blue"),
    note(126, 234, "morsels", anchor="middle"),
    morsels(44, 244),
    arrow(248, 264, 306, 264),
    label(277, 252, "select k", anchor="middle", size=11.5),
    card(312, 236, 214, 56, "heap_select_k", "a heap of k, per morsel"),
    note(419, 312, "a row that cannot reach the answer costs", anchor="middle"),
    note(419, 330, "one comparison against the current worst", anchor="middle"),
    arrow(536, 264, 594, 264),
    label(565, 252, "keys only", anchor="middle", size=11.5),
    card(600, 236, 214, 56, "narrow candidates", "the payload stays behind"),
    arrow(824, 264, 882, 264),
    label(853, 252, "merge", anchor="middle", size=11.5),
    chip(888, 244, 54, 40),
    note(915, 304, "k rows", anchor="middle"),
    note(707, 330, "ties break on (morsel, row), so the", anchor="middle"),
    note(707, 348, "survivor matches the stable sort's", anchor="middle"),
    card(340, 386, 300, 56, "gather the payload once", "interleave, over k rows"),
    curve(908, 306, 908, 414, 646, 414, "amber"),
    label(676, 442, "the wide columns are touched once, at the end", size=11.5),
    note(180, 414, "The relation is never", anchor="middle"),
    note(180, 432, "concatenated and never", anchor="middle"),
    note(180, 450, "fully sorted.", anchor="middle"),
]

body.append(
    note(
        490,
        508,
        "The heap is used when k is small against the morsel. Above that the morsel is sorted "
        "and sliced, because a linear sort costs no more.",
        anchor="middle",
    )
)
body.append(
    note(
        490,
        530,
        "A shared bound can skip a whole morsel whose key range cannot reach the answer, and "
        "switches itself off after 32 checks that excluded nothing.",
        anchor="middle",
    )
)

write("topn_heap", svg(W, H, "".join(body)))
print("wrote topn_heap.svg")
