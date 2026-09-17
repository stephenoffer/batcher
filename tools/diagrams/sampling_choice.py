#!/usr/bin/env python3
"""Draw `sampling_choice.svg` -- which sampling or splitting operator to reach for.

Source of truth: `docs/user-guide/transform/rows/sampling.md`, where it is embedded:
`sample(fraction)` keeps rows whose seeded value hash falls under the fraction and streams,
with a binomial size; `sample(n=...)` keeps the `n` smallest-hash rows and so is a breaker;
`stratified_split(by, test_size)` ranks rows by hash within each group; `split_at_indices`
and `split_proportionately` cut at row positions and need a sort first; and a number rather
than rows is a sketch (`approx_count_distinct`, `approx_quantile`). `ml.train_test_split`
with `key=` is the page's modeling split.

Layout: the questions as a ladder, each "yes" exit to the right, and the fall-through, a
plain fraction, at the bottom.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, step, svg, tint, write

W, H = 980, 600
CX = 245
CW = 340
X0 = CX - CW / 2
RX = 596
RW = 344
YS = (58, 156, 254, 352)
QH = 56

LADDER = [
    (
        "A number, not rows?",
        "a distinct count, a quantile",
        "approx_count_distinct, approx_quantile",
        "a mergeable sketch, one pass",
    ),
    (
        "Consecutive ranges?",
        "a chronological holdout",
        "split_at_indices, split_proportionately",
        "cut by position, so sort first",
    ),
    (
        "Each group keeps its share?",
        "a rare class must not starve",
        "stratified_split(by, test_size)",
        "hashed within each group",
    ),
    (
        "Exactly n rows?",
        "a fixture of a known size",
        "sample(n=...)",
        "ranks every row by hash: a breaker",
    ),
]

body: list[str] = [band(20, 20, 940, 504, "WHICH SAMPLE OR SPLIT?", "grey")]
for i, (y, (q, qs, a, asub)) in enumerate(zip(YS, LADDER, strict=True)):
    mid = y + QH / 2
    body += [
        step(52, mid, i + 1),
        card(X0, y, CW, QH, q, qs),
        arrow(X0 + CW + 4, mid, RX - 6, mid),
        label((X0 + CW + RX) / 2, mid - 10, "yes", anchor="middle", size=11.5),
        tint(RX, y, RW, QH, a, asub, "blue"),
        arrow(CX, y + QH + 4, CX, y + 94),
        label(CX + 12, y + QH + 26, "no", size=11.5),
    ]
body += [
    tint(X0, 450, CW, 50, "sample(fraction, seed=)", "streams, and the size is binomial", "amber"),
    note(RX, 466, "Disjoint train and test sets: ml.train_test_split"),
    note(RX, 484, "with key= a stable id, so a row never changes side."),
    note(
        490,
        556,
        "Every row-choosing form but the positional one assigns rows by a seeded hash "
        "of their values,",
        anchor="middle",
    ),
    note(
        490,
        576,
        "so one seed gives the same rows however the data is laid out or distributed.",
        anchor="middle",
    ),
]

write("sampling_choice", svg(W, H, "".join(body)))
print("wrote sampling_choice.svg")
