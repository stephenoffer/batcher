#!/usr/bin/env python3
"""Draw `dedup_choice.svg` -- which deduplication operator answers which question.

Source of truth: `docs/user-guide/transform/rows/distinct-and-dedup.md`, where it is
embedded, and only the operators that page covers: `distinct()` (SQL `DISTINCT *`),
`distinct(subset, keep=..., order_by=...)` with `keep="any"` taking an arbitrary row and
`"first"`/`"last"` requiring `order_by`, `ds.ml.drop_near_duplicates` (MinHash and LSH,
recall not total, every returned pair verified against `threshold`), and
`drop_duplicates_within_watermark`, which forgets a key once the event-time watermark
passes it.

Layout: the questions as a ladder, each "yes" exit to the right, and the fall-through at
the bottom of the ladder.
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
        "Nearly the same, not identical?",
        "a re-sent row, a re-headed article",
        "ml.drop_near_duplicates(col)",
        "may miss a pair, never a false drop",
    ),
    (
        "Is the input an unbounded stream?",
        "memory has to stay bounded",
        "drop_duplicates_within_watermark",
        "a key is forgotten past the watermark",
    ),
    ("Does every column define it?", "rows equal in every column", "distinct()", "SQL DISTINCT *"),
    (
        "Does the surviving row matter?",
        "the newest, the first, the paid one",
        "distinct(keys, keep=, order_by=)",
        'keep="first" or "last"',
    ),
]

body: list[str] = [band(20, 20, 940, 504, "WHICH DEDUP DO YOU NEED?", "grey")]
for i, (y, (q, qs, a, asub)) in enumerate(zip(YS, LADDER, strict=True)):
    mid = y + QH / 2
    body += [
        step(52, mid, i + 1),
        card(X0, y, CW, QH, q, qs),
        arrow(X0 + CW + 4, mid, RX - 6, mid),
        label((X0 + CW + RX) / 2, mid - 10, "yes", anchor="middle", size=11.5),
        tint(RX, y, RW, QH, a, asub, "amber" if i == 3 else "blue"),
        arrow(CX, y + QH + 4, CX, y + 94),
        label(CX + 12, y + QH + 26, "no", size=11.5),
    ]
body += [
    tint(X0, 450, CW, 50, "distinct(keys)", 'keep="any": an arbitrary row per key'),
    note(RX, 466, 'keep="any" may pick a different row between runs,'),
    note(RX, 484, "so use it only when the other columns follow the key."),
    note(
        490,
        556,
        "distinct hashes a float key canonically: 0.0 and -0.0 are one key, and every NaN is one.",
        anchor="middle",
    ),
    note(
        490,
        576,
        "Count before you drop: value_counts and is_duplicated mark rows without removing them.",
        anchor="middle",
    ),
]

write("dedup_choice", svg(W, H, "".join(body)))
print("wrote dedup_choice.svg")
