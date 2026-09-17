#!/usr/bin/env python3
"""Draw `group_by_flow.svg` -- rows become groups, and each group becomes one output row.

Source of truth: `docs/user-guide/analyze/aggregations.md`. The data is the page's setup
dataset (`category` a, b, a, b, a with `price` 10, 20, 30, 40, 50). The group sums (a 90.0,
b 60.0) are the page's `bt.sum("price")` output, and the counts (a 3, b 2) its
`orders=bt.count()` output. The no-key row (`total` 150.0, `rows` 5) is the page's
`group_by().agg(total=..., rows=bt.count())` example, printed there verbatim.
"""

from __future__ import annotations

from _authoring import (
    AMBER,
    BLUE_MID,
    FONT,
    MONO,
    arrow,
    band,
    code,
    heading,
    label,
    note,
    svg,
    write,
)

W, H = 980, 516

CELL_H = 30
FILL = {"a": BLUE_MID, "b": AMBER}


def row(x: float, y: float, cells: list[str], widths: list[int], group: str) -> str:
    """One table row, tinted by its group. The key text carries the group, not the tint."""
    out = ""
    left = x
    for text, w in zip(cells, widths, strict=True):
        tint = FILL.get(group, "#94a3b8")
        out += (
            f'<rect x="{left}" y="{y}" width="{w}" height="{CELL_H}" fill="{tint}" '
            f'fill-opacity="0.16" stroke="{tint}" stroke-opacity="0.7" stroke-width="1"/>'
            f'<text x="{left + w / 2}" y="{y + 20}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="13" font-weight="600" class="t-title">{text}</text>'
        )
        left += w
    return out


def header(x: float, y: float, names: list[str], widths: list[int]) -> str:
    """Column names over a table, in the code face."""
    out = ""
    left = x
    for name, w in zip(names, widths, strict=True):
        out += (
            f'<text x="{left + w / 2}" y="{y}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="12" font-weight="700" class="t-code">{name}</text>'
        )
        left += w
    return out


INPUT = (("a", "10.0"), ("b", "20.0"), ("a", "30.0"), ("b", "40.0"), ("a", "50.0"))
IN_W = [90, 80]
GRP_W = [60, 70]
OUT_W = [90, 70, 56]

body = [
    band(20, 20, 940, 344, "FIVE ROWS IN, ONE ROW PER GROUP OUT", "blue"),
    # Input rows, in arrival order.
    heading(125, 82, "INPUT ROWS", anchor="middle"),
    header(40, 112, ["category", "price"], IN_W),
]
for i, (cat, price) in enumerate(INPUT):
    body.append(row(40, 124 + i * CELL_H, [cat, price], IN_W, cat))

# Groups: rows sharing a key land together.
body += [
    arrow(222, 199, 330, 199),
    label(276, 184, "group_by", anchor="middle", size=12),
    note(276, 220, "same key,", anchor="middle"),
    note(276, 236, "same group", anchor="middle"),
    heading(412, 82, "GROUPS", anchor="middle"),
    header(347, 112, ["key", "price"], GRP_W),
]
y = 124
for cat, prices in (("a", ("10.0", "30.0", "50.0")), ("b", ("20.0", "40.0"))):
    for p in prices:
        body.append(row(347, y, [cat, p], GRP_W, cat))
        y += CELL_H
    y += 12

# One output row per group.
body += [
    arrow(496, 169, 640, 169),
    label(568, 158, "agg", anchor="middle", size=12),
    note(568, 207, "each group reduces", anchor="middle"),
    note(568, 223, "to one row", anchor="middle"),
    arrow(496, 256, 640, 256),
    label(568, 245, "agg", anchor="middle", size=12),
    heading(756, 82, "OUTPUT", anchor="middle"),
    header(648, 112, ["category", "total", "rows"], OUT_W),
    row(648, 154, ["a", "90.0", "3"], OUT_W, "a"),
    row(648, 241, ["b", "60.0", "2"], OUT_W, "b"),
    code(
        60,
        306,
        ['ds.group_by("category").agg(total=bt.col("price").sum(), rows=bt.count())'],
        860,
        size=12.5,
    ),
    # No keys at all: the whole dataset is one group.
    band(20, 384, 940, 112, "NO KEYS: THE WHOLE DATASET IS ONE GROUP", "grey"),
    f'<text x="60" y="450" font-family="{MONO}" font-size="12.5" class="t-code">'
    "ds.group_by().agg(total=..., rows=bt.count())</text>",
    arrow(430, 446, 602, 446, "grey"),
    label(516, 436, "one output row", anchor="middle", size=12),
    header(648, 428, ["total", "rows"], [90, 70]),
    row(648, 438, ["150.0", "5"], [90, 70], ""),
]

write("group_by_flow", svg(W, H, "".join(body)))
print("wrote group_by_flow.svg")
