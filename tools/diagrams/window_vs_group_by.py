#!/usr/bin/env python3
"""Draw `window_vs_group_by.svg` -- group_by collapses each group, a window keeps every row.

Source of truth: `docs/user-guide/analyze/window-functions.md`. The input is the page's `ds`
(`category` a, a, a, b, b; `product` x, y, z, p, q; `price` 30, 10, 20, 40, 15). The window
output is its `ds.window(partition_by=["category"], functions={"cat_total": ("sum",
"price")})` example, printed there as `cat_total` 60, 60, 60, 55, 55. The collapsed output is
the same sum as a `group_by("category").agg(...)`, which the page's opening names as the
thing a window does not do.
"""

from __future__ import annotations

from _authoring import AMBER, BLUE_MID, FONT, MONO, arrow, band, code, label, note, svg, write

W, H = 980, 490

CELL_H = 30
FILL = {"a": BLUE_MID, "b": AMBER}


def row(x: float, y: float, cells: list[str], widths: list[int], group: str, hot: int = -1) -> str:
    """One table row, tinted by its partition; the key text carries the partition too."""
    out, left = "", x
    tint = FILL[group]
    for i, (text, w) in enumerate(zip(cells, widths, strict=True)):
        weight = "800" if i == hot else "600"
        out += (
            f'<rect x="{left}" y="{y}" width="{w}" height="{CELL_H}" fill="{tint}" '
            f'fill-opacity="0.16" stroke="{tint}" stroke-opacity="0.7" stroke-width="1"/>'
            f'<text x="{left + w / 2}" y="{y + 20}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="13" font-weight="{weight}" class="t-title">{text}</text>'
        )
        left += w
    return out


def header(x: float, y: float, names: list[str], widths: list[int]) -> str:
    """Column names over a table, in the code face."""
    out, left = "", x
    for name, w in zip(names, widths, strict=True):
        out += (
            f'<text x="{left + w / 2}" y="{y}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="12" font-weight="700" class="t-code">{name}</text>'
        )
        left += w
    return out


INPUT = (("a", "x", "30"), ("a", "y", "10"), ("a", "z", "20"), ("b", "p", "40"), ("b", "q", "15"))
IN_W = [72, 64, 56]
TOT_W = [72, 64]
WIN_W = [72, 64, 56, 80]

IN_X, IN_Y = 40, 206

body = [
    label(IN_X, IN_Y - 42, "input: 5 rows"),
    header(IN_X, IN_Y - 10, ["category", "product", "price"], IN_W),
]
for i, (cat, prod, price) in enumerate(INPUT):
    body.append(row(IN_X, IN_Y + i * CELL_H, [cat, prod, price], IN_W, cat))

# group_by collapses.
body += [
    band(290, 20, 670, 168, "GROUP_BY: ONE ROW PER GROUP", "grey"),
    code(314, 56, ['.group_by("category")', '.agg(total=bt.col("price").sum())'], 280, size=12),
    note(314, 150, "Two rows come out. product and price are gone."),
    header(780, 72, ["category", "total"], TOT_W),
    row(780, 84, ["a", "60"], TOT_W, "a", hot=1),
    row(780, 114, ["b", "55"], TOT_W, "b", hot=1),
    arrow(236, IN_Y + 20, 284, 104, "grey"),
    label(222, 128, "collapse", anchor="middle", size=12),
]

# The window keeps every row and adds a column.
body += [
    band(290, 206, 670, 264, "WINDOW: EVERY ROW STAYS, A COLUMN IS ADDED", "blue"),
    code(
        314,
        250,
        [
            ".window(",
            '  partition_by=["category"],',
            "  functions={",
            '    "cat_total": ("sum", "price")})',
        ],
        280,
        size=12,
    ),
    note(314, 392, "Five rows come out. Each row reads"),
    note(314, 410, "its partition's total without losing"),
    note(314, 428, "its own product and price."),
    header(664, 270, ["category", "product", "price", "cat_total"], WIN_W),
]
for i, (cat, prod, price) in enumerate(INPUT):
    total = "60" if cat == "a" else "55"
    body.append(row(664, 282 + i * CELL_H, [cat, prod, price, total], WIN_W, cat, hot=3))
body += [
    arrow(236, IN_Y + 110, 284, 338),
    label(260, 360, "keep", anchor="middle", size=12),
]

write("window_vs_group_by", svg(W, H, "".join(body)))
print("wrote window_vs_group_by.svg")
