#!/usr/bin/env python3
"""Draw `pivot_long_wide.svg` -- a pivot from long to wide, and the unpivot back.

Source of truth: `docs/user-guide/analyze/pivoting.md`. The long table is the page's `sales`
setup data; the wide table is its `sales.pivot(index=["region"], on="quarter",
values="amount")` output (west has two `q1` rows, 10.0 and 5.0, which sum to 15.0); the
right-hand table is its `wide.unpivot(...)` round trip after `drop_nulls` and a sort. The
footer restates the page's comparison table: a pivot's output columns depend on the data
unless `columns=` is passed, and it groups, while an unpivot's schema is fixed by its
arguments and it streams.
"""

from __future__ import annotations

from _authoring import AMBER, BLUE_MID, FONT, MONO, arrow, band, heading, label, note, svg, write

W, H = 980, 452

CELL_H = 30
LONG_W = [66, 66, 72]
WIDE_W = [66, 60, 60]


def row(x: float, y: float, cells: list[str], widths: list[int], hot: tuple[int, ...]) -> str:
    """One table row; cells whose index is in `hot` are drawn in the accent."""
    out, left = "", x
    for i, (text, w) in enumerate(zip(cells, widths, strict=True)):
        tint = AMBER if i in hot else BLUE_MID
        weight = "800" if i in hot else "600"
        out += (
            f'<rect x="{left}" y="{y}" width="{w}" height="{CELL_H}" fill="{tint}" '
            f'fill-opacity="{0.28 if i in hot else 0.12}" stroke="{tint}" stroke-opacity="0.7" '
            f'stroke-width="1"/>'
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


LONG = (
    ("west", "q1", "10.0"),
    ("west", "q2", "20.0"),
    ("east", "q1", "30.0"),
    ("east", "q2", "40.0"),
    ("west", "q1", "5.0"),
)
WIDE = (("east", "30.0", "40.0"), ("west", "15.0", "20.0"))
BACK = (
    ("east", "q1", "30.0"),
    ("east", "q2", "40.0"),
    ("west", "q1", "15.0"),
    ("west", "q2", "20.0"),
)

LX, WX, BX = 44, 386, 732
TOP = 140

body = [
    band(20, 20, 940, 290, "LONG TO WIDE AND BACK", "blue"),
    heading(LX + 102, 74, "LONG", anchor="middle"),
    note(LX + 102, 94, "one row per reading", anchor="middle"),
    header(LX, 126, ["region", "quarter", "amount"], LONG_W),
    heading(WX + 93, 74, "WIDE", anchor="middle"),
    note(WX + 93, 94, "one row per region", anchor="middle"),
    header(WX, TOP + 16, ["region", "q1", "q2"], WIDE_W),
    heading(BX + 102, 74, "LONG AGAIN", anchor="middle"),
    note(BX + 102, 94, "one row per region and quarter", anchor="middle"),
    header(BX, TOP + 1, ["region", "quarter", "amount"], LONG_W),
]
for i, cells in enumerate(LONG):
    hot = (0, 1, 2) if cells[0] == "west" and cells[1] == "q1" else ()
    body.append(row(LX, TOP + i * CELL_H, list(cells), LONG_W, hot))
for i, cells in enumerate(WIDE):
    body.append(row(WX, TOP + 30 + i * CELL_H, list(cells), WIDE_W, (1,) if i == 1 else ()))
for i, cells in enumerate(BACK):
    hot = (0, 1, 2) if cells[0] == "west" and cells[1] == "q1" else ()
    body.append(row(BX, TOP + 15 + i * CELL_H, list(cells), LONG_W, hot))

MID = TOP + 75
body += [
    arrow(262, MID, 372, MID),
    label(317, MID - 12, "pivot", anchor="middle", size=12.5),
    note(317, MID + 22, "sums each cell", anchor="middle"),
    arrow(588, MID, 718, MID),
    label(653, MID - 12, "unpivot", anchor="middle", size=12.5),
    note(653, MID + 22, "columns to rows", anchor="middle"),
    note(WX + 93, TOP + 118, "west q1: 10.0 + 5.0 = 15.0", anchor="middle"),
    note(
        490, 290, 'on="quarter" names the columns, and aggregate="sum" fills them.', anchor="middle"
    ),
    # The asymmetry the page's comparison table spells out.
    band(20, 328, 460, 104, "PIVOT", "amber"),
    note(44, 374, "Output columns come from the data, so a"),
    note(44, 392, "pre-pass reads quarter unless you pass columns=."),
    note(44, 410, "It groups, so it is a pipeline breaker."),
    band(500, 328, 460, 104, "UNPIVOT", "grey"),
    note(524, 374, "The schema is fixed by the arguments."),
    note(524, 392, "No pre-pass and no breaker: it streams"),
    note(524, 410, "and distributes like a select."),
]

write("pivot_long_wide", svg(W, H, "".join(body)))
print("wrote pivot_long_wide.svg")
