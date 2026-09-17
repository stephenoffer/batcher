#!/usr/bin/env python3
"""Draw `explain_line_anatomy.svg`: one operator line of `explain(analyze=True)`, annotated.

Source of truth: `docs/user-guide/operate/tuning/explain-plans.md`. The line is copied from
the page's own measured output for the example query (the `aggregate` row), split into its
columns so each can carry a callout. The column meanings come from the page's column table:
`est≈N` is the rows Kyber planned for, `actual=N` the rows the operator produced, `MISS`
reads `exact` or `Nx over` / `Nx under` (over means the plan expected more rows than
arrived), `OP SHARE` is a share of total operator time rather than of the wall clock, and
`NOTES` carries the backend and conditional clauses such as `spill`.

The worked miss in the bottom band is the page's own example: a `100.0x under` on a join
input means the optimizer planned for 100 rows and got 10,000, and the fix is upstream.

Form: the estimate, actual and miss cells are grouped under one amber bracket, because
reading them together is the point of the page's "estimate vs actual" section.
"""

from __future__ import annotations

from _authoring import (
    AMBER_DEEP,
    BLUE,
    FONT,
    GREY,
    MONO,
    band,
    note,
    svg,
    write,
)

W, H = 980, 520

CHAR = 8.4  # monospace advance at 14 px
PAD = 12
GAP = 8
ROW_Y, ROW_H = 100, 42
MIN_CELL = 76

CELLS = (
    ("OPERATOR", "▶ └─ aggregate  [by region · sum]", "blue"),
    ("ESTIMATE", "est≈2", "amber"),
    ("ACTUAL", "actual=2", "amber"),
    ("MISS", "exact", "amber"),
    ("TIME", "316µs", "blue"),
    ("OP SHARE", "65%", "blue"),
    ("NOTES", "interp", "blue"),
)


def cell(x: float, w: float, text: str, kind: str) -> str:
    """One column of the operator line, as a code chip."""
    stroke = AMBER_DEEP if kind == "amber" else "#cbd5e1"
    sw = "1.8" if kind == "amber" else "1.1"
    return (
        f'<rect x="{x}" y="{ROW_Y}" width="{w}" height="{ROW_H}" rx="7" class="code-bg" '
        f'style="stroke:{stroke};stroke-width:{sw}"/>'
        f'<text x="{x + w / 2}" y="{ROW_Y + 26}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="14" class="t-code" xml:space="preserve">{text}</text>'
    )


def leader(x1: float, y1: float, x2: float, y2: float, color: str) -> str:
    """A thin elbow line from a cell down to its callout, ending in a dot."""
    mid = (y1 + y2) / 2
    return (
        f'<path d="M {x1} {y1} L {x1} {mid} L {x2} {mid} L {x2} {y2}" fill="none" '
        f'stroke="{color}" stroke-width="1.4"/>'
        f'<circle cx="{x1}" cy="{y1}" r="3" fill="{color}"/>'
    )


def callout(x: float, y: float, w: float, title: str, lines: list[str], kind: str) -> str:
    """A white card holding a short title and a few lines of explanation."""
    h = 40 + 18 * len(lines)
    stroke = AMBER_DEEP if kind == "amber" else "#cbd5e1"
    bar = AMBER_DEEP if kind == "amber" else BLUE
    out = (
        f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'class="surface" stroke="{stroke}" stroke-width="1.2"/></g>'
        f'<rect x="{x}" y="{y + 12}" width="4" height="{h - 24}" rx="2" fill="{bar}"/>'
        f'<text x="{x + 18}" y="{y + 26}" font-family="{FONT}" font-size="13.5" '
        f'font-weight="700" class="t-title">{title}</text>'
    )
    for i, line in enumerate(lines):
        out += (
            f'<text x="{x + 18}" y="{y + 48 + i * 18}" font-family="{FONT}" font-size="11.5" '
            f'class="t-sub">{line}</text>'
        )
    return out


body: list[str] = [
    band(20, 20, 940, 190, "ONE OPERATOR LINE FROM EXPLAIN(ANALYZE=TRUE)", "grey"),
]

# Lay the cells out left to right, centred as a group.
widths = [max(MIN_CELL, len(text) * CHAR + 2 * PAD) for _, text, _ in CELLS]
total = sum(widths) + GAP * (len(CELLS) - 1)
x = (W - total) / 2
centres: list[float] = []
edges: list[tuple[float, float]] = []
for (head, text, kind), w in zip(CELLS, widths, strict=True):
    body.append(
        f'<text x="{x + w / 2}" y="{ROW_Y - 12}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="10.5" font-weight="800" letter-spacing="0.6" class="t-sub">{head}</text>'
    )
    body.append(cell(x, w, text, kind))
    centres.append(x + w / 2)
    edges.append((x, x + w))
    x += w + GAP

# The bracket under the three cells that are read together.
bx1, bx2 = edges[1][0], edges[3][1]
by = ROW_Y + ROW_H + 12
body.append(
    f'<path d="M {bx1} {by - 6} L {bx1} {by} L {bx2} {by} L {bx2} {by - 6}" fill="none" '
    f'stroke="{AMBER_DEEP}" stroke-width="2"/>'
)
body.append(
    f'<text x="{(bx1 + bx2) / 2}" y="{by + 18}" text-anchor="middle" font-family="{FONT}" '
    f'font-size="12" font-weight="800" letter-spacing="1.2" fill="{AMBER_DEEP}">'
    f"ESTIMATE VS ACTUAL</text>"
)

CALL_Y = 240
body += [
    leader(centres[0], ROW_Y + ROW_H, 142, CALL_Y, GREY),
    callout(
        36,
        CALL_Y,
        212,
        "the optimized plan",
        ["▶ marks the critical path.", "The bracket says what the", "operator does."],
        "blue",
    ),
    leader((bx1 + bx2) / 2, by + 26, 390, CALL_Y, AMBER_DEEP),
    callout(
        264,
        CALL_Y,
        252,
        "rows planned vs produced",
        [
            "est≈ is what Kyber planned for.",
            "actual= is what arrived.",
            "MISS says how far, which way.",
        ],
        "amber",
    ),
    f'<path d="M {edges[4][0]} {by - 6} L {edges[4][0]} {by} L {edges[5][1]} {by} '
    f'L {edges[5][1]} {by - 6}" fill="none" stroke="{GREY}" stroke-width="2"/>',
    leader((edges[4][0] + edges[5][1]) / 2, by, 642, CALL_Y, GREY),
    callout(
        532,
        CALL_Y,
        220,
        "where time went",
        ["TIME: the operator's wall time.", "OP SHARE ranks it against", "all operator time."],
        "blue",
    ),
    leader(centres[6], ROW_Y + ROW_H, centres[6], CALL_Y, GREY),
    callout(
        768,
        CALL_Y,
        176,
        "conditional notes",
        ["interp or jit, broadcast,", "spill, pushed[...]. Each", "only when it applies."],
        "blue",
    ),
]

# ---- How to read a miss --------------------------------------------------------------------
body += [
    band(20, 372, 940, 128, "READING THE MISS COLUMN", "amber"),
    f'<text x="44" y="424" font-family="{MONO}" font-size="14" class="t-code">exact</text>',
    note(160, 424, "The estimate matched what the operator produced."),
    f'<text x="44" y="452" font-family="{MONO}" font-size="14" class="t-code">10.0x over</text>',
    note(160, 452, "The plan expected ten times the rows that arrived."),
    f'<text x="44" y="480" font-family="{MONO}" font-size="14" class="t-code">100.0x under</text>',
    note(
        160,
        480,
        "Planned for 100 rows, got 10,000. Fix the statistics upstream, not the join.",
    ),
]

write("explain_line_anatomy", svg(W, H, "".join(body)))
print("wrote explain_line_anatomy.svg")
