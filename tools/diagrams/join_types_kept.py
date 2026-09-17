#!/usr/bin/env python3
"""Draw `join_types_kept.svg` -- which rows each join type keeps, and what an as-of join matches.

Source of truth: `docs/user-guide/analyze/joins.md`. The `how` values are the page's list
(`inner`, `left`, `right`, `full`/`outer`, `semi`, `anti`); a left join fills the right
columns with null where there is no match, right and full are the mirror and the union; semi
and anti filter the left side by existence and add no right columns. The as-of panel draws
the page's own `trades`/`quotes` example: `join_asof(quotes, on="t", by="sym")` matches each
trade to the last quote at or before it, and `tolerance=5` leaves the `B` trade at `t=10`
unmatched because its only quote is from `t=1`.

The key sets in the matrix (left 1, 2, 3; right 2, 3, 4) are illustrative, chosen so that
every join type has a key only on the left, keys on both sides, and a key only on the right.
"""

from __future__ import annotations

from _authoring import (
    AMBER_DEEP,
    BLUE,
    FONT,
    GREY,
    MONO,
    band,
    heading,
    label,
    mark,
    note,
    pill,
    svg,
    write,
)

W, H = 980, 744

HOW_X = 60
COLS = (330, 500, 670)
OUT_X = 858
ROW0, ROW_H = 204, 40

# (how, key 1 kept, keys 2/3 kept, key 4 kept, null note for key 1, null note for key 4, output)
ROWS = (
    ("inner", False, True, False, "", "", "left + right"),
    ("left", True, True, False, "right is null", "", "left + right"),
    ("right", False, True, True, "", "left is null", "left + right"),
    ("full", True, True, True, "right is null", "left is null", "left + right"),
    ("semi", False, True, False, "", "", "left only"),
    ("anti", True, False, False, "", "", "left only"),
)


def mono(x: float, y: float, text: str, size: float = 13, anchor: str = "start") -> str:
    """A code-styled word, for an argument value."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" font-size="{size}" '
        f'font-weight="700" class="t-code">{text}</text>'
    )


def chip(x: float, y: float, text: str, kind: str) -> str:
    """One key of an input, drawn as a small square."""
    stroke = {"blue": BLUE, "amber": AMBER_DEEP}[kind]
    return (
        f'<rect x="{x - 15}" y="{y - 15}" width="30" height="30" rx="6" class="surface" '
        f'style="stroke: {stroke}" stroke-width="1.8"/>'
        f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-family="{FONT}" font-size="13.5" '
        f'font-weight="700" class="t-title">{text}</text>'
    )


body = [
    band(20, 20, 940, 438, "WHICH ROWS A JOIN KEEPS", "blue"),
    # The two inputs.
    label(60, 76, "left keys"),
    chip(150, 71, "1", "blue"),
    chip(190, 71, "2", "blue"),
    chip(230, 71, "3", "blue"),
    label(330, 76, "right keys"),
    chip(428, 71, "2", "amber"),
    chip(468, 71, "3", "amber"),
    chip(508, 71, "4", "amber"),
    note(560, 76, "Each output row is one left row, one right row, or a matched pair."),
    # Column heads.
    heading(HOW_X, 136, "HOW="),
    heading(COLS[0], 136, "KEY 1", anchor="middle"),
    note(COLS[0], 156, "left only", anchor="middle"),
    heading(COLS[1], 136, "KEYS 2, 3", anchor="middle"),
    note(COLS[1], 156, "on both sides", anchor="middle"),
    heading(COLS[2], 136, "KEY 4", anchor="middle"),
    note(COLS[2], 156, "right only", anchor="middle"),
    heading(OUT_X, 136, "COLUMNS", anchor="middle"),
    note(OUT_X, 156, "in the output", anchor="middle"),
    f'<path d="M 44 176 H 936" stroke="{GREY}" stroke-opacity="0.5" stroke-width="1.2"/>',
]

for i, (how, k1, k23, k4, n1, n4, out) in enumerate(ROWS):
    y = ROW0 + i * ROW_H
    shift1 = -44 if n1 else 0
    shift4 = -44 if n4 else 0
    body += [
        mono(HOW_X, y + 5, f'"{how}"'),
        mark(COLS[0] + shift1, y, k1),
        mark(COLS[1], y, k23),
        mark(COLS[2] + shift4, y, k4),
        pill(OUT_X, y + 4, out, "blue" if out == "left + right" else "grey", anchor="middle"),
    ]
    if n1:
        body.append(note(COLS[0] - 26, y + 4, n1))
    if n4:
        body.append(note(COLS[2] - 26, y + 4, n4))

body.append(
    note(
        490,
        ROW0 + 6 * ROW_H + 4,
        '"outer" is the same as "full". Semi and anti filter the left rows by existence.',
        anchor="middle",
    )
)

# --- the as-of panel -------------------------------------------------------------------
T0, PX = 150, 16.5  # screen x of t=0, pixels per unit of t
TRACK_A, TRACK_B = 566, 648


def tx(t: float) -> float:
    """Screen x for a value of the `on` key."""
    return T0 + t * PX


def track(y: float, sym: str) -> str:
    """One `by` group's axis."""
    return f'<path d="M {tx(0)} {y} H {tx(46)}" stroke="{GREY}" stroke-width="1.4"/>' + label(
        60, y + 5, f"sym {sym}"
    )


def quote(t: float, y: float, price: str) -> str:
    """A right row, drawn as a square above the axis."""
    x = tx(t)
    return (
        f'<rect x="{x - 7}" y="{y - 7}" width="14" height="14" rx="2" fill="{AMBER_DEEP}"/>'
        + note(x, y - 34, f"quote t={t}, {price}", anchor="middle")
    )


def trade(t: float, y: float) -> str:
    """A left row, drawn as a circle below the axis."""
    x = tx(t)
    return f'<circle cx="{x}" cy="{y}" r="7" fill="{BLUE}"/>' + note(
        x, y + 26, f"trade t={t}", anchor="middle"
    )


def link(t_from: float, t_to: float, y: float, matched: bool) -> str:
    """The match an as-of join makes, looking backward from the trade."""
    color, dash = (AMBER_DEEP, "") if matched else (GREY, ' stroke-dasharray="4 3"')
    x1, x2 = tx(t_from) - 3, tx(t_to) + 3
    return (
        f'<path d="M {x1} {y - 9} Q {(x1 + x2) / 2} {y - 36} {x2} {y - 9}" fill="none" '
        f'stroke="{color}" stroke-width="2.6"{dash}/>'
    )


body += [
    band(
        20, 474, 940, 246, "JOIN_ASOF: EVERY LEFT ROW, THE LAST RIGHT ROW AT OR BEFORE IT", "amber"
    ),
    track(TRACK_A, "A"),
    quote(8, TRACK_A, "1.0"),
    quote(38, TRACK_A, "1.1"),
    trade(10, TRACK_A),
    trade(40, TRACK_A),
    link(10, 8, TRACK_A, matched=True),
    link(40, 38, TRACK_A, matched=True),
    label(tx(12), TRACK_A - 12, "price 1.0", size=12),
    label(tx(36), TRACK_A - 12, "price 1.1", anchor="end", size=12),
    track(TRACK_B, "B"),
    quote(1, TRACK_B, "9.0"),
    trade(10, TRACK_B),
    link(10, 1, TRACK_B, matched=False),
    label(tx(14.5), TRACK_B + 26, "9 units back: price 9.0, or null with tolerance=5", size=12),
    note(
        490,
        706,
        "by=\"sym\" keeps one symbol's quotes away from another's trades.",
        anchor="middle",
    ),
]

write("join_types_kept", svg(W, H, "".join(body)))
print("wrote join_types_kept.svg")
