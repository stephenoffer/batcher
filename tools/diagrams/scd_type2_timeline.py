#!/usr/bin/env python3
"""Draw `scd_type2_timeline.svg` -- one key's version history, before and after a load.

Source of truth: `python/batcher/api/dataset/scd.py::DatasetSCD.type2`. The rows drawn
here are the ones its own doctest produces: ``id=1, city='NYC'`` loaded ``as_of
'2024-01-01'``, then ``city='LA'`` loaded ``as_of '2024-06-01'``, after which
``valid_from`` reads ``['2024-01-01', '2024-06-01']`` and ``is_current`` reads
``[False, True]``. The method's body is what the picture states: the superseded current
row is expired with ``valid_to = as_of`` and ``is_current = False`` while keeping its
original ``valid_from``, and a new version is appended with ``valid_from = as_of``,
``valid_to`` NULL and ``is_current = True``.

The figure earns its place because type 2 is a *history*, and the thing readers get wrong
is which of the four columns moves. Two bars on a date axis show that the closed one keeps
its start and gains an end, and that the two meet at exactly the load date with no gap and
no overlap. A column-by-column table asserts that; the axis lets the reader see it.

Two facts on the page that the picture cannot carry, and which the prose beside it should:
a key whose tracked attributes did not change is not touched at all, and a key the target
has never seen is inserted as a first open version. Both are in the same method.
"""

from __future__ import annotations

from _authoring import AMBER, AMBER_DEEP, BLUE_MID, FONT, GREY, arrow, band, label, note, svg, write

W, H = 980, 616

T0, T1, T_END = 150, 520, 780  # 2024-01-01, 2024-06-01, and the right edge of "now"


def bar(x0: float, x1: float, y: float, h: float, color: str, lines: tuple[str, ...]) -> str:
    """One version of the row, drawn as the interval it is valid over."""
    out = (
        f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{h}" rx="7" fill="{color}" '
        f'fill-opacity="0.16" stroke="{color}" stroke-width="1.8"/>'
    )
    weights = ("700", "400", "400")
    sizes = (12.5, 11, 11)
    cls = ("t-title", "t-sub", "t-sub")
    for i, text in enumerate(lines):
        out += (
            f'<text x="{x0 + 14}" y="{y + 22 + 17 * i}" font-family="{FONT}" '
            f'font-size="{sizes[i]}" font-weight="{weights[i]}" class="{cls[i]}">{text}</text>'
        )
    return out


body: list[str] = [
    note(
        490,
        50,
        "One natural key, id = 1, tracked on the city column. Each bar is a row of the dimension table and the interval it is valid over.",
        anchor="middle",
    ),
    # Before.
    band(20, 74, 940, 128, "THE TABLE BEFORE THE 2024-06-01 LOAD", "grey"),
    bar(
        T0,
        T_END,
        114,
        70,
        BLUE_MID,
        (
            "city = NYC",
            "valid_from = 2024-01-01 · valid_to = NULL",
            "is_current = true",
        ),
    ),
    arrow(T_END, 149, T_END + 34, 149, "grey"),
    label(T_END + 42, 153, "open"),
    # The load itself, carried on the arrow between the two states.
    arrow(80, 210, 80, 252, "amber"),
    label(96, 228, "ds.scd.type2(as_of='2024-06-01') with city = LA for id = 1"),
    # After.
    band(20, 258, 940, 214, "THE SAME TABLE AFTER IT", "blue"),
    bar(
        T0,
        T1,
        300,
        70,
        AMBER,
        (
            "city = NYC",
            "valid_from = 2024-01-01 (unchanged)",
            "valid_to = 2024-06-01 · is_current = false",
        ),
    ),
    bar(
        T1,
        T_END,
        386,
        70,
        BLUE_MID,
        (
            "city = LA",
            "valid_from = 2024-06-01",
            "valid_to = NULL · is_current = true",
        ),
    ),
    arrow(T_END, 421, T_END + 34, 421, "grey"),
    label(T_END + 42, 425, "open"),
    label(40, 334, "expired"),
    label(40, 420, "appended"),
    # The date axis, shared by both bars, drawn once beneath them.
    f'<path d="M {T0} 452 H 860" stroke="{GREY}" stroke-width="1.6" marker-end="url(#arG)"/>',
    f'<path d="M {T0} 448 V 456 M {T1} 448 V 456 M {T_END} 448 V 456" stroke="{GREY}" stroke-width="1.4"/>',
    note(T0, 470, "2024-01-01", anchor="middle"),
    note(T1, 470, "2024-06-01", anchor="middle"),
    note(T_END, 470, "now", anchor="middle"),
    # The load date, marked through the after-band so the join between the two bars is visible.
    f'<path d="M {T1} 284 V 300 M {T1} 370 V 386" stroke="{AMBER_DEEP}" stroke-width="2" stroke-dasharray="5 4"/>',
    label(T1 + 10, 296, "as_of"),
    band(20, 492, 940, 108, "WHAT THE LOAD DID", "grey"),
    note(
        40,
        532,
        "The previous version keeps its valid_from. Only valid_to and is_current change on it, and the new version starts exactly where the old one ends.",
        anchor="start",
    ),
    note(
        40,
        554,
        "A key whose tracked columns did not change is not touched at all, and a key the target has never seen is inserted as a first open version.",
        anchor="start",
    ),
    note(
        40,
        578,
        "No new operator is involved: history, untouched current rows, expired rows and new versions are unioned and written back over the target.",
        anchor="start",
    ),
]

write("scd_type2_timeline", svg(W, H, "".join(body)))
print("wrote scd_type2_timeline.svg")
