#!/usr/bin/env python3
"""Draw `type_widening.svg` -- narrow numeric types widen at the FFI edge, and what comes back out.

Source of truth: `crates/bc-py/src/normalize.rs` -- `widen_to` maps Int8/16/32 and
UInt8/16/32/64 to Int64 and Float16/32 to Float64, and `normalize_to` decodes a Dictionary
column to its value type -- and `docs/user-guide/transform/columns/type-system.md`, where the
figure is embedded ("Narrow numerics widen at the boundary", "Get narrow types back on
output"). `ExecutionConfig.shrink_output_dtypes` (`python/batcher/config/config.py`) is off by
default so output types match `Dataset.schema`; when on, a pass-through of a narrow source
column is cast back to its source width where lossless, a derived column is not, and a bare
scan with no operations skips it.

Layout: three columns left to right -- the source, the engine, the result -- with the FFI
edge drawn as a dashed line the rows cross.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, arrow, band, card, heading, label, note, svg, write

W, H = 980, 478
TOP = 44
ROWS = (
    ("Int8, Int16, Int32", "Int64", "widen"),
    ("UInt8 to UInt64", "Int64", "widen"),
    ("Float16, Float32", "Float64", "widen"),
    ("dictionary-encoded", "its value type", "decode"),
)

body: list[str] = [
    band(20, TOP, 250, 320, "IN THE SOURCE", "grey"),
    band(410, TOP, 230, 320, "IN THE ENGINE", "blue"),
    band(700, TOP, 260, 320, "WHAT COMES BACK", "amber"),
    f'<path d="M 340 {TOP - 10} L 340 {TOP + 330}" stroke="{AMBER_DEEP}" stroke-width="2" '
    'stroke-dasharray="5 5" fill="none"/>',
    heading(340, TOP - 18, "FFI EDGE", anchor="middle", kind="amber"),
]
for i, (src, dst, verb) in enumerate(ROWS):
    y = TOP + 80 + i * 56
    body += [
        label(44, y + 4, src, size=12.5),
        arrow(210, y, 432, y),
        label(386, y - 9, verb, anchor="middle", size=11),
        label(444, y + 4, dst, size=12.5),
    ]


def result(y: float, title: str, lines: tuple[str, str]) -> list[str]:
    """An output card: a bold title and two lines of explanation."""
    return [
        card(716, y, 228, 96, ""),
        label(830, y + 32, title, anchor="middle", size=13),
        note(830, y + 56, lines[0], anchor="middle"),
        note(830, y + 74, lines[1], anchor="middle"),
    ]


body += [
    note(525, TOP + 294, "Dataset.schema already reports", anchor="middle"),
    note(525, TOP + 310, "these before anything runs", anchor="middle"),
    *result(TOP + 52, "By default", ("the widened type, exactly", "what schema promised")),
    *result(
        TOP + 196,
        "shrink_output_dtypes=True",
        ("a pass-through column narrows", "back to its source width"),
    ),
    arrow(646, TOP + 150, 710, TOP + 104, "amber"),
    label(664, TOP + 108, "default", anchor="middle", size=11),
    arrow(646, TOP + 170, 710, TOP + 240, "amber"),
    label(662, TOP + 226, "opt in", anchor="end", size=11),
    # ---- the caveats ------------------------------------------------------------------------
    note(
        490,
        406,
        "A UInt64 value above the Int64 maximum raises at the edge rather than wrapping.",
        anchor="middle",
    ),
    note(
        490,
        428,
        "Only a lossless pass-through narrows. A derived column stays wide, so cast it explicitly.",
        anchor="middle",
    ),
    note(
        490,
        450,
        "A cast inside a query does produce the narrow type. A bare scan skips the re-narrowing.",
        anchor="middle",
    ),
]

write("type_widening", svg(W, H, "".join(body)))
print("wrote type_widening.svg")
