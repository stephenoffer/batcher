#!/usr/bin/env python3
"""Draw `example_library_map.svg`, the runnable example scripts counted by section.

Source of truth: the generated library tables in `docs/examples/*.md` (one row per script,
written by `tools/example_library.py`), and the 16 tour scripts at the root of `examples/`.
The counts here are those row counts. Re-run this script after the library is regenerated,
because a count that drifts from the table is exactly what the table exists to prevent.

Drawn as a sorted bar chart with a zero baseline and a direct label on every bar, so the
reader can compare sections without a legend or an axis to decode.
"""

from __future__ import annotations

from _authoring import BLUE_MID, FONT, heading, note, svg, write

SECTIONS = [
    ("Relational operations", 115),
    ("Expressions", 101),
    ("Machine learning", 57),
    ("Reading and writing", 47),
    ("Statistics, time series, geo, graph", 46),
    ("Operating the engine", 40),
    ("TPC-H", 30),
    ("Data quality and governance", 23),
    ("Distributed and streaming", 18),
    ("Root tour scripts", 16),
    ("Multimodal and text", 11),
    ("Accelerators", 8),
]

W = 960
LEFT = 290  # where the bars start; the section names sit right-aligned before it
BAR_MAX = 560
ROW = 34
TOP = 78
H = TOP + ROW * len(SECTIONS) + 52
PEAK = max(n for _, n in SECTIONS)

body = [
    heading(40, 40, "512 RUNNABLE SCRIPTS, BY SECTION"),
    note(40, 60, "Each script runs end to end and asserts on its own output."),
]
for i, (name, n) in enumerate(SECTIONS):
    y = TOP + i * ROW
    width = BAR_MAX * n / PEAK
    body.append(
        f'<text x="{LEFT - 14}" y="{y + 17}" text-anchor="end" font-family="{FONT}" '
        f'font-size="13" font-weight="600" class="t-title">{name}</text>'
    )
    body.append(
        f'<rect x="{LEFT}" y="{y + 4}" width="{width:.1f}" height="20" rx="4" '
        f'fill="{BLUE_MID}" fill-opacity="{0.95 if i < 2 else 0.8}"/>'
    )
    body.append(
        f'<text x="{LEFT + width + 8:.1f}" y="{y + 18}" font-family="{FONT}" font-size="12.5" '
        f'font-weight="700" class="t-arrow">{n}</text>'
    )
body.append(
    f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP + ROW * len(SECTIONS)}" '
    f'stroke="#94a3b8" stroke-width="1.2"/>'
)
body.append(
    note(
        LEFT,
        TOP + ROW * len(SECTIONS) + 30,
        "Bars start at zero. Counts are rows in each page's generated table.",
    )
)

write("example_library_map", svg(W, H, "".join(body)))
print("wrote example_library_map.svg")
