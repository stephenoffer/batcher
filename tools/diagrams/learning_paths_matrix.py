#!/usr/bin/env python3
"""Draw `learning_paths_matrix.svg`: which topics each role's learning path covers.

Source of truth: the "Reading order" lists in
`docs/getting-started/tutorials/paths/data-engineer.md`, `data-scientist.md`,
`ml-engineer.md` and `platform-engineer.md`. A check means that path's reading order
links a page on the topic; a cross means it does not. Every path opens with Getting
started and closes with an API reference, so those two are stated once in the footnote
rather than drawn as four checks each.

Keep the rows in step with those four pages: a path that gains a page changes a mark.
"""

from __future__ import annotations

from _authoring import FONT, heading, mark, note, svg, write

W, H = 980, 620

ROLES = [("Data", "engineer"), ("Data", "scientist"), ("ML", "engineer"), ("Platform", "engineer")]
COLS = (520, 640, 760, 880)

ROWS = [
    ("Your first pipeline tutorial", (1, 0, 1, 0)),
    ("Core concepts", (0, 1, 0, 0)),
    ("Reading and writing data", (1, 0, 0, 0)),
    ("Expressions and filtering", (1, 1, 0, 0)),
    ("Aggregations and window functions", (1, 1, 0, 0)),
    ("Joins", (1, 0, 0, 0)),
    ("SQL", (0, 1, 0, 0)),
    ("Lakehouse tables and data quality", (1, 0, 0, 0)),
    ("Inference, features, and GPUs", (0, 0, 1, 0)),
    ("Installation and configuration", (0, 0, 0, 1)),
    ("Cloud storage", (1, 0, 0, 1)),
    ("Best practices and troubleshooting", (1, 0, 0, 1)),
]

TOP, ROW_H = 104, 38


def text(x: float, y: float, s: str, size: float, weight: int, cls: str, anchor: str) -> str:
    """A plain text run in the shared font, themed through a style class."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" class="{cls}">{s}</text>'
    )


body = [heading(40, 40, "WHAT EACH PATH COVERS", kind="grey")]
for x, (first, second) in zip(COLS, ROLES, strict=True):
    body.append(text(x, 74, first, 13.5, 700, "t-title", "middle"))
    body.append(text(x, 91, second, 13.5, 700, "t-title", "middle"))

for i, (topic, marks) in enumerate(ROWS):
    y = TOP + i * ROW_H
    if i % 2 == 0:
        body.append(
            f'<rect x="24" y="{y}" width="932" height="{ROW_H}" rx="8" class="band-grey" '
            'stroke-width="0"/>'
        )
    body.append(text(40, y + ROW_H / 2 + 5, topic, 13.5, 600, "t-title", "start"))
    for x, on in zip(COLS, marks, strict=True):
        body.append(mark(x, y + ROW_H / 2, bool(on)))

body.append(
    note(
        40,
        TOP + len(ROWS) * ROW_H + 34,
        "Every path starts at Getting started and ends at an API reference.",
    )
)

write("learning_paths_matrix", svg(W, H, "".join(body)))
print("wrote learning_paths_matrix.svg")
