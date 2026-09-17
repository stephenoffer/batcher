#!/usr/bin/env python3
"""Draw `reader_choice.svg` -- which constructor or reader to start a pipeline with.

Source of truth: `docs/user-guide/moving-data/reading-data.md`, which is where the figure
is embedded. The in-memory constructors (`from_pydict`, `from_arrow`, `from_numpy`,
`from_items`, `from_batches`, and the framework adapters such as `from_pandas` and
`from_polars`) and the path readers (`bt.read` with format detection from the extension or
from the files inside a directory, the typed `read.parquet`/`read.csv`, `read.delta`,
`read.iceberg`, `read.images`, `read.video`, `read.sql`, `read.snowflake`) are all named on
that page. A directory holding two data formats declining detection and asking for
`format=` is stated there too.

Layout: one question at the top, then the two answers as two columns of input-to-reader
rows. Rows rather than boxes, because twelve boxes would be a wall and the choice is a
lookup once the first branch is taken.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, arrow, band, card, label, note, svg, write

W, H = 1040, 550
BAND_W = 490
BAND_Y = 132
ROW0 = BAND_Y + 76
STEP = 44

LEFT = [
    ("a column dict", "from_pydict"),
    ("an Arrow table or batches", "from_arrow"),
    ("a NumPy array", "from_numpy"),
    ("a pandas or Polars frame", "from_pandas, from_polars"),
    ("a list of Python items", "from_items"),
    ("a factory yielding batches", "from_batches"),
]
RIGHT = [
    ("the extension names it", "bt.read(path)"),
    ("a directory of one format", "bt.read(directory)"),
    ("you name the format", "read.parquet, read.csv"),
    ("a Delta or Iceberg table", "read.delta, read.iceberg"),
    ("images, audio or video", "read.images, read.video"),
    ("a database or warehouse", "read.sql, read.snowflake"),
]


def tag(x: float, y: float, text: str, kind: str) -> str:
    """A code pill sized with room to spare, so a wider fallback font stays inside it."""
    w = 18 + 7.4 * len(text)
    return (
        f'<rect x="{x}" y="{y - 13}" width="{w}" height="21" rx="10" class="pill-{kind}"/>'
        f'<text x="{x + w / 2}" y="{y + 1.5}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="11" font-weight="700" class="pt-{kind}">{text}</text>'
    )


def elbow(x1: float, y1: float, x2: float, y2: float, kind: str) -> str:
    """A right-angled connector: across, then down, with one arrowhead at the end."""
    stroke, marker = (BLUE, "arB") if kind == "blue" else (AMBER_DEEP, "arA")
    return (
        f'<path d="M {x1} {y1} L {x2 - 10 if x2 > x1 else x2 + 10} {y1} Q {x2} {y1} {x2} {y1 + 10} '
        f'L {x2} {y2}" fill="none" stroke="{stroke}" stroke-width="2.4" '
        f'marker-end="url(#{marker})"/>'
    )


def rows(x0: float, entries: list[tuple[str, str]], kind: str) -> list[str]:
    """Input description, a short labeled-by-position arrow, then the reader as a pill."""
    out: list[str] = []
    for i, (what, reader) in enumerate(entries):
        y = ROW0 + i * STEP
        out.append(label(x0 + 24, y + 1, what, size=12.5))
        out.append(arrow(x0 + 238, y - 3, x0 + 268, y - 3, kind))
        out.append(tag(x0 + 280, y, reader, kind))
    return out


body: list[str] = [
    card(360, 22, 320, 64, "Where is the data now?", "every answer is a lazy Dataset"),
    elbow(356, 54, 265, BAND_Y - 6, "blue"),
    label(310, 42, "in Python", anchor="middle", size=11.5),
    elbow(684, 54, 775, BAND_Y - 6, "amber"),
    label(730, 42, "at a path", anchor="middle", size=11.5),
    band(20, BAND_Y, BAND_W, 356, "IN MEMORY: bt.from_*", "blue"),
    band(530, BAND_Y, BAND_W, 356, "IN FILES OR A BUCKET: bt.read", "amber"),
    note(44, BAND_Y + 48, "you hold ...", anchor="start"),
    note(300, BAND_Y + 48, "start with", anchor="start"),
    note(554, BAND_Y + 48, "the input is ...", anchor="start"),
    note(810, BAND_Y + 48, "start with", anchor="start"),
]
body += rows(20, LEFT, "blue")
body += rows(530, RIGHT, "amber")
body += [
    note(265, BAND_Y + 336, "No files and no credentials needed.", anchor="middle"),
    note(775, BAND_Y + 336, "Two formats in one directory: pass format=.", anchor="middle"),
    note(
        520,
        524,
        "Both columns are lazy. Nothing is read until a terminal operation such as collect runs.",
        anchor="middle",
    ),
]

write("reader_choice", svg(W, H, "".join(body)))
print("wrote reader_choice.svg")
