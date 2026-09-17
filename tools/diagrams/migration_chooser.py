#!/usr/bin/env python3
"""Draw `migration_chooser.svg`: which migration page to read first, by source system.

Source of truth: `docs/getting-started/migration/index.md`, the "Coming from" cards. Each
card names the one shift that matters most and links a first page: pandas, Polars and
PySpark to "Transforming and collecting", DuckDB and SQL to the SQL guide
(`/user-guide/analyze/sql`), Daft to "Batch inference and ML", and Ray Data to the Ray
Data page. "Differences and verification" shows how to prove the port matches, and the
generated name-by-name references cover PySpark, Polars, Daft and Ray Data.

Layout: sources across the top, their first page below each, and one verification step
every path ends in.
"""

from __future__ import annotations

from _authoring import arrow, card, heading, hero, label, note, svg, tint, write

W, H = 980, 520

CW, GAP, X0 = 146, 12, 22  # source column width, gutter, left edge
SRC_Y, DST_Y, END_Y = 64, 220, 380


def col_x(i: int) -> float:
    """Left edge of source column `i`."""
    return X0 + i * (CW + GAP)


def mid(i: int) -> float:
    """Centre line of source column `i`."""
    return col_x(i) + CW / 2


SOURCES = [
    ("pandas", "eager to lazy"),
    ("Polars", "LazyFrame ports"),
    ("PySpark", "no SparkSession"),
    ("DuckDB, SQL", "query often ports"),
    ("Daft", "UDF contract"),
    ("Ray Data", "no object store"),
]

body = [heading(490, 40, "WHERE YOUR CODE RUNS TODAY", anchor="middle", kind="grey")]
for i, (name, shift) in enumerate(SOURCES):
    body.append(card(col_x(i), SRC_Y, CW, 76, name, shift))
    body.append(arrow(mid(i), SRC_Y + 76, mid(i), DST_Y - 6))

body += [
    label(mid(0) + 12, 186, "read first"),
    tint(
        col_x(0),
        DST_Y,
        3 * CW + 2 * GAP,
        76,
        "Transforming and collecting",
        "the verb-by-verb table",
    ),
    tint(col_x(3), DST_Y, CW, 76, "SQL guide", "bt.sql"),
    tint(col_x(4), DST_Y, CW, 76, "ML pipelines", "batch inference"),
    tint(col_x(5), DST_Y, CW, 76, "Ray Data", "the port guide"),
]

# Every first page leads to the same proof step.
for x in (mid(1), mid(3), mid(4), mid(5)):
    body.append(arrow(x, DST_Y + 76, x, END_Y - 6, "amber"))

body += [
    label(mid(1) + 12, 342, "then prove it"),
    hero(
        X0,
        END_Y,
        6 * CW + 5 * GAP,
        80,
        "Differences and verification",
        "what Batcher leaves out, and how to prove the port returns the same rows",
    ),
    note(
        490,
        496,
        "Name-by-name references list every public name for PySpark, Polars, Daft, and Ray Data.",
        anchor="middle",
    ),
]

write("migration_chooser", svg(W, H, "".join(body)))
print("wrote migration_chooser.svg")
