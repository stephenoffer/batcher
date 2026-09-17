#!/usr/bin/env python3
"""Draw `transform_rows_vs_columns.svg` -- which verbs change rows and which change columns.

Source of truth: `docs/user-guide/transform/index.md` (the section splits into the verbs that
decide which rows survive and in what order, and the expression language that decides what a
column contains; an expression lowers to a Rust expression tree over whole Arrow batches, and
its example chain `filter`, `with_columns(... .str.to_case("title") ... .floor())`, `sort`),
`docs/user-guide/transform/rows/index.md` (selecting and deriving, filtering, sorting,
deduplicating and sampling) and `.claude/rules/python-control-plane.md` (`select` chooses or
derives the full output; `with_columns` adds or replaces).

The matrix marks only what each verb is for. A check means the verb's job is to change that
property; a cross means it leaves it alone.
"""

from __future__ import annotations

from _authoring import arrow, band, code, heading, label, mark, note, svg, write

W, H = 980, 440

VERB_X = 64
COLS = (298, 402, 506)
ROW0, ROW_H = 142, 40

# (verb, drops rows, sets order, changes columns)
VERBS = (
    (".filter(pred)", True, False, False),
    (".distinct()", True, False, False),
    (".sample(...)", True, False, False),
    (".sort(key)", False, True, False),
    (".select(...)", False, False, True),
    (".with_columns(...)", False, False, True),
)


def verb(y: float, text: str) -> str:
    """A verb name in the matrix's first column."""
    return (
        f'<text x="{VERB_X}" y="{y + 5}" font-family="Menlo,Consolas,DejaVu Sans Mono,monospace" '
        f'font-size="13" font-weight="700" class="t-code">{text}</text>'
    )


body = [
    band(20, 20, 560, 400, "ROW VERBS AND COLUMN VERBS", "blue"),
    heading(COLS[0], 76, "ROWS", anchor="middle"),
    note(COLS[0], 96, "which survive", anchor="middle"),
    heading(COLS[1], 76, "ORDER", anchor="middle"),
    note(COLS[1], 96, "set by the verb", anchor="middle"),
    heading(COLS[2], 76, "COLUMNS", anchor="middle"),
    note(COLS[2], 96, "what they hold", anchor="middle"),
    '<path d="M 44 118 H 556" stroke="#cbd5e1" stroke-width="1.2"/>',
]

for i, (name, *cells) in enumerate(VERBS):
    y = ROW0 + i * ROW_H
    body.append(verb(y, name))
    body += [mark(x, y, ok) for x, ok in zip(COLS, cells, strict=True)]

# The seam between the two halves of the section.
seam = ROW0 + 3.5 * ROW_H
body += [
    f'<path d="M 44 {seam} H 556" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5 4"/>',
    note(
        300,
        400,
        "Every call returns a new lazy Dataset. Nothing runs until a result is asked for.",
        anchor="middle",
    ),
]

# The column language feeds the verbs that evaluate an expression.
body += [
    band(620, 20, 340, 400, "THE COLUMN LANGUAGE", "amber"),
    note(644, 68, "An Expr says what a column contains:"),
    code(
        644,
        84,
        [
            'bt.col("age") >= 18',
            'bt.col("name").str.to_case("title")',
            '(bt.col("age") / 10).floor()',
        ],
        296,
        size=12,
    ),
    note(644, 194, "It lowers to a Rust expression tree that"),
    note(644, 214, "runs over whole Arrow batches, so column"),
    note(644, 234, "work never becomes one Python call per row."),
    arrow(630, 322, 572, 322, "amber"),
    label(644, 316, "evaluated by .filter,"),
    label(644, 336, ".select and .with_columns"),
    note(644, 376, "A batch UDF is the escape hatch: your"),
    note(644, 396, "Python over whole Arrow batches."),
]

write("transform_rows_vs_columns", svg(W, H, "".join(body)))
print("wrote transform_rows_vs_columns.svg")
