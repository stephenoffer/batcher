#!/usr/bin/env python3
"""Draw `dq_contract_matrix.svg` -- row-level, relation-level and schema checks, by terminal.

Source of truth: `docs/user-guide/trust/data-contracts.md` and
`python/batcher/api/dataset/dq/apply.py`. `reject_unfilterable` raises `PlanError` from
`drop`, `quarantine` and `annotate` whenever the chain holds an `AggregateConstraint` (a row
count, a mean, a freshness bound), because no row violates one. `schema_gate` raises
`DataQualityError` from the same three terminals when an enforced schema constraint is
unsatisfied, before any row work. `validate` measures the row-level counts and every
relation-level aggregate in a single keyless aggregate (uniqueness and referential integrity
take one extra pass apiece), and `fail` raises on the report
(`api/dataset/dq/accessor.py`). The schema is known before anything runs, which is the
page's reason a schema check costs nothing.

The figure exists because the three families share one accessor and one chain and part
company only at the terminal, which is a grid: three kinds of check against the terminals
that accept, refuse, or gate on each.
"""

from __future__ import annotations

from _authoring import (
    AMBER_DEEP,
    FONT,
    band,
    heading,
    label,
    mark,
    note,
    svg,
    write,
)

W, H = 980, 584

COLS = (430, 634, 838)  # column centres
LEFT = 38
ROW_Y = (170, 232, 294, 368, 450)  # vertical centre of each row


def text(x: float, y: float, s: str, bold: bool = False) -> str:
    """A centred cell line, in the title colour when bold and the muted colour otherwise."""
    cls, weight = ("t-title", "700") if bold else ("t-sub", "400")
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{FONT}" font-size="12" '
        f'font-weight="{weight}" class="{cls}">{s}</text>'
    )


def gate(x: float, y: float) -> str:
    """A diamond: the terminal runs only if the schema contract holds."""
    return (
        f'<path d="M {x} {y - 11} L {x + 11} {y} L {x} {y + 11} L {x - 11} {y} Z" '
        f'fill="{AMBER_DEEP}"/>'
        f'<path d="M {x} {y - 5} L {x} {y + 1.5}" stroke="#ffffff" stroke-width="2.2" '
        f'stroke-linecap="round"/><circle cx="{x}" cy="{y + 5}" r="1.3" fill="#ffffff"/>'
    )


body: list[str] = [
    note(
        490,
        40,
        "All three live on ds.dq and chain together. They part company at the terminal.",
        anchor="middle",
    ),
    band(20, 62, 940, 498, "", "grey"),
    heading(COLS[0], 100, "ROW-LEVEL", anchor="middle"),
    heading(COLS[1], 100, "RELATION-LEVEL", anchor="middle", kind="amber"),
    heading(COLS[2], 100, "SCHEMA", anchor="middle", kind="grey"),
    note(COLS[0], 120, "data-quality page", anchor="middle"),
    note(COLS[1], 120, "this page", anchor="middle"),
    note(COLS[2], 120, "this page", anchor="middle"),
]

# Row dividers.
for y in (140, 201, 263, 325, 410):
    body.append(f'<path d="M 36 {y} L 944 {y}" stroke="#cbd5e1" stroke-width="1"/>')

body += [
    label(LEFT, ROW_Y[0] + 5, "What it decides"),
    text(COLS[0], ROW_Y[0] + 5, "each row: valid or not", bold=True),
    text(COLS[1], ROW_Y[0] + 5, "one number for the table", bold=True),
    text(COLS[2], ROW_Y[0] + 5, "column names and types", bold=True),
    label(LEFT, ROW_Y[1] + 5, "For example"),
    text(COLS[0], ROW_Y[1] - 4, "not_null, in_range,"),
    text(COLS[0], ROW_Y[1] + 13, "not_in_future"),
    text(COLS[1], ROW_Y[1] - 4, "row_count_between,"),
    text(COLS[1], ROW_Y[1] + 13, "mean_between, fresh_within"),
    text(COLS[2], ROW_Y[1] - 4, "has_columns,"),
    text(COLS[2], ROW_Y[1] + 13, "column_types"),
    label(LEFT, ROW_Y[2] + 5, "What it costs"),
    text(COLS[0], ROW_Y[2] - 4, "one keyless aggregate;"),
    text(COLS[0], ROW_Y[2] + 13, "unique, references add a pass"),
    text(COLS[1], ROW_Y[2] - 4, "the same"),
    text(COLS[1], ROW_Y[2] + 13, "keyless aggregate"),
    text(COLS[2], ROW_Y[2] - 4, "nothing: known"),
    text(COLS[2], ROW_Y[2] + 13, "before anything runs"),
    # validate / fail
    label(LEFT, ROW_Y[3] - 2, "validate() / fail()"),
    note(LEFT, ROW_Y[3] + 16, "report, or raise"),
    mark(COLS[0], ROW_Y[3] - 8, True),
    text(COLS[0], ROW_Y[3] + 22, "counts violations"),
    mark(COLS[1], ROW_Y[3] - 8, True),
    text(COLS[1], ROW_Y[3] + 22, "carries the measured value"),
    mark(COLS[2], ROW_Y[3] - 8, True),
    text(COLS[2], ROW_Y[3] + 22, "names the mismatch"),
    # drop / quarantine / annotate
    label(LEFT, ROW_Y[4] - 10, "drop() / quarantine() /"),
    label(LEFT, ROW_Y[4] + 8, "annotate()"),
    note(LEFT, ROW_Y[4] + 26, "act on individual rows"),
    mark(COLS[0], ROW_Y[4] - 10, True),
    text(COLS[0], ROW_Y[4] + 20, "removes, splits or"),
    text(COLS[0], ROW_Y[4] + 36, "labels failing rows"),
    mark(COLS[1], ROW_Y[4] - 10, False),
    text(COLS[1], ROW_Y[4] + 20, "refused with PlanError:", bold=True),
    text(COLS[1], ROW_Y[4] + 36, "no violating row exists"),
    gate(COLS[2], ROW_Y[4] - 10),
    text(COLS[2], ROW_Y[4] + 20, "a gate: DataQualityError", bold=True),
    text(COLS[2], ROW_Y[4] + 36, "first, if the schema is unmet"),
    note(
        490,
        538,
        "Check relation-level bounds with validate() or fail(), and keep row-level checks "
        "in the chain you drop or quarantine on.",
        anchor="middle",
    ),
]

write("dq_contract_matrix", svg(W, H, "".join(body)))
print("wrote dq_contract_matrix.svg")
