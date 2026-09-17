#!/usr/bin/env python3
"""Draw `sql_dataframe_one_plan.svg`: SQL and the DataFrame API build one logical plan.

Source of truth: `docs/getting-started/tutorials/foundations/sql-to-dataframe.md`, step 4
("Both spellings build one `LogicalPlan`, push it through one optimizer, and run on one
Rust data plane"), and `python/batcher/api/session/sql.py::sql`, whose docstring says the
query "is parsed and optimized through the same engine as the DataFrame API" with a
sqlglot read dialect defaulting to ``duckdb``. The translator lives in
`python/batcher/_sql/parser/`.

The point of the picture is the merge: the two inputs differ only before the plan exists,
so `explain()` on the optimized plan cannot tell them apart.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, arrow, band, card, hero, label, note, pill, svg, write

W, H = 980, 420

CARD_H = 88
ROW_A, ROW_B = 70, 246  # the two input cards
MID_A, MID_B = ROW_A + CARD_H / 2, ROW_B + CARD_H / 2
MID = (MID_A + MID_B) / 2  # the single row everything after the merge sits on
HERO_H = 110


def bend(x1: float, y1: float, x2: float, y2: float) -> str:
    """An S-shaped connector with an arrowhead, for the merge of the two rows."""
    mx = (x1 + x2) / 2
    return (
        f'<path d="M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}" fill="none" '
        f'stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arB)"/>'
    )


body = [
    band(20, 20, 262, 350, "TWO SPELLINGS", "grey"),
    card(40, ROW_A, 222, CARD_H, "SQL string", "bt.sql, ds.sql, Session.sql"),
    card(40, ROW_B, 222, CARD_H, "DataFrame chain", ".filter .group_by .agg"),
    band(302, 20, 658, 350, "ONE QUERY", "blue"),
    hero(418, MID - HERO_H / 2, 166, HERO_H, "LogicalPlan", "one tree"),
    bend(262, MID_A, 416, MID - 24),
    label(318, MID_A - 6, "sqlglot parse", size=11.5),
    bend(262, MID_B, 416, MID + 24),
    label(318, MID_B + 22, "one node per call", size=11.5),
    card(652, MID - CARD_H / 2, 142, CARD_H, "Kyber", "the optimizer"),
    arrow(584, MID, 650, MID),
    label(617, MID - 12, "optimize", anchor="middle", size=11.5),
    card(858, MID - CARD_H / 2, 84, CARD_H, "Rust", "engine"),
    arrow(794, MID, 856, MID, "amber"),
    label(825, MID - 12, "JSON IR", anchor="middle", size=11.5),
    f'<path d="M 825 {MID + 8} L 825 {MID + 92}" fill="none" stroke="{AMBER_DEEP}" '
    'stroke-width="1.4" stroke-dasharray="3 3"/>',
    pill(825, MID + 112, "explain() renders this", kind="amber", anchor="middle"),
    note(825, MID + 136, "identical for both spellings", anchor="middle"),
    note(
        490,
        400,
        "Nothing runs until a terminal such as to_pydict(). There is no second SQL engine.",
        anchor="middle",
    ),
]

write("sql_dataframe_one_plan", svg(W, H, "".join(body)))
print("wrote sql_dataframe_one_plan.svg")
