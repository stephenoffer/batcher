#!/usr/bin/env python3
"""Draw `column_lineage.svg` - which inputs an output column derives from.

Source of truth: `python/batcher/governance/lineage.py::column_lineage`. A pure
`LogicalPlan -> dict` analysis computed bottom-up, executing nothing.

Three of its rules are what the picture states, because all three surprise people:
a join output takes its values from one named column of one side; `count()` has no
input and so no origin; and a filter contributes nothing at all, because the analysis
tracks data flow rather than control flow. The fourth is the safety direction - an
operator it does not model, `map_batches` above all, reports every output column as
deriving from every input column.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 650

body = [
    # The plan, read without running it.
    band(20, 20, 940, 200, "ONE PLAN, ANALYZED WITHOUT RUNNING IT", "grey"),
    card(40, 56, 178, 58, "scan orders", "customer_id, amount"),
    card(40, 144, 178, 58, "scan customers", "id, region, ssn"),
    card(288, 92, 150, 74, "join", "customer_id = id"),
    card(500, 92, 166, 74, "filter", "ssn is not null"),
    card(728, 92, 208, 74, "aggregate", "group by region"),
    arrow(218, 85, 282, 110, "blue"),
    label(230, 74, "left", size=12),
    arrow(218, 173, 282, 148, "blue"),
    label(230, 190, "right", size=12),
    arrow(438, 129, 494, 129, "grey"),
    label(466, 112, "rows only", anchor="middle", size=12),
    arrow(666, 129, 722, 129, "blue"),
    label(694, 112, "reduce", anchor="middle", size=12),
    # The answer, per output column.
    band(20, 252, 940, 268, 'column_lineage(plan, ["orders", "customers"])', "blue"),
    card(44, 300, 280, 86, "region", "customers.region"),
    card(350, 300, 280, 86, "total", "orders.amount"),
    card(656, 300, 280, 86, "n", "no origin at all"),
    note(184, 408, "the group key's own expression", anchor="middle"),
    note(490, 408, "the sum's input column", anchor="middle"),
    note(796, 408, "count() reads no column", anchor="middle"),
    note(490, 448, "customers.ssn appears in none of them. The filter chose which rows survived,", anchor="middle"),
    note(490, 466, "not what any value is, so it carries no lineage. Data flow, not control flow.", anchor="middle"),
    note(490, 498, "Tag customers.ssn as PII and the tag follows nothing here. Tag customers.region and it follows the first column.", anchor="middle"),
    # The safety direction.
    band(20, 552, 940, 86, "AN OPERATOR THE ANALYSIS DOES NOT MODEL", "amber"),
    note(490, 596, "map_batches is opaque, so every output column is reported as deriving from every input column.", anchor="middle"),
    note(490, 618, "It over-approximates on purpose: a false 'might carry PII' costs a review, a false 'cannot' costs a breach.", anchor="middle"),
]

write("column_lineage", svg(W, H, "".join(body)))
print("wrote column_lineage.svg")
