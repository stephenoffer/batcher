#!/usr/bin/env python3
"""Draw `quickstart_lazy_plan.svg`: lazy steps build a plan, and one terminal call runs it.

Source of truth: `docs/getting-started/quickstart.md` ("A Dataset is lazy", "Run the plan
and inspect it", "Next steps") and `docs/getting-started/concepts/expressions.md` (the
Rust data plane evaluates expressions over whole Arrow batches). Each transformation
returns a new `Dataset` describing a plan; a terminal operation such as `collect` or
`to_pydict` runs it. Before anything runs, the optimizer pushes filters into the scan,
prunes unused columns, and picks join strategies.

Layout: top row is the lazy half (code, then the plan it builds); bottom row is what
the terminal call sets off, left to right.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, arrow, band, card, code, hero, label, note, svg, tint, write

W, H = 980, 520

CODE = [
    'bt.read.parquet("sales.parquet")',
    '  .filter(bt.col("price") >= 30.0)',
    "  .with_columns(total=...)",
    '  .group_by("category").agg(...)',
]

# The terminal call is the one amber edge: it leaves the plan and enters the execution
# row at its first step, as an elbow so it does not cross the other cards.
elbow = (
    f'<path d="M 766 176 L 766 304 L 179 304 L 179 324" fill="none" stroke="{AMBER_DEEP}" '
    'stroke-width="2.4" stroke-linejoin="round" marker-end="url(#arA)"/>'
)

body = [
    band(20, 20, 940, 206, "LAZY  ·  EACH STEP RETURNS A NEW DATASET, NOTHING RUNS", "blue"),
    code(48, 62, CODE, 380, size=13),
    arrow(440, 123, 590, 123),
    label(515, 111, "describes", anchor="middle"),
    hero(600, 70, 332, 106, "A query plan", "scan, filter, project, aggregate"),
    note(490, 208, "Chain as many steps as you like. No row has been read yet.", anchor="middle"),
    band(20, 262, 940, 234, "TERMINAL CALL  ·  THE PLAN RUNS ONCE", "amber"),
    elbow,
    label(752, 249, "collect()  ·  to_pydict()  ·  write", anchor="end"),
    card(48, 330, 262, 92, "Optimize", "push filters, prune columns"),
    card(372, 330, 262, 92, "Rust engine", "whole Arrow batches"),
    tint(696, 330, 236, 92, "Result", "Table, dict, or files", "amber"),
    arrow(310, 376, 372, 376),
    label(341, 362, "plan", anchor="middle"),
    arrow(634, 376, 696, 376),
    label(665, 362, "rows", anchor="middle"),
    note(
        490,
        466,
        "The optimizer sees the whole plan before any work, so it can move a filter into the read.",
        anchor="middle",
    ),
]


write("quickstart_lazy_plan", svg(W, H, "".join(body)))
print("wrote quickstart_lazy_plan.svg")
