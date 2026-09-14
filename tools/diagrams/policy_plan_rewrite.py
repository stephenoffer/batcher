#!/usr/bin/env python3
"""Draw `policy_plan_rewrite.svg` - governance as a plan rewrite, before and after.

Source of truth: `python/batcher/governance/enforce.py`. Every governed `Scan` becomes
``Project(visible columns, masked) -> Filter(row-access predicate) -> Scan(table)``, and
the rewrite runs before the optimizer. The two orderings are what the picture exists to
state: the filter sits *below* the projection, so a row policy may reference a column the
principal holds no SELECT on; the projection sits at the leaf, so nothing the user wrote
can ever see a raw value.

A denied column is removed from the scan's output rather than flagged, which is why there
is no runtime check to bypass. Keep this diagram in step with that module's docstring.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 700

LEFT_X = 230  # centre of the "plan you wrote" column
RIGHT_X = 740  # centre of the "plan that runs" column

body = [
    band(20, 20, 940, 92, "GOVERNANCE RUNS ONCE, BEFORE THE OPTIMIZER", "grey"),
    card(
        230,
        44,
        520,
        56,
        "enforce(plan, tables, principal, catalog)",
        "pure: a plan goes in, a governed plan and its audit events come out",
    ),
    # The plan the user wrote.
    band(20, 132, 420, 388, "THE PLAN YOU WROTE", "grey"),
    card(60, 170, 340, 52, "aggregate", "sum(amount) by region"),
    card(60, 446, 340, 52, "scan customers", "every column, every row"),
    arrow(LEFT_X, 446, LEFT_X, 228, "grey"),
    label(LEFT_X + 14, 330, "all rows, all columns", size=12),
    # The rewrite.
    arrow(444, 325, 514, 325, "amber"),
    label(479, 310, "rewrite", anchor="middle", size=12),
    # The plan that actually runs.
    band(520, 132, 440, 388, "THE PLAN THAT RUNS", "blue"),
    card(550, 170, 380, 52, "aggregate", "your query, untouched"),
    card(550, 262, 380, 52, "project  (injected)", "the columns you may read, through their masks"),
    card(550, 354, 380, 52, "filter  (injected)", "the row-access predicate"),
    card(550, 446, 380, 52, "scan customers", "every column, every row"),
    arrow(RIGHT_X, 446, RIGHT_X, 412, "blue"),
    label(RIGHT_X + 14, 434, "raw rows", size=12),
    arrow(RIGHT_X, 354, RIGHT_X, 320, "amber"),
    label(RIGHT_X + 14, 342, "only the rows the policy allows", size=12),
    arrow(RIGHT_X, 262, RIGHT_X, 228, "amber"),
    label(RIGHT_X + 14, 250, "masked values, pruned columns", size=12),
    # Why the order, not just the presence, is the point.
    band(20, 552, 940, 128, "WHY THE ORDER IS LOAD-BEARING", "amber"),
    note(180, 596, "Filter below project:", anchor="middle"),
    note(180, 614, "a row policy may read a column", anchor="middle"),
    note(180, 632, "you hold no SELECT on.", anchor="middle"),
    note(490, 596, "Project at the leaf:", anchor="middle"),
    note(490, 614, "no operator you wrote", anchor="middle"),
    note(490, 632, "ever sees a raw value.", anchor="middle"),
    note(800, 596, "A denied column is removed", anchor="middle"),
    note(800, 614, "from the output, not flagged.", anchor="middle"),
    note(800, 632, "There is no check to bypass.", anchor="middle"),
    note(490, 662, "collect, count, iter_batches, write and the distributed path all run this same plan.", anchor="middle"),
]

write("policy_plan_rewrite", svg(W, H, "".join(body)))
print("wrote policy_plan_rewrite.svg")
