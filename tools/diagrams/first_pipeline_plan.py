#!/usr/bin/env python3
"""Draw `first_pipeline_plan.svg`: each step of the first pipeline beside the plan node it adds.

Source of truth: `docs/getting-started/tutorials/foundations/first-pipeline.md`, the
chained `result` pipeline and the `explain()` output in its "What a plan looks like"
dropdown: `scan [source 0]`, `project`, `aggregate [by category · sum, count_star]`,
`sort [revenue]`. Every transform returns a new `Dataset` and runs nothing; the terminal
operation (`to_pydict`, `collect`, `count`, `write`) executes the plan.

Layout: code on the left in the order you write it, the node each line adds on the
right, and data flowing down the node column. `explain()` prints the same tree with the
scan at the bottom, which the footnote says so the two pictures don't seem to disagree.
"""

from __future__ import annotations

from _authoring import arrow, card, code, heading, label, note, svg, tint, write

W, H = 980, 600

CODE_X, CODE_W = 30, 440
NODE_X, NODE_W = 590, 340
NODE_MID = NODE_X + NODE_W / 2
TOP, CARD_H, PITCH = 74, 60, 96

STEPS = [
    ("bt.from_pydict({...})", "scan", "source 0"),
    (".with_columns(total=...)", "project", "adds total"),
    ('.group_by("category").agg(...)', "aggregate", "by category · sum, count_star"),
    ('.sort("revenue", descending=True)', "sort", "by revenue"),
]

body = [
    heading(CODE_X, 40, "WHAT YOU WRITE", kind="grey"),
    heading(NODE_X, 40, "THE PLAN NODE IT ADDS", kind="blue"),
]

for i, (snippet, node, detail) in enumerate(STEPS):
    y = TOP + i * PITCH
    code_h = 20 + (13 + 7)
    body.append(code(CODE_X, y + (CARD_H - code_h) / 2, [snippet], CODE_W, size=13))
    body.append(arrow(CODE_X + CODE_W + 8, y + CARD_H / 2, NODE_X - 8, y + CARD_H / 2))
    body.append(label((CODE_X + CODE_W + NODE_X) / 2, y + CARD_H / 2 - 10, "adds", anchor="middle"))
    body.append(card(NODE_X, y, NODE_W, CARD_H, node, detail))
    if i < len(STEPS) - 1:
        body.append(arrow(NODE_MID, y + CARD_H, NODE_MID, y + PITCH - 4))
        body.append(label(NODE_MID + 12, y + CARD_H + 23, "feeds"))

# The terminal call adds no node. It runs the tree above.
y = TOP + len(STEPS) * PITCH
code_h = 20 + (13 + 7)
body += [
    code(CODE_X, y + (CARD_H - code_h) / 2, [".to_pydict()"], CODE_W, size=13),
    arrow(CODE_X + CODE_W + 8, y + CARD_H / 2, NODE_X - 8, y + CARD_H / 2, "amber"),
    label((CODE_X + CODE_W + NODE_X) / 2, y + CARD_H / 2 - 10, "runs", anchor="middle"),
    tint(NODE_X, y, NODE_W, CARD_H, "Executes the plan", "returns a column dict", "amber"),
    arrow(NODE_MID, y - PITCH + CARD_H, NODE_MID, y - 4, "amber"),
    label(NODE_MID + 12, y - PITCH + CARD_H + 23, "result"),
    note(
        CODE_X,
        y + CARD_H + 44,
        "Nothing above the last line runs. explain() prints this tree upside down, "
        "sort on top and scan at the bottom.",
    ),
]

write("first_pipeline_plan", svg(W, H, "".join(body)))
print("wrote first_pipeline_plan.svg")
