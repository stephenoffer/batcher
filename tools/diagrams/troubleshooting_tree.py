#!/usr/bin/env python3
"""Draw `troubleshooting_tree.svg`: from a symptom, to its usual cause, to what to do.

Source of truth: `docs/user-guide/operate/running/troubleshooting.md`, one row per section or
symptom-table row, each cause taken from the page or the page it routes to:

* nothing ran: a Dataset is lazy, so call a terminal operation (this page);
* `ColumnNotFoundError`: a typo, or an earlier step dropped or renamed the column; `.columns`
  shows the schema at any point (this page);
* `PlanError` from `filter`: a raw string is not a predicate, so build it with `bt.col` or use
  `ds.sql` (this page);
* correct but slow: `explain-plans.md` names a predicate that failed to push, a build side
  chosen the wrong way round, and an estimate off by 100x;
* the same query recomputes: `caching.md`, a Dataset is a plan, so two terminals run it twice;
* out of memory: stateful operators hold state in memory by default; `spill=True` lets them
  spill (this page).

Form: a decision tree whose leaves are laid out as rows, so the three columns read as
symptom, cause and fix, and the reader scans down the first column only.
"""

from __future__ import annotations

from _authoring import (
    FONT,
    arrow,
    band,
    heading,
    hero,
    label,
    ribbon,
    svg,
    tint,
    write,
)

W, H = 980, 560

Y0, ROW_H, PITCH = 96, 56, 72
SYM_X, SYM_W = 244, 200
CAUSE_X = 506
FIX_X, FIX_W = 800, 144

ROWS = (
    (
        "Nothing ran",
        "a Dataset repr came back",
        ("A Dataset is lazy: no terminal", "operation was called."),
        "call collect()",
        "this page",
    ),
    (
        "ColumnNotFoundError",
        "",
        ("A typo, or an earlier step", "dropped or renamed it."),
        "check .columns",
        "this page",
    ),
    (
        "PlanError on filter()",
        "",
        ("A string was passed where", "an expression belongs."),
        "use bt.col()",
        "or ds.sql()",
    ),
    (
        "Correct but slow",
        "",
        ("A filter didn't push, a wrong", "build side, or an estimate miss."),
        "read the plan",
        "explain-plans",
    ),
    (
        "Recomputes each time",
        "",
        ("A Dataset is a plan, so every", "terminal runs it again."),
        "cache() it",
        "caching",
    ),
    (
        "Out of memory",
        "",
        ("State is held in memory;", "spilling is off by default."),
        "spill=True",
        "spilling",
    ),
)


def text_lines(x: float, y: float, lines: tuple[str, str]) -> str:
    """Two lines of cause text, vertically centred on `y`."""
    return "".join(
        f'<text x="{x}" y="{y - 3 + i * 17}" font-family="{FONT}" font-size="11.5" '
        f'class="t-sub">{line}</text>'
        for i, line in enumerate(lines)
    )


body: list[str] = [
    band(20, 20, 940, 520, "START FROM THE SYMPTOM", "grey"),
    heading(SYM_X + SYM_W / 2, 80, "SYMPTOM", anchor="middle"),
    heading(CAUSE_X + 108, 80, "USUAL CAUSE", anchor="middle", kind="grey"),
    heading(FIX_X + FIX_W / 2, 80, "WHAT TO DO", anchor="middle", kind="amber"),
]

centres = [Y0 + i * PITCH + ROW_H / 2 for i in range(len(ROWS))]
root_y = (centres[0] + centres[-1]) / 2
body.append(hero(36, root_y - 52, 164, 104, "Start here", "what do you see?"))

for cy, (symptom, sub, cause, fix, where) in zip(centres, ROWS, strict=True):
    top = cy - ROW_H / 2
    body += [
        ribbon(200, root_y, SYM_X - 2, cy),
        tint(SYM_X, top, SYM_W, ROW_H, symptom, sub),
        arrow(SYM_X + SYM_W + 6, cy, CAUSE_X - 10, cy, "grey"),
        label((SYM_X + SYM_W + CAUSE_X) / 2 - 2, cy - 8, "why", anchor="middle", size=10.5),
        text_lines(CAUSE_X, cy, cause),
        arrow(FIX_X - 40, cy, FIX_X - 6, cy, "amber"),
        label(FIX_X - 23, cy - 8, "fix", anchor="middle", size=10.5),
        tint(FIX_X, top, FIX_W, ROW_H, fix, where, "amber"),
    ]

write("troubleshooting_tree", svg(W, H, "".join(body)))
print("wrote troubleshooting_tree.svg")
