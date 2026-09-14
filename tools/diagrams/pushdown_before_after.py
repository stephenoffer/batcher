#!/usr/bin/env python3
"""Draw `pushdown_before_after.svg` — one plan tree, before and after pushdown.

Source of truth:

* `python/batcher/kyber/rules/pushdown.py` — `rewrite_predicate`, registered as the
  whole-plan `predicate_pushdown` rule in `Phase.PUSHDOWN`. Its module docstring states
  the split exactly: the predicate is split on `AND`, each conjunct that references only
  one side of the join is rewritten into that side's column names and attached beneath
  the join, and "conjuncts spanning both sides stay above the join".
* `python/batcher/kyber/rules/projections.py` — `rewrite_projection`, the whole-plan
  `projection_rewrite` rule, plus `required_columns_per_source`, the companion analysis
  that arrives at each `Scan` with exactly the columns it must produce.
* `python/batcher/plan/physical.py::PhysicalPlan` — where both analyses land, as
  `source_projections` and `source_predicates`, keyed per scan `source_id`.

The fact the picture is built around, and the one a before-and-after can carry where a
sentence cannot: **the `Filter` is still there afterwards.** `source_predicates` is a
hint to the connector, and the engine keeps its own filter as a re-check, so a source
that translates none of the predicate, or only part of it, is still correct. An "after"
tree without that node would show an optimization the code does not perform.

Edge annotations say what each edge carries in rows and columns. They are shapes, not
measurements: no row count here is claimed as a benchmark.

Form: two trees at the same scale, with every node that did not move left at the same
height, so the eye lands on the two that did.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, GREY, band, label, note, svg, write

W, H = 980, 540

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

NW, NH = 150, 46
SW, SH = 168, 52


def node(x: float, y: float, w: float, h: float, title: str, sub: str, changed: bool = False) -> str:
    """One plan node. `changed` outlines a node this rewrite moved or narrowed."""
    stroke = AMBER_DEEP if changed else "#cbd5e1"
    width = "2.2" if changed else "1.2"
    return (
        f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" '
        f'class="surface" stroke="{stroke}" stroke-width="{width}"/></g>'
        f'<text x="{x + w / 2}" y="{y + 20}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="13" font-weight="700" class="t-title">{title}</text>'
        f'<text x="{x + w / 2}" y="{y + 37}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="10.5" fill="{BLUE}">{sub}</text>'
    )


def up(x1: float, y1: float, x2: float, y2: float, kind: str = "grey") -> str:
    """An edge, drawn the way the rows travel: upward, out of the scans."""
    stroke, marker = {"grey": (GREY, "arG"), "amber": (AMBER_DEEP, "arA")}[kind]
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{stroke}" stroke-width="2.2" '
        f'marker-end="url(#{marker})"/>'
    )


body = [
    band(20, 20, 456, 440, "AS WRITTEN", "grey"),
    band(504, 20, 456, 440, "AS KYBER LEAVES IT", "blue"),
]

# ---- Left: the plan as written -------------------------------------------
body += [
    node(170, 70, NW, NH, "Project", "o_id, c_name"),
    node(170, 150, NW, NH, "Filter", "o_date &gt;= X"),
    node(170, 230, NW, NH, "Join", "on c_id"),
    node(46, 344, SW, SH, "Scan orders", "every column"),
    node(276, 344, SW, SH, "Scan customers", "every column"),
    up(245, 148, 245, 120),
    up(245, 228, 245, 200),
    up(130, 342, 206, 280),
    up(360, 342, 284, 280),
    label(328, 142, "surviving rows", size=11),
    label(328, 222, "every matching row", size=11),
    note(130, 416, "every row, every column", anchor="middle"),
    note(360, 416, "every row, every column", anchor="middle"),
]

# ---- Right: the same plan after PUSHDOWN ---------------------------------
body += [
    node(660, 70, NW, NH, "Project", "o_id, c_name"),
    node(660, 150, NW, NH, "Join", "on c_id"),
    node(585, 246, NW, NH, "Filter", "o_date &gt;= X", changed=True),
    node(576, 344, SW, SH, "Scan orders", "o_id, c_id, o_date", changed=True),
    node(766, 344, SW, SH, "Scan customers", "c_id, c_name", changed=True),
    up(735, 148, 735, 120),
    up(660, 244, 700, 200, "amber"),
    up(660, 342, 660, 296),
    up(850, 342, 780, 200),
    label(818, 142, "fewer and narrower rows", size=11),
    f'<text x="648" y="226" text-anchor="end" font-family="{FONT}" font-size="11" '
    f'font-weight="700" fill="{AMBER_DEEP}">gone before the build</text>',
    note(660, 416, "3 columns, and the predicate,", anchor="middle"),
    note(660, 434, "so the source may skip row groups", anchor="middle"),
    note(850, 416, "2 columns", anchor="middle"),
]

body += [
    note(490, 492,
         "The Filter stays put. source_predicates is a hint, so a connector that translates none of it, or only part of it, is still correct.",
         anchor="middle"),
    f'<text x="490" y="516" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" fill="{AMBER_DEEP}">Only a conjunct naming one side of the join '
    f'moves below it. One that names both stays above.</text>',
]

write("pushdown_before_after", svg(W, H, "".join(body)))
print("wrote pushdown_before_after.svg")
