#!/usr/bin/env python3
"""Draw `physical_properties.svg` — ordering and partitioning, which do not work alike.

Source of truth, all under `python/batcher/`:

* `kyber/properties.py` — `PhysicalProperties(ordering, hash_partitioned_on,
  clustered_on)` and `satisfies()`, whose docstring calls the opposite containment
  "the single most error-prone thing here". An ordering satisfies a requirement when it
  is a prefix-extension of it; a partitioning satisfies one when the delivered keys are a
  *subset* of the required ones.
* `kyber/stats/estimator.py` — where an ordering actually lives. It is the fourth
  positional argument to each `RelStats(...)` the estimator builds, propagated bottom-up
  and memoized, and an operator destroys it by omitting the argument. `Scan` establishes
  from `SourceStatistics.sorted_by`, `Sort` from `_canonical_sort_prefix`; `Filter`,
  `Limit`, `Window`, `Sample` preserve; `Project` preserves renamed through
  `properties.project_ordering`; `Unnest` truncates; `Aggregate`, `Distinct`, `Join`,
  `Union` and `MapBatches` destroy.
* `io/stats/sortedness.py::proved_sorted_by` — a scan's ordering is *proved* from Parquet
  footers, not assumed: same leading column and direction in every row group of every
  file, row groups ordered within a file, files ordered across the dataset.
* `kyber/rules/ordering.py::sort_elimination_from_ordering` — the one rewrite rule that
  reads an ordering (`Sort(x, keys) -> x`).
* `kyber/properties.py::hash_partitioned_on` / `clustered_on` — recomputed recursively at
  the point of decision. Nothing stores a partitioning, and `properties.py`'s module
  docstring says why there is deliberately **no `Exchange` node and no enforcer**: it
  would mean a new `RelOp`, a matching Rust operator, and a two-sided wire-contract
  change, while `dist` already decides its own shuffles.
* `kyber/cost/shuffle.py::_already_partitioned_on` and `dist/executor.py`
  (`_fusable_join_aggregate`, `_partition_aligned_aggregate` / `_distinct` / `_window`) —
  the two consumers: a net cost of `0.0` in the model, and a skipped shuffle in `dist`.

**Note for anyone updating this:** `PhysicalPlan.PlanProperties` is *not* what this
diagram draws. Despite the name, it holds cardinality, cost and provenance only. Neither
sortedness nor partitioning is a field on `PhysicalPlan`.

The reason this is a diagram and not a table: the two properties do not merely differ in
their rules, they differ in *shape*. One travels along the plan and can be drawn as a
chain; the other is never carried at all and has to be drawn as a lookup that reaches
back into the plan. Rendering them as two columns of a comparison would hide exactly
that.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, GREY, MUTED, band, card, note, svg, write

W, H = 980, 640

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

PW, PH = 170, 52


def pill(x: float, y: float, w: float, title: str) -> str:
    """One operator in a plan chain."""
    return (
        f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{w}" height="{PH}" rx="9" '
        f'class="surface" stroke-width="1.2"/></g>'
        f'<text x="{x + w / 2}" y="{y + PH / 2 + 5}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="13" font-weight="700" class="t-title">{title}</text>'
    )


def keys(x: float, y: float, text: str) -> str:
    """The ordering an edge carries, in the engine's own key spelling."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{MONO}" font-size="11" '
        f'font-weight="700" fill="{BLUE}">{text}</text>'
    )


def verb(x: float, y: float, text: str) -> str:
    """What the operator below does to the property."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
        f'class="t-sub">{text}</text>'
    )


body: list[str] = []

# ---- Band 1: an ordering travels along the plan --------------------------
XS = [30, 280, 530, 780]
body += [
    band(20, 20, 940, 186, "AN ORDERING TRAVELS WITH THE PLAN", "blue"),
    pill(XS[0], 86, PW, "Scan orders"),
    pill(XS[1], 86, PW, "Filter"),
    pill(XS[2], 86, PW, "Project"),
    pill(XS[3], 86, PW, "Aggregate"),
    verb(XS[0] + PW / 2, 158, "establishes it, from"),
    verb(XS[0] + PW / 2, 174, "a proved footer order"),
    verb(XS[1] + PW / 2, 158, "preserves it"),
    verb(XS[2] + PW / 2, 158, "preserves it, under"),
    verb(XS[2] + PW / 2, 174, "the new column name"),
    verb(XS[3] + PW / 2, 158, "destroys it: a later"),
    verb(XS[3] + PW / 2, 174, "sort has to run"),
]
for i in range(3):
    x1 = XS[i] + PW + 8
    x2 = XS[i + 1] - 10
    body += [
        f'<path d="M {x1} 112 L {x2} 112" fill="none" stroke="{BLUE}" stroke-width="2.4" '
        f'marker-end="url(#arB)"/>',
        keys((x1 + x2) / 2, 104, ["(o_date)", "(o_date)", "(day)"][i]),
    ]
body.append(keys(XS[3] + PW / 2, 76, "and out the far side: ( )"))

# ---- Band 2: a partitioning is never carried -----------------------------
body += [
    band(20, 222, 940, 206, "A PARTITIONING IS NEVER CARRIED, ONLY RECOMPUTED", "grey"),
    pill(60, 268, 200, "Join on k"),
    pill(300, 268, 200, "Filter"),
    pill(540, 268, 200, "Aggregate on (k, x)"),
    f'<path d="M 268 294 L 290 294" fill="none" stroke="{GREY}" stroke-width="2.2" '
    f'marker-end="url(#arG)"/>',
    f'<path d="M 508 294 L 530 294" fill="none" stroke="{GREY}" stroke-width="2.2" '
    f'marker-end="url(#arG)"/>',
    note(279, 284, "rows", anchor="middle"),
    note(519, 284, "rows", anchor="middle"),
    card(540, 352, 400, 62, "dist scheduling, or the cost model", "the point of decision"),
    f'<path d="M 620 350 L 182 324" fill="none" stroke="{AMBER_DEEP}" stroke-width="2.4" '
    f'stroke-dasharray="6 4" marker-end="url(#arA)"/>',
    f'<text x="300" y="348" font-family="{FONT}" font-size="11.5" font-weight="700" '
    f'fill="{AMBER_DEEP}">walks the plan again, here and now</text>',
    note(60, 404, "Nothing stores it, and there is no Exchange node to enforce it."),
    note(60, 420, "The delivered (k) sits inside the required (k, x), so the shuffle is skipped."),
]

# ---- Band 3: the containment, which runs opposite ways -------------------
body += [
    band(20, 444, 940, 178, "AND THE TWO CONTAIN IN OPPOSITE DIRECTIONS", "amber"),
    card(
        50,
        482,
        400,
        84,
        "An ordering: longer satisfies shorter",
        "sorted by (a, b) is also sorted by (a)",
    ),
    card(
        510,
        482,
        400,
        84,
        "A partitioning: subset satisfies superset",
        "partitioned on (a) keeps every (a, b) group whole",
    ),
    f'<text x="250" y="592" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" fill="{AMBER_DEEP}">(a) alone does not satisfy (a, b).</text>',
    f'<text x="730" y="592" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" fill="{AMBER_DEEP}">(a, b) does not keep an (a) group whole.</text>',
    note(
        490,
        612,
        "Getting this backwards drops a sort that was needed, or skips a shuffle that was not optional.",
        anchor="middle",
    ),
]

body.append(
    f'<text x="490" y="{H - 8}" text-anchor="middle" font-family="{FONT}" font-size="11" '
    f'fill="{MUTED}">A wrong claim about either is a wrong answer, not a slow one.</text>'
)

write("physical_properties", svg(W, H, "".join(body)))
print("wrote physical_properties.svg")
