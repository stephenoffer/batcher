#!/usr/bin/env python3
"""Draw `cookbook_map.svg` -- the cookbook's four groups and the domains in each.

Source of truth: `docs/cookbook/index.md`. The groups are its four `{toctree}` captions, the
domains are the entries under each, and the counts are its tables' "Recipes" column, which
sum to the 145 the page opens with. Nothing here is counted independently of that page, so
when a recipe is added the page's tables and this script move together.
"""

from __future__ import annotations

from _authoring import FONT, band, heading, hero, note, svg, tint, write

W, H = 980, 606

CW, CH, GAP = 218, 78, 16
CX0 = 262  # first card column
BAND_H = 106

GROUPS = (
    (
        ("THE RELATIONAL", "CORE"),
        "blue",
        (
            ("Dataset", "joins, grouping, reshaping", 14),
            ("Expressions", "strings, dates, nested", 39),
            ("I/O", "Parquet, text, Arrow", 6),
        ),
    ),
    (
        ("BUILDING AND", "RUNNING PIPELINES"),
        "amber",
        (
            ("Data engineering", "ingest, reconcile, repair", 11),
            ("Analytics", "cohorts, funnels, sessions", 11),
            ("Streaming", "time and restarts", 7),
        ),
    ),
    (
        ("MODELS AND", "MEASUREMENT"),
        "blue",
        (
            ("ML", "preprocess to inference", 27),
            ("Metrics", "metrics and statistics", 20),
        ),
    ),
    (
        ("RUNNING IT", "SAFELY"),
        "grey",
        (
            ("Governance", "masks, row filters, lineage", 3),
            ("Operations", "config, plans, memory", 7),
        ),
    ),
)


def count(x: float, y: float, n: int, kind: str) -> str:
    """A recipe count, set as a small badge on a card's top-right corner."""
    cls = "amber" if kind == "amber" else "blue"
    return (
        f'<rect x="{x - 34}" y="{y - 10}" width="34" height="20" rx="10" class="pill-{cls}"/>'
        f'<text x="{x - 17}" y="{y + 4}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="11" font-weight="800" class="pt-{cls}">{n}</text>'
    )


body: list[str] = [
    hero(290, 20, 400, 74, "Cookbook", "145 runnable recipes, each asserting its own output")
]

y = 120
for (line1, line2), kind, domains in GROUPS:
    total = sum(n for _, _, n in domains)
    body += [
        band(20, y, 940, BAND_H, "", kind),
        heading(44, y + 40, line1, kind="grey"),
        heading(44, y + 58, line2, kind="grey"),
        note(44, y + 84, f"{total} recipes"),
    ]
    for i, (title, sub, n) in enumerate(domains):
        x = CX0 + i * (CW + GAP)
        card_kind = "amber" if kind == "amber" else "blue"
        body += [
            tint(x, y + 14, CW, CH, title, sub, kind=card_kind),
            count(x + CW - 8, y + 28, n, kind),
        ]
    y += BAND_H + 14

write("cookbook_map", svg(W, H, "".join(body)))
print("wrote cookbook_map.svg")
