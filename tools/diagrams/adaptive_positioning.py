#!/usr/bin/env python3
"""Draw `adaptive_positioning.svg` -- what the adaptive loop actually buys, honestly.

This diagram carries a claim the project has explicitly disciplined, so read
`docs/internals/competitive_architecture.md` ("Claims to retire", item 1) before
changing it. The retired claim is that Batcher re-optimizes *more finely* than Spark
AQE. It does not. `python/batcher/api/adaptive/` says so in its own first line:
stage-boundary re-optimization, the same mechanism and the same granularity as AQE.

The differentiator is therefore not *how often* the plan is revised inside one query.
It is two other things, and this diagram is built to show exactly those:

1. **Where it is available.** AQE is a cluster mechanism. Batcher runs the same
   stage-boundary loop on a single node, where DuckDB has no equivalent at all.
2. **Whether anything survives the query.** Batcher's measurements are recorded to the
   MetadataHub and read by the *next* run: sketch-backed cardinality, cost coefficients
   calibrated from measured operator times, and a UCB1 bandit over join strategies
   (`kyber/learning.py`, `kyber/learned_tuning/`). Neither DuckDB nor Spark has a
   comparable cross-query loop.

The honest caveat is drawn, not omitted. The within-query loop engages only on a plan
that contains a join and whose total scan input clears `_ADAPTIVE_MIN_INPUT_ROWS`
(20,000,000 rows) **or** `_ADAPTIVE_MIN_INPUT_BYTES` (20,000,000 x 64, about 1.3 GB),
both in `python/batcher/api/adaptive/gating.py`. Most small queries never reach it.

If the gate constant or the sanctioned wording changes, change this diagram with it.

Form: a capability matrix rather than a timeline, because a timeline invites exactly
the "more marks means better" reading the retired claim was made of. Every cell states
its answer in words as well as a glyph, so nothing is carried by colour or shape alone.
"""

from __future__ import annotations

from _authoring import FONT, write

W, H = 980, 400

COL_X = (300, 528, 756)        # left edge of each capability column
COL_W = 208
ROW_Y = (150, 226, 302)
ROW_H = 62
NAME_X = 40

STYLE = """<style>
  .surf { fill: #ffffff; }
  .cell { fill: #f8fafc; stroke: #e2e8f0; }
  .cell-yes { fill: #eff6ff; stroke: #bfdbfe; }
  .cell-bt { fill: #dbeafe; stroke: #2563eb; }
  .t-head { fill: #1e293b; }
  .t-sub { fill: #5b6675; }
  .t-yes { fill: #1d4ed8; }
  .t-no { fill: #64748b; }
  .colhead { fill: #1e293b; }
  @media (prefers-color-scheme: dark) {
    .surf { fill: #131c31; }
    .cell { fill: #0f172a; stroke: #243049; }
    .cell-yes { fill: #172554; stroke: #1e3a8a; }
    .cell-bt { fill: #1e3a8a; stroke: #60a5fa; }
    .t-head { fill: #e2e8f0; }
    .t-sub { fill: #94a3b8; }
    .t-yes { fill: #93c5fd; }
    .t-no { fill: #94a3b8; }
    .colhead { fill: #e2e8f0; }
  }
</style>"""


def cell(cx: float, cy: float, has: bool, text: str, emphasis: bool = False) -> str:
    """One capability answer. The glyph and the words agree, so neither stands alone."""
    cls = "cell-bt" if emphasis else ("cell-yes" if has else "cell")
    tcls = "t-yes" if has else "t-no"
    glyph = "yes" if has else "no"
    return (
        f'<rect x="{cx}" y="{cy}" width="{COL_W}" height="{ROW_H}" rx="9" class="{cls}" '
        f'stroke-width="1.4"/>'
        f'<text x="{cx + 16}" y="{cy + 26}" font-family="{FONT}" font-size="13" '
        f'font-weight="700" class="{tcls}">{glyph}</text>'
        f'<text x="{cx + 16}" y="{cy + 46}" font-family="{FONT}" font-size="11" '
        f'class="t-sub">{text}</text>'
    )


parts = [
    f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" class="surf"/>',
    f'<text x="40" y="46" font-family="{FONT}" font-size="19" font-weight="700" class="t-head">'
    f'What the adaptive loop actually buys</text>',
    f'<text x="40" y="70" font-family="{FONT}" font-size="13" class="t-sub">'
    f'Not a finer re-planning grain than Spark AQE. The same grain, in two places AQE and '
    f'DuckDB do not reach.</text>',
]

headers = (
    ("Re-plans inside", "one query"),
    ("Runs on a", "single node"),
    ("Carries what it", "learned to the next run"),
)
for x, (h1, h2) in zip(COL_X, headers):
    parts += [
        f'<text x="{x + 16}" y="{ROW_Y[0] - 30}" font-family="{FONT}" font-size="12.5" '
        f'font-weight="700" class="colhead">{h1}</text>',
        f'<text x="{x + 16}" y="{ROW_Y[0] - 14}" font-family="{FONT}" font-size="12.5" '
        f'font-weight="700" class="colhead">{h2}</text>',
    ]

rows = (
    ("DuckDB", "static optimizer",
     [(False, "optimizes once, up front"), (True, "single-node by design"), (False, "no cross-run state")]),
    ("Spark AQE", "cluster only",
     [(True, "at stage boundaries"), (False, "needs shuffle stages"), (False, "no cross-run state")]),
    ("Batcher", "same grain, wider reach",
     [(True, "at stage boundaries"), (True, "same loop, one node"),
      (True, "sketches, costs, bandit")]),
)
for y, (name, sub, cells) in zip(ROW_Y, rows):
    emph = name == "Batcher"
    parts += [
        f'<text x="{NAME_X}" y="{y + 26}" font-family="{FONT}" font-size="15" font-weight="700" '
        f'class="t-head">{name}</text>',
        f'<text x="{NAME_X}" y="{y + 45}" font-family="{FONT}" font-size="11" '
        f'class="t-sub">{sub}</text>',
    ]
    for x, (has, text) in zip(COL_X, cells):
        parts.append(cell(x, y, has, text, emphasis=emph and has))

parts += [
    f'<text x="40" y="{H - 42}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f'The within-query loop is the same mechanism and granularity as Spark AQE, not something '
    f'finer. It also engages only on a joined query whose scan input clears 20M rows or '
    f'roughly 1.3 GB,</text>',
    f'<text x="40" y="{H - 24}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f'so most small queries never use it. The third column is the half with no DuckDB or Spark '
    f'equivalent. Source: docs/internals/competitive_architecture.md.</text>',
]

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
    f'width="{W}" height="{H}"><defs>{STYLE}</defs>{"".join(parts)}</svg>'
)
write("adaptive_positioning", svg)
print("wrote adaptive_positioning.svg")
