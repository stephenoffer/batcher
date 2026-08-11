#!/usr/bin/env python3
"""Draw `tpch_sf10.svg` -- the TPC-H scale-factor-10 standing on like-for-like input.

Every figure here is quoted from `benchmarks/BENCHMARK_RESULTS.md`, the entry dated
2026-07-27 ("Three things the join path could not do"). Provenance, in full, because
the charts rule requires it:

* **Suite**: TPC-H, all 22 queries, **scale factor 10**.
* **Hardware**: single node, 96 cores, release build.
* **Gate**: correctness-gated, all 22 reported `OK`. A query whose result disagrees
  with the oracle produces no timing at all.
* **Batcher's own total**: 4,453 ms (down from 4,993 ms in the same entry).

The two comparisons plotted are the like-for-like ones, where every engine reads the
same Arrow, and both are stated directly by the entry:

    duckdb_arrow    Batcher 1.89x faster overall, wins 21 of 22 (q9 is 1.01x, a tie)
    polars          Batcher 2.26x faster overall, wins 17 of 22

Only Batcher's and duckdb_arrow's absolute totals (4,453 ms and 8,436 ms) are recorded,
so this chart plots the **ratios**, which are all stated, rather than back-solving
totals that were never measured. That is deliberate: a derived number presented as a
measured one is exactly what the documentation contract forbids.

Form: bars anchored on a 1.0x parity line, so the magnitude of each margin is read
against a fixed reference rather than against each other. Bars are direct-labelled with
the ratio and the win count, so no legend is needed.
"""

from __future__ import annotations

from _authoring import FONT, write

W, H = 980, 334

AXIS_Y = 234           # baseline for the bars
MID_X = 500            # the 1.0x parity line
PX_PER_X = 150         # horizontal pixels per 1x of ratio
BAR_H = 44
ROW_Y = (120, 186)

STYLE = """<style>
  .surf { fill: #ffffff; }
  .plot { fill: #f8fafc; stroke: #e2e8f0; }
  .win { fill: #2563eb; }
  .parity { stroke: #64748b; }
  .t-head { fill: #1e293b; }
  .t-sub { fill: #5b6675; }
  .t-val { fill: #1e293b; }
  @media (prefers-color-scheme: dark) {
    .surf { fill: #131c31; }
    .plot { fill: #0f172a; stroke: #243049; }
    .win { fill: #60a5fa; }
    .parity { stroke: #94a3b8; }
    .t-head { fill: #e2e8f0; }
    .t-sub { fill: #94a3b8; }
    .t-val { fill: #e2e8f0; }
  }
</style>"""


def bar(y: float, ratio: float, engine: str, detail: str, wins: str) -> list[str]:
    """One comparison: a bar running right from parity, direct-labelled."""
    w = (ratio - 1.0) * PX_PER_X
    val_x = MID_X + w + 12
    return [
        f'<text x="40" y="{y + 20}" font-family="{FONT}" font-size="14.5" font-weight="700" '
        f'class="t-head">{engine}</text>',
        f'<text x="40" y="{y + 38}" font-family="{FONT}" font-size="11.5" '
        f'class="t-sub">{detail}</text>',
        f'<rect x="{MID_X}" y="{y}" width="{w}" height="{BAR_H}" rx="5" class="win"/>',
        f'<text x="{val_x}" y="{y + 22}" font-family="{FONT}" '
        f'font-size="16" font-weight="700" class="t-val">{ratio:.2f}x faster</text>',
        f'<text x="{val_x}" y="{y + 39}" font-family="{FONT}" '
        f'font-size="11.5" class="t-sub">{wins}</text>',
    ]


parts = [
    f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" class="surf"/>',
    f'<text x="40" y="50" font-family="{FONT}" font-size="19" font-weight="700" class="t-head">'
    f'TPC-H scale factor 10, all 22 queries</text>',
    f'<text x="40" y="74" font-family="{FONT}" font-size="13" class="t-sub">'
    f'Batcher total 4,453 ms on 96 cores, correctness-gated. Bars show the suite ratio '
    f'against each engine.</text>',
    # the parity rule, drawn behind the bars
    f'<line x1="{MID_X}" y1="100" x2="{MID_X}" y2="{AXIS_Y + 4}" class="parity" '
    f'stroke-width="1.6" stroke-dasharray="5 4"/>',
    f'<text x="{MID_X}" y="{AXIS_Y + 24}" text-anchor="middle" font-family="{FONT}" '
    f'font-size="11.5" font-weight="700" class="t-sub">1.0x parity</text>',
    f'<text x="{MID_X + 14}" y="112" font-family="{FONT}" font-size="11.5" '
    f'class="t-sub">Batcher ahead &#8594;</text>',
]

parts += bar(ROW_Y[0], 1.89, "DuckDB on the same Arrow",
             "like-for-like: identical zero-copy input", "wins 21 of 22 (q9 a tie at 1.01x)")
parts += bar(ROW_Y[1], 2.26, "Polars",
             "same Arrow input", "wins 17 of 22")

parts += [
    f'<text x="40" y="{H - 44}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f'Source: benchmarks/BENCHMARK_RESULTS.md, 2026-07-27. Ratios are suite totals; only '
    f'Batcher (4,453 ms) and DuckDB-on-Arrow (8,436 ms) have</text>',
    f'<text x="40" y="{H - 26}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f'recorded absolute totals, so the Polars bar is plotted from its stated ratio. '
    f'The box was shared during the run, and the entry notes totals swing about 25%.</text>',
]

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
    f'width="{W}" height="{H}"><defs>{STYLE}</defs>{"".join(parts)}</svg>'
)
write("tpch_sf10", svg)
print("wrote tpch_sf10.svg")
