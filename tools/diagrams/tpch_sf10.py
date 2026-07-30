#!/usr/bin/env python3
"""Draw `tpch_sf10.svg` -- the TPC-H scale-factor-10 standing, wins and losses together.

Every figure here is quoted from `benchmarks/BENCHMARK_RESULTS.md`, the entry dated
2026-07-27 ("Three things the join path could not do"). Provenance, in full, because
the charts rule requires it:

* **Suite**: TPC-H, all 22 queries, **scale factor 10**.
* **Hardware**: single node, 96 cores, release build.
* **Gate**: correctness-gated, all 22 reported `OK`. A query whose result disagrees
  with the oracle produces no timing at all.
* **Batcher's own total**: 4,453 ms (down from 4,993 ms in the same entry).

The three comparisons are the ones the entry states directly:

    duckdb_arrow    Batcher 1.89x faster overall, wins 21 of 22 (q9 is 1.01x, a tie)
    polars          Batcher 2.26x faster overall, wins 17 of 22
    duckdb native   Batcher 2.08x behind,          wins  4 of 22 (q11, q15, q16, q22)

Only Batcher's and duckdb_arrow's absolute totals (4,453 ms and 8,436 ms) are recorded,
so this chart plots the **ratios**, which are all stated, rather than back-solving
totals that were never measured. That is deliberate: a derived number presented as a
measured one is exactly what the documentation contract forbids.

Form: a diverging bar around a 1.0x parity line, because the story is that Batcher is
ahead of two bars and behind one, and a parity-anchored axis is the only encoding that
shows direction and magnitude at once. Bars are direct-labelled with the ratio and the
win count, so no legend is needed. Win/loss is carried by position relative to the
parity rule and by the label text, never by colour alone.
"""

from __future__ import annotations

from _authoring import FONT, write

W, H = 980, 400

AXIS_Y = 300           # baseline for the bars
MID_X = 500            # the 1.0x parity line
PX_PER_X = 150         # horizontal pixels per 1x of ratio
BAR_H = 44
ROW_Y = (120, 186, 252)

STYLE = """<style>
  .surf { fill: #ffffff; }
  .plot { fill: #f8fafc; stroke: #e2e8f0; }
  .win { fill: #2563eb; }
  .loss { fill: #d97706; }
  .parity { stroke: #64748b; }
  .t-head { fill: #1e293b; }
  .t-sub { fill: #5b6675; }
  .t-val { fill: #1e293b; }
  @media (prefers-color-scheme: dark) {
    .surf { fill: #131c31; }
    .plot { fill: #0f172a; stroke: #243049; }
    .win { fill: #60a5fa; }
    .loss { fill: #f59e0b; }
    .parity { stroke: #94a3b8; }
    .t-head { fill: #e2e8f0; }
    .t-sub { fill: #94a3b8; }
    .t-val { fill: #e2e8f0; }
  }
</style>"""


def bar(y: float, ratio: float, faster: bool, engine: str, detail: str,
        wins: str) -> list[str]:
    """One comparison: a bar left or right of parity, direct-labelled."""
    length = (ratio - 1.0) * PX_PER_X
    cls = "win" if faster else "loss"
    if faster:
        x, w = MID_X, length
        val_x, val_anchor = MID_X + w + 12, "start"
    else:
        x, w = MID_X - length, length
        val_x, val_anchor = MID_X - w - 12, "end"
    arrow = "faster" if faster else "behind"
    return [
        f'<text x="40" y="{y + 20}" font-family="{FONT}" font-size="14.5" font-weight="700" '
        f'class="t-head">{engine}</text>',
        f'<text x="40" y="{y + 38}" font-family="{FONT}" font-size="11.5" '
        f'class="t-sub">{detail}</text>',
        f'<rect x="{x}" y="{y}" width="{w}" height="{BAR_H}" rx="5" class="{cls}"/>',
        f'<text x="{val_x}" y="{y + 22}" text-anchor="{val_anchor}" font-family="{FONT}" '
        f'font-size="16" font-weight="700" class="t-val">{ratio:.2f}x {arrow}</text>',
        f'<text x="{val_x}" y="{y + 39}" text-anchor="{val_anchor}" font-family="{FONT}" '
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
    f'<text x="{MID_X - 14}" y="112" text-anchor="end" font-family="{FONT}" font-size="11.5" '
    f'class="t-sub">&#8592; Batcher behind</text>',
]

parts += bar(ROW_Y[0], 1.89, True, "DuckDB on the same Arrow",
             "like-for-like: identical zero-copy input", "wins 21 of 22 (q9 a tie at 1.01x)")
parts += bar(ROW_Y[1], 2.26, True, "Polars",
             "same Arrow input", "wins 17 of 22")
parts += bar(ROW_Y[2], 2.08, False, "DuckDB, native store",
             "its own compressed format, no Arrow ingest",
             "wins 4 of 22")

parts += [
    f'<text x="40" y="{H - 44}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f'Source: benchmarks/BENCHMARK_RESULTS.md, 2026-07-27. Ratios are suite totals; only '
    f'Batcher (4,453 ms) and DuckDB-on-Arrow (8,436 ms) have</text>',
    f'<text x="40" y="{H - 26}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f'recorded absolute totals, so the other bars are plotted from their stated ratios. '
    f'The box was shared during the run, and the entry notes totals swing about 25%.</text>',
]

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
    f'width="{W}" height="{H}"><defs>{STYLE}</defs>{"".join(parts)}</svg>'
)
write("tpch_sf10", svg)
print("wrote tpch_sf10.svg")
