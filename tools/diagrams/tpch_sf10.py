"""Draw `tpch_sf10.svg` -- the TPC-H scale-factor-10 standing on like-for-like input.

Every figure here is quoted from `benchmarks/BENCHMARK_RESULTS.md`, the entry dated
2026-08-28 ("The single-node board, re-measured against five engines"). Provenance, in
full, because the charts rule requires it:

* **Suite**: TPC-H, all 22 queries, **scale factor 10**, local mirror.
* **Hardware**: single node, 92 cores, best of 3, one process per suite.
* **Gate**: correctness-gated. A query whose result disagrees with the oracle produces no
  timing at all.

The two comparisons plotted are the like-for-like ones, where every engine reads the
same Arrow. The entry records them as `b/x` suite ratios:

    duckdb_arrow    0.33   (Batcher about 3.0x faster)
    polars          0.35   (Batcher about 2.9x faster)

The entry records no per-query win counts for this board, so the bars carry none. The same
board reads 1.10 against DuckDB's native compressed store at sf10, a loss this chart does
not plot because it is a storage comparison rather than an execution one; the docs that
embed the chart say so.

Form: bars anchored on a 1.0x parity line, so the magnitude of each margin is read
against a fixed reference rather than against each other. Bars are direct-labelled with
the speedup, so no legend is needed.
"""

from __future__ import annotations

from _authoring import FONT, write

W, H = 980, 334

AXIS_Y = 234  # baseline for the bars
MID_X = 500  # the 1.0x parity line
PX_PER_X = 120  # horizontal pixels per 1x of ratio
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
    f"TPC-H scale factor 10, all 22 queries</text>",
    f'<text x="40" y="74" font-family="{FONT}" font-size="13" class="t-sub">'
    f"92 cores, best of 3, correctness-gated. Bars show the suite speedup against each "
    f"engine on the same Arrow input.</text>",
    # the parity rule, drawn behind the bars
    f'<line x1="{MID_X}" y1="100" x2="{MID_X}" y2="{AXIS_Y + 4}" class="parity" '
    f'stroke-width="1.6" stroke-dasharray="5 4"/>',
    f'<text x="{MID_X}" y="{AXIS_Y + 24}" text-anchor="middle" font-family="{FONT}" '
    f'font-size="11.5" font-weight="700" class="t-sub">1.0x parity</text>',
    f'<text x="{MID_X + 14}" y="112" font-family="{FONT}" font-size="11.5" '
    f'class="t-sub">Batcher ahead &#8594;</text>',
]

parts += bar(
    ROW_Y[0],
    1 / 0.33,
    "DuckDB on the same Arrow",
    "like-for-like: identical zero-copy input",
    "suite ratio 0.33",
)
parts += bar(ROW_Y[1], 1 / 0.35, "Polars", "same Arrow input", "suite ratio 0.35")

parts += [
    f'<text x="40" y="{H - 44}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f"Source: benchmarks/BENCHMARK_RESULTS.md, 2026-08-28. Ratios are suite ratios of "
    f"Batcher's time over the other engine's.</text>",
    f'<text x="40" y="{H - 26}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f"Against DuckDB's own compressed store the same board reads 1.10, a storage "
    f"comparison this chart does not plot.</text>",
]

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
    f'width="{W}" height="{H}"><defs>{STYLE}</defs>{"".join(parts)}</svg>'
)
write("tpch_sf10", svg)
print("wrote tpch_sf10.svg")
