#!/usr/bin/env python3
"""Draw `stage_overlap.svg` — what stage-overlapped execution bought, measured.

Figures from `benchmarks/BENCHMARK_RESULTS.md` (around line 2938): a two-stage
ResNet-50 pipeline went from 942 to 2504 img/s and GPU utilization from about 30%
to 81% when `execute_with_udfs` stopped running a `map_batches` chain stage-at-a-time
and started overlapping each stage on its own thread. The change is result-identical
to the materializing path, with order preserved.

Form: two measures on different scales, so this is **small multiples**, never a dual
axis. Each panel holds one measure with a before and an after bar, direct-labeled, so
no legend is needed: the panel title names the measure and the bars are labeled in place.

Color: before is a recessive slate, after is the project blue. The pair differs in
lightness as well as hue, so the comparison survives colorblindness and greyscale.
"""

from __future__ import annotations

from _authoring import FONT, write

W, H = 980, 340
PANEL_W = 440
PANEL_X = (40, 500)
BAR_W = 120
BASE_Y = 262          # bars grow upward from here
MAX_BAR = 132

STYLE = """<style>
  .surf { fill: #ffffff; }
  .panel { fill: #f8fafc; stroke: #e2e8f0; }
  .before { fill: #94a3b8; }
  .after { fill: #2563eb; }
  .t-head { fill: #1e293b; }
  .t-sub { fill: #5b6675; }
  .t-val { fill: #1e293b; }
  .base { stroke: #cbd5e1; }
  @media (prefers-color-scheme: dark) {
    .surf { fill: #131c31; }
    .panel { fill: #0f172a; stroke: #243049; }
    .before { fill: #64748b; }
    .after { fill: #60a5fa; }
    .t-head { fill: #e2e8f0; }
    .t-sub { fill: #94a3b8; }
    .t-val { fill: #e2e8f0; }
    .base { stroke: #334155; }
  }
</style>"""


def bar(x: float, h: float, cls: str, r: float = 4.0) -> str:
    """A column anchored square on the baseline with only its data end rounded."""
    y = BASE_Y - h
    r = min(r, h)
    return (
        f'<path d="M {x} {BASE_Y} V {y + r} Q {x} {y} {x + r} {y} '
        f"H {x + BAR_W - r} Q {x + BAR_W} {y} {x + BAR_W} {y + r} "
        f'V {BASE_Y} Z" class="{cls}"/>'
    )


def panel(px: float, title: str, unit: str, before: float, after: float,
          fmt: str, ceiling: float) -> list[str]:
    """One measure: a before bar and an after bar, each labeled in place."""
    out = [
        f'<rect x="{px}" y="96" width="{PANEL_W}" height="212" rx="12" class="panel" stroke-width="1"/>',
        f'<text x="{px + 24}" y="128" font-family="{FONT}" font-size="14.5" font-weight="700" '
        f'class="t-head">{title}</text>',
        f'<text x="{px + 24}" y="148" font-family="{FONT}" font-size="11.5" class="t-sub">{unit}</text>',
        f'<line x1="{px + 24}" y1="{BASE_Y}" x2="{px + PANEL_W - 24}" y2="{BASE_Y}" '
        f'class="base" stroke-width="1.5"/>',
    ]
    slots = ((px + 78, before, "before", "Stage at a time"), (px + 250, after, "after", "Overlapped"))
    for bx, val, cls, caption in slots:
        h = max(6.0, val / ceiling * MAX_BAR)
        out.append(bar(bx, h, cls))
        out.append(
            f'<text x="{bx + BAR_W / 2}" y="{BASE_Y - h - 12}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="16" font-weight="700" class="t-val">{fmt.format(val)}</text>'
        )
        out.append(
            f'<text x="{bx + BAR_W / 2}" y="{BASE_Y + 22}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11.5" class="t-sub">{caption}</text>'
        )
    return out


parts = [
    f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" class="surf"/>',
    f'<text x="40" y="52" font-family="{FONT}" font-size="19" font-weight="700" class="t-head">'
    f"Overlapping the CPU decode with the GPU forward</text>",
    f'<text x="40" y="76" font-family="{FONT}" font-size="13" class="t-sub">'
    f"Two-stage ResNet-50 pipeline. Same result, same order, one scheduling change.</text>",
]
parts += panel(PANEL_X[0], "Throughput", "images per second", 942, 2504, "{:,.0f}", 2600)
parts += panel(PANEL_X[1], "GPU utilization", "percent of the device kept busy", 30, 81, "{:.0f}%", 100)
parts.append(
    f'<text x="40" y="{H - 20}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f"Source: benchmarks/BENCHMARK_RESULTS.md. The device idled through the whole decode "
    f"until each stage got its own thread.</text>"
)

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
    f'width="{W}" height="{H}"><defs>{STYLE}</defs>{"".join(parts)}</svg>'
)
write("stage_overlap", svg)
print("wrote stage_overlap.svg")
