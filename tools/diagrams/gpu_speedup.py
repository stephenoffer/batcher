#!/usr/bin/env python3
"""Draw `gpu_speedup.svg` — GPU workload throughput, Batcher against Ray Data.

Every figure here is a committed measurement from `benchmarks/BENCHMARK_RESULTS.md`
(the per-workload sections around lines 3063-3136), run on 8xT4 with real models and
100% output agreement. Do not add a bar that no committed benchmark produces, and
re-check these when the suite is re-run.

Form: ranked magnitude, one series, so horizontal bars sorted descending with no
legend (the title names the series) and a direct label on every bar. The scale is
linear because the quantity *is* a ratio; a log scale would flatten exactly the
difference the chart exists to show. The 1x parity line anchors the reading.

Color: one hue, the project blue, stepped for each surface. Contrast measured rather
than eyeballed: 5.17:1 on white and 6.67:1 on the dark surface, with every text
element clearing 4.5:1 in both modes.
"""

from __future__ import annotations

from _authoring import FONT, write

# (label, speedup, note) — ordered as drawn, descending.
DATA = [
    ("Text embeddings", 47.0, "sentence-transformers MiniLM"),
    ("Audio feature extraction", 12.5, "torchaudio mel + ResNet-18"),
    ("LLM batch inference", 11.1, "HF gpt2, FP16"),
    ("Image generation", 8.6, "diffusion"),
    ("Training ingest", 3.0, "iter_torch_batches"),
    ("Batch inference", 2.05, "ResNet-50"),
]

W, H = 980, 486
LABEL_W = 268          # left column for category names
X0 = LABEL_W + 16      # plot origin
X1 = W - 108           # leave room for the value label outside the bar
PLOT_W = X1 - X0
BAR_H = 36
GAP = 16
TOP = 132
MAX = 50.0             # a round ceiling above the 47x maximum
PX = PLOT_W / MAX

STYLE = """<style>
  .surf { fill: #ffffff; }
  .bar { fill: #2563eb; }
  .t-head { fill: #1e293b; }
  .t-sub { fill: #5b6675; }
  .t-cat { fill: #1e293b; }
  .t-val { fill: #1e293b; }
  .rule { stroke: #e2e8f0; }
  .parity { stroke: #94a3b8; }
  @media (prefers-color-scheme: dark) {
    .surf { fill: #131c31; }
    .bar { fill: #60a5fa; }
    .t-head { fill: #e2e8f0; }
    .t-sub { fill: #94a3b8; }
    .t-cat { fill: #e2e8f0; }
    .t-val { fill: #e2e8f0; }
    .rule { stroke: #243049; }
    .parity { stroke: #64748b; }
  }
</style>"""


def bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """A bar anchored square at the baseline with only its data end rounded."""
    r = min(r, w)
    return (
        f"M {x} {y} H {x + w - r} Q {x + w} {y} {x + w} {y + r} "
        f"V {y + h - r} Q {x + w} {y + h} {x + w - r} {y + h} H {x} Z"
    )


parts = [
    f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" class="surf"/>',
    f'<text x="40" y="52" font-family="{FONT}" font-size="19" font-weight="700" class="t-head">'
    f"GPU throughput, Batcher against Ray Data</text>",
    f'<text x="40" y="76" font-family="{FONT}" font-size="13" class="t-sub">'
    f"8xT4, real models, 100% output agreement. Higher is faster.</text>",
]

# Gridlines every 10x, drawn under the bars and kept recessive.
for gv in range(10, int(MAX) + 1, 10):
    gx = X0 + gv * PX
    parts.append(
        f'<line x1="{gx}" y1="{TOP - 14}" x2="{gx}" y2="{TOP + len(DATA) * (BAR_H + GAP) - GAP + 6}" '
        f'class="rule" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{gx}" y="{TOP - 22}" text-anchor="middle" font-family="{FONT}" font-size="11" '
        f'class="t-sub">{gv}x</text>'
    )

# The parity line: where the two engines would be equal.
px = X0 + 1.0 * PX
parts.append(
    f'<line x1="{px}" y1="{TOP - 14}" x2="{px}" y2="{TOP + len(DATA) * (BAR_H + GAP) - GAP + 6}" '
    f'class="parity" stroke-width="1.5" stroke-dasharray="4 4"/>'
)
parts.append(
    f'<text x="{px}" y="{TOP - 22}" text-anchor="middle" font-family="{FONT}" font-size="11" '
    f'class="t-sub">1x</text>'
)

for i, (name, val, note) in enumerate(DATA):
    y = TOP + i * (BAR_H + GAP)
    w = val * PX
    parts.append(f'<path d="{bar_path(X0, y, w, BAR_H)}" class="bar"/>')
    parts.append(
        f'<text x="{LABEL_W}" y="{y + 20}" text-anchor="end" font-family="{FONT}" font-size="13.5" '
        f'font-weight="700" class="t-cat">{name}</text>'
    )
    parts.append(
        f'<text x="{LABEL_W}" y="{y + 35}" text-anchor="end" font-family="{FONT}" font-size="11" '
        f'class="t-sub">{note}</text>'
    )
    parts.append(
        f'<text x="{X0 + w + 12}" y="{y + BAR_H / 2 + 6}" font-family="{FONT}" font-size="15" '
        f'font-weight="700" class="t-val">{val:g}x</text>'
    )

parts.append(
    f'<text x="40" y="{H - 24}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f"Source: benchmarks/BENCHMARK_RESULTS.md. Correctness-gated: a timing is only "
    f"recorded once the outputs match.</text>"
)

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
    f'width="{W}" height="{H}"><defs>{STYLE}</defs>{"".join(parts)}</svg>'
)
write("gpu_speedup", svg)
print("wrote gpu_speedup.svg")
