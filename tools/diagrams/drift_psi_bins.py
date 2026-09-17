#!/usr/bin/env python3
"""Draw `drift_psi_bins.svg` -- reference quantile edges reused on current data, then PSI.

Source of truth: `python/batcher/ml/stats/drift.py` (`_numeric_edges`, `_aligned_shares`,
`population_stability_index`) and the page `docs/ml/evaluation/statistics-and-drift.md`.

The numbers are the page's own example, computed with those functions rather than invented:
``train`` is ``x = 0..199``, ``today`` is the same shifted by 60, ``buckets=5``. The reference
edges come out at 39.8, 79.6, 119.4 and 159.2, every reference bin holds 20%, and the current
data lands 0%, 10%, 20%, 20% and 50% in those same bins. The empty bin is floored at 1e-6 so
the log ratio stays finite, and the PSI is 2.7854. The reading bands are the module docstring's.
"""

from __future__ import annotations

from _authoring import (
    AMBER_DEEP,
    BLUE_MID,
    FONT,
    GREY,
    arrow,
    band,
    code,
    label,
    mark,
    note,
    pill,
    svg,
    write,
)

W, H = 980, 440

BASE = 316  # the histograms' shared baseline
SCALE = 300  # pixels per share of 1.0
BIN = 46
EDGES = ("39.8", "79.6", "119.4", "159.2")
REF = (0.2, 0.2, 0.2, 0.2, 0.2)
CUR = (0.0, 0.1, 0.2, 0.2, 0.5)


def histogram(x0: float, shares: tuple[float, ...], color: str, ghost: bool) -> list[str]:
    """Five contiguous bars over the reference edges, each with its share printed above."""
    out: list[str] = []
    for i, share in enumerate(shares):
        x = x0 + i * BIN
        if ghost:
            out.append(
                f'<rect x="{x + 3}" y="{BASE - 0.2 * SCALE}" width="{BIN - 6}" '
                f'height="{0.2 * SCALE}" '
                f'fill="none" stroke="{GREY}" stroke-width="1.4" stroke-dasharray="4 3"/>'
            )
        h = share * SCALE
        if h:
            out.append(
                f'<rect x="{x + 3}" y="{BASE - h}" width="{BIN - 6}" height="{h}" rx="3" '
                f'fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="1.8"/>'
            )
        top = BASE - max(h, 0.2 * SCALE if ghost else h)
        out.append(label(x + BIN / 2, top - 8, f"{round(share * 100)}%", "middle", 11.5))
    out.append(
        f'<path d="M {x0 - 6} {BASE} L {x0 + 5 * BIN + 6} {BASE}" '
        f'stroke="{GREY}" stroke-width="1.5"/>'
    )
    # The interior edges, dashed through the chart, with their values beneath.
    for i, edge in enumerate(EDGES):
        ex = x0 + (i + 1) * BIN
        out.append(
            f'<path d="M {ex} {BASE - 190} L {ex} {BASE + 6}" stroke="{AMBER_DEEP}" '
            f'stroke-width="1.4" stroke-dasharray="5 4"/>'
        )
        out.append(
            f'<text x="{ex}" y="{BASE + 22}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="10.5" class="t-sub">{edge}</text>'
        )
    return out


AX, BX = 42, 382
body: list[str] = [
    band(20, 20, 280, 400, "1 · EDGES FROM REFERENCE", "blue"),
    band(360, 20, 280, 400, "2 · SAME EDGES, TODAY", "amber"),
    band(700, 20, 260, 400, "3 · SCORE THE SHIFT", "grey"),
]
body += histogram(AX, REF, BLUE_MID, ghost=False)
body += histogram(BX, CUR, AMBER_DEEP, ghost=True)
body += [
    label(44, 70, "reference: x = 0..199", size=12.5),
    note(44, 92, "buckets=5, cut at its quantiles"),
    label(384, 70, "today: x = 60..259", size=12.5),
    note(384, 92, "binned on the reference edges"),
    note(160, 364, "Quantile cuts: each bin holds 20%.", anchor="middle"),
    note(160, 384, "The outer bins are open-ended.", anchor="middle"),
    note(500, 364, "Mass moves between bins.", anchor="middle"),
    note(500, 384, "Dashed outline: the reference share.", anchor="middle"),
    # Between the panels.
    arrow(304, 200, 356, 200, "amber"),
    label(330, 186, "edges", "middle", 11.5),
    arrow(644, 200, 696, 200, "blue"),
    label(670, 186, "shares", "middle", 11.5),
    # The score.
    label(720, 70, "Summed over the five bins:", size=12),
    code(720, 84, ["sum((cur - ref)", "  * ln(cur / ref))"], 220, size=12),
    note(720, 172, "An empty bin counts as 1e-6."),
    label(720, 222, "PSI = 2.7854", size=16),
    note(720, 272, "Reading the number:"),
    pill(720, 308, "below 0.1: stable", "grey"),
    pill(720, 338, "0.1 to 0.25: moderate", "grey"),
    pill(720, 368, "above 0.25: significant", "amber"),
    mark(912, 363, True),
]

write("drift_psi_bins", svg(W, H, "".join(body)))
print("wrote drift_psi_bins.svg")
