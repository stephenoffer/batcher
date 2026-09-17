#!/usr/bin/env python3
"""Draw `elastic_resume.svg` -- a mid-epoch resume on a smaller world size.

Source of truth: `python/batcher/ml/streaming_sampler/ordering.py` (`elastic_shard`,
`_rank_positions`, `usable_length`), `python/batcher/ml/streaming_sampler/resumable.py`
(`ResumableSampler.state_dict`), and the page `docs/ml/training/distributed-training.md`.

What the picture states, each from that code: the global order is a function of
``(seed, epoch)`` and not of ``world_size``; rank *r* strides the epoch positions
``r, r + world_size, ...`` starting from ``global_consumed``; a checkpoint taken between steps
records a ``global_consumed`` that is a multiple of ``world_size``; and on resume the strided
classes partition ``[global_consumed, usable)``, so nothing already consumed is repeated and
nothing is skipped, under a different ``world_size`` too. The page's example is 64 ranks
resuming on 32; the figure draws 4 resuming on 2 over 24 positions so every cell is readable.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, FONT, arrow, band, label, note, pill, svg, write

W, H = 980, 470

N = 24
CONSUMED = 12
CELL, GAP = 34, 3
X0 = 44 + (892 - N * (CELL + GAP) + GAP) / 2  # centre the strip inside the bands
CH = 34


def cell(i: int, y: float, text: str, kind: str) -> str:
    """One epoch position, labeled with the rank that reads it."""
    x = X0 + i * (CELL + GAP)
    if kind == "done":
        return (
            f'<rect x="{x}" y="{y}" width="{CELL}" height="{CH}" rx="6" class="band-grey" '
            f'stroke-width="1" stroke-dasharray="4 3"/>'
        )
    return (
        f'<rect x="{x}" y="{y}" width="{CELL}" height="{CH}" rx="6" class="pill-{kind}"/>'
        f'<text x="{x + CELL / 2}" y="{y + CH / 2 + 4}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="11" font-weight="700" class="pt-{kind}">{text}</text>'
    )


def cut(y0: float, y1: float) -> str:
    """The checkpoint position, drawn as a vertical rule through a strip."""
    x = X0 + CONSUMED * (CELL + GAP) - GAP / 2
    return (
        f'<path d="M {x} {y0} L {x} {y1}" stroke="{AMBER_DEEP}" stroke-width="2.4" '
        f'stroke-dasharray="6 4"/>'
    )


TOP_Y = 108
BOT_Y = 324
body: list[str] = [
    note(
        490,
        40,
        "One epoch's global order, computed from (seed, epoch) alone. "
        "Each cell is a position; its label is the rank that reads it.",
        anchor="middle",
    ),
    band(20, 58, 940, 144, "BEFORE THE CRASH  ·  WORLD_SIZE = 4", "blue"),
    band(20, 274, 940, 176, "RESUMED  ·  WORLD_SIZE = 2", "amber"),
]

for i in range(N):
    body.append(cell(i, TOP_Y, f"r{i % 4}", "blue" if i < CONSUMED else "grey"))
    if i < CONSUMED:
        body.append(cell(i, BOT_Y, "", "done"))
    else:
        body.append(cell(i, BOT_Y, f"r{i % 2}", "amber"))

mid = X0 + CONSUMED * (CELL + GAP) - GAP / 2
body += [
    cut(TOP_Y - 14, TOP_Y + CH + 14),
    cut(BOT_Y - 14, BOT_Y + CH + 14),
    label(
        X0 + CONSUMED * (CELL + GAP) / 2,
        TOP_Y + CH + 28,
        "read before the checkpoint",
        "middle",
        12,
    ),
    note(mid + 10, TOP_Y - 18, "checkpoint"),
    label(mid + (N - CONSUMED) * (CELL + GAP) / 2, TOP_Y + CH + 28, "not yet read", "middle", 12),
    # The hand-off between the two runs.
    arrow(mid, 210, mid, 268, "amber"),
    pill(mid + 14, 236, "state_dict(): global_consumed = 12", "amber"),
    note(mid + 14, 264, "taken between steps, so it is a multiple of 4"),
    label(
        X0 + CONSUMED * (CELL + GAP) / 2, BOT_Y + CH + 28, "consumed: never repeated", "middle", 12
    ),
    label(
        mid + (N - CONSUMED) * (CELL + GAP) / 2,
        BOT_Y + CH + 28,
        "strided over 2 ranks: none skipped",
        "middle",
        12,
    ),
    note(
        490,
        426,
        "Rank r reads positions r, r + world_size, ... from global_consumed, "
        "so the new ranks split the tail exactly.",
        anchor="middle",
    ),
]

write("elastic_resume", svg(W, H, "".join(body)))
print("wrote elastic_resume.svg")
