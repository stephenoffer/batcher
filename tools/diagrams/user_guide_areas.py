#!/usr/bin/env python3
"""Draw `user_guide_areas.svg` -- the five user-guide sections laid out along one pipeline.

Source of truth: `docs/user-guide/index.md` (its toctree names the five sections, and its
opening says the guide is organized by job: shape the data, analyze it, move it in and out,
trust what it says, run it in production), `docs/user-guide/moving-data/index.md` (`bt.read`
and `ds.write` are the two entry points), `docs/user-guide/trust/index.md` (checks and
policies run inside the query plan) and `docs/user-guide/operate/index.md` (`explain()`,
tuning, and keeping a pipeline healthy while it runs).

Move data appears twice because a pipeline enters and leaves through it. Trust and Operate
are drawn as bands across the chain rather than as steps in it, because neither is a stage a
row passes through: a check lowers into the same plan, and operating is about the whole run.
"""

from __future__ import annotations

from _authoring import arrow, band, label, note, svg, tint, write

W, H = 980, 420

ROW_Y, ROW_H = 156, 92
MID = ROW_Y + ROW_H / 2

body = [
    # Trust sits over the middle of the chain: it rewrites the plan the steps run in.
    band(247, 20, 486, 84, "TRUST", "amber"),
    note(490, 66, "Data-quality checks and row and column policies", anchor="middle"),
    note(490, 86, "lower into the same plan the steps below run in.", anchor="middle"),
    arrow(490, 104, 490, 150, "amber"),
    label(502, 132, "same plan", size=12),
    # The chain itself.
    tint(30, ROW_Y, 150, ROW_H, "Move data", "in: bt.read", "amber"),
    tint(256, ROW_Y, 190, ROW_H, "Transform", "which rows, what columns"),
    tint(534, ROW_Y, 190, ROW_H, "Analyze", "group, join, window, SQL"),
    tint(800, ROW_Y, 150, ROW_H, "Move data", "out: ds.write", "amber"),
    arrow(180, MID, 252, MID),
    label(216, MID - 14, "lazy", anchor="middle", size=11.5),
    arrow(446, MID, 530, MID),
    label(488, MID - 14, "rows kept", anchor="middle", size=11.5),
    arrow(724, MID, 796, MID),
    label(760, MID - 14, "answers", anchor="middle", size=11.5),
    # Operate spans the whole run.
    band(30, 304, 920, 96, "OPERATE", "grey"),
    note(
        490,
        346,
        "explain() shows the plan before it runs, and explain(analyze=True) measures it.",
        anchor="middle",
    ),
    note(
        490,
        368,
        "Tuning and caching make it fast. Progress events and metrics show a run as it happens.",
        anchor="middle",
    ),
    arrow(490, 304, 490, 254, "grey"),
    label(502, 284, "inspect and tune", size=12),
]

write("user_guide_areas", svg(W, H, "".join(body)))
print("wrote user_guide_areas.svg")
