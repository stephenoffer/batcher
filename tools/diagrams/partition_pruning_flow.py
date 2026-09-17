#!/usr/bin/env python3
"""Draw `partition_pruning_flow.svg`: how a Hive-partitioned table is pruned before tasks exist.

Source of truth: `docs/user-guide/operate/tuning/large-tables.md`, "Partition pruning happens
before the tasks exist" and "A join can prune too". The order drawn is the page's:

1. the driver enumerates only the top-level `day=` directories, in one non-recursive listing;
2. a filter on the partition column is applied to that directory list before the splits are
   built, exactly, because the directory name records the partition value;
3. each surviving directory goes to a worker, which lists only its own subtree;
4. splits, and so tasks, are built from what survived.

The predicate has two sources on the page: a `filter` you wrote, or dynamic partition pruning,
where the smaller join side's key range becomes a filter on the partition column. A directory
the predicate rules out is never listed, never opened, and never becomes a task. The 3,650 to 1
figure is the page's ten-years-of-days example.
"""

from __future__ import annotations

from _authoring import (
    FONT,
    arrow,
    band,
    card,
    label,
    note,
    step,
    svg,
    tint,
    write,
)

W, H = 980, 470

SW, SH = 180, 84
SY = 206
XS = (44, 280, 516, 752)  # four step cards, 56 px gutters
CX = [x + SW / 2 for x in XS]


def step_card(x: float, n: int, title: str, lines: tuple[str, str], kind: str = "blue") -> str:
    """A step: a card with a title, two short lines, and its number on the top-left corner."""
    out = (
        f'<g filter="url(#sh)"><rect x="{x}" y="{SY}" width="{SW}" height="{SH}" rx="10" '
        f'class="surface" stroke-width="1.2"/></g>'
        f'<text x="{x + SW / 2}" y="{SY + 32}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="13.5" font-weight="700" class="t-title">{title}</text>'
    )
    for i, line in enumerate(lines):
        out += (
            f'<text x="{x + SW / 2}" y="{SY + 53 + i * 16}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="11" class="t-sub">{line}</text>'
        )
    return out + step(x + 4, SY + 4, n, kind)


body: list[str] = [
    band(20, 20, 940, 430, "PLAN TIME, BEFORE ANY TASK EXISTS", "blue"),
    # ---- Where the predicate comes from ------------------------------------------------------
    tint(84, 66, 220, 58, "filter(day == ...)", "written by you"),
    tint(436, 66, 240, 58, "join on day", "small side's key range", "amber"),
    arrow(194, 126, CX[1] - 24, SY - 8),
    arrow(556, 126, CX[1] + 24, SY - 8, "amber"),
    label(250, 176, "predicate", anchor="end", size=11.5),
    label(482, 176, "dynamic pruning", size=11.5),
    note(706, 92, "Either becomes a filter on the"),
    note(706, 110, "directory list, not on rows."),
    # ---- The flow ---------------------------------------------------------------------------
    step_card(XS[0], 1, "list top level", ("driver: day= dirs only", "one cheap listing")),
    step_card(
        XS[1], 2, "prune the list", ("exact: the name records", "the partition value"), "amber"
    ),
    step_card(XS[2], 3, "list subtrees", ("one worker per survivor", "O(subtree) each")),
    step_card(XS[3], 4, "build splits", ("tasks come from", "survivors only")),
]

for i, text in enumerate(("dirs", "survivors", "files")):
    x1, x2 = XS[i] + SW + 4, XS[i + 1] - 8
    body.append(arrow(x1, SY + SH / 2, x2, SY + SH / 2))
    body.append(label((x1 + x2) / 2 - 2, SY + SH / 2 - 12, text, anchor="middle", size=10.5))

# ---- What pruning saves ---------------------------------------------------------------------
body += [
    arrow(CX[1], SY + SH + 4, CX[1], 348, "grey"),
    label(CX[1] + 12, 326, "ruled out", size=11.5),
    card(XS[1] - 70, 354, 320, 64, "never listed, opened, or tasked", "a pruned day= directory"),
    note(730, 372, "A directory per day for ten years:", anchor="middle"),
    note(730, 390, "3,650 tasks become one, same rows.", anchor="middle"),
    note(730, 414, "Where the layout can't decide, every", anchor="middle"),
    note(730, 432, "directory survives; rows are filtered.", anchor="middle"),
]

write("partition_pruning_flow", svg(W, H, "".join(body)))
print("wrote partition_pruning_flow.svg")
