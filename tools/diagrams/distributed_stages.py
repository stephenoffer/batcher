#!/usr/bin/env python3
"""Draw `distributed_stages.svg` -- where a distributed query is cut, and what crosses a cut.

Source of truth, read before drawing:
  * `python/batcher/dist/executors/plan_analysis.py` -- `_has_breaker` (Aggregate, Sort,
    Join, Distinct, Limit), `_is_row_wise` / `is_partition_independent`, and `_split_at`,
    which walks down pass-through nodes to the first breaker. That list *is* the cut set.
  * `python/batcher/dist/executors/ray_runtime/policies/_barrier.py::map_barrier` -- the
    barrier keeps exactly `workers` tasks in flight and hands each new source to whichever
    actor just went idle, so the assignment is dynamic rather than dealt up front.
  * `python/batcher/dist/executors/ray_runtime/reducers.py::map_partitions` -- the input is
    cut into `workers x distributed.map_partition_multiplier` partitions (4x by default).
  * `python/batcher/dist/executor.py` -- `materialize=False` lets a stage keep its result
    partitioned and return handles; `_staged_aggregate_over_join` and
    `_staged_aggregate_over_distinct` are the two-stage shapes built on that.

The claim the picture carries, and the reason it is worth a figure: **a stage boundary is a
plan property, not a cluster property.** The cut set is the breaker list above, decided from
the plan alone, and what crosses a cut is a list of handles rather than rows. A reader who
takes "distributed" to mean "the driver gathers and re-scatters" has the wrong model, and
that is the model the middle band exists to displace.

Layout: three zoom levels -- cut the plan, run one stage, then what crosses to the next.
"""

from __future__ import annotations

from _authoring import BLUE_MID, GREY, arrow, band, card, label, note, svg, write

W, H = 980, 620


def chip(x: float, y: float, w: float, h: float, kind: str = "blue") -> str:
    """A small data rectangle: a partition, a bucket. Not a card -- no title."""
    fill = {"blue": BLUE_MID, "grey": GREY}[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
        f'fill-opacity="0.28" stroke="{fill}" stroke-width="1.4"/>'
    )


def cut(x: float, y1: float, y2: float) -> str:
    """A stage boundary drawn through the plan: the breaker is where the cut falls."""
    return (
        f'<path d="M {x} {y1} L {x} {y2}" fill="none" stroke="#d97706" stroke-width="2.4" '
        f'stroke-dasharray="5 5"/>'
    )


body: list[str] = []

# ---- Zoom 1: the cut set is a plan property ----------------------------------------
body.append(band(20, 24, 940, 142, "CUT THE PLAN AT ITS PIPELINE BREAKERS", "grey"))
body.append(note(44, 62, "the cut set is decided from the plan alone (plan_analysis._has_breaker); "
                         "the cluster does not enter into it"))

body.append(card(44, 76, 230, 62, "scan / filter / project", "row-wise: no cut"))
body.append(arrow(278, 107, 326, 107))
body.append(label(302, 94, "rows", anchor="middle", size=11.5))
body.append(card(332, 76, 190, 62, "aggregate", "BREAKER"))
body.append(arrow(526, 107, 574, 107))
body.append(label(550, 94, "groups", anchor="middle", size=11.5))
body.append(card(580, 76, 190, 62, "sort", "BREAKER"))
body.append(arrow(774, 107, 822, 107))
body.append(label(798, 94, "rows", anchor="middle", size=11.5))
body.append(card(828, 76, 110, 62, "limit", "BREAKER"))

for x in (328, 576, 824):
    body.append(cut(x, 70, 150))
body.append(note(452, 162, "Aggregate, Sort, Join, Distinct, Limit -- every other node "
                           "runs inside the stage it is in", anchor="middle"))

# ---- Zoom 2: one stage ---------------------------------------------------------------
body.append(arrow(490, 172, 490, 208))
body.append(label(504, 196, "one cut = one stage", size=11.5))

body.append(band(20, 214, 940, 216, "ONE STAGE: MAP TASKS, A BARRIER, REDUCE TASKS", "blue"))

body.append(note(150, 250, "workers x 4 partitions", anchor="middle"))
x = 44
for _ in range(4):
    body.append(chip(x, 262, 48, 96))
    x += 54
body.append(note(150, 378, "each a durable descriptor:", anchor="middle"))
body.append(note(150, 394, "splits + pushed projection", anchor="middle"))

body.append(arrow(266, 306, 322, 306))
body.append(label(294, 292, "dealt to", anchor="middle", size=11.5))
body.append(label(294, 326, "an idle actor", anchor="middle", size=11.5))

body.append(card(328, 262, 214, 96, "map tasks", "partial_aggregate"))
body.append(note(435, 378, "exactly `workers` in flight,", anchor="middle"))
body.append(note(435, 394, "the slow node takes fewer", anchor="middle"))

body.append(arrow(546, 306, 618, 306, "amber"))
body.append(label(582, 292, "one bucket", anchor="middle", size=11.5))
body.append(label(582, 326, "per reducer", anchor="middle", size=11.5))

body.append(card(624, 262, 214, 96, "reduce tasks", "combine / combine_finalize"))
body.append(note(731, 378, "a bucket is reduced by the", anchor="middle"))
body.append(note(731, 394, "one worker it hashes to", anchor="middle"))

body.append(note(896, 300, "BARRIER", anchor="middle"))
body.append(cut(584, 250, 370))

# ---- Zoom 3: what crosses a cut ------------------------------------------------------
body.append(arrow(490, 436, 490, 470))
body.append(label(504, 460, "and then?", size=11.5))

body.append(band(20, 476, 940, 118, "WHAT CROSSES TO THE NEXT STAGE", "amber"))
body.append(card(44, 508, 268, 66, "handles", "one per reducer bucket"))
body.append(arrow(316, 541, 386, 541))
body.append(label(351, 528, "scanned in place", anchor="middle", size=11.5))
body.append(card(392, 508, 268, 66, "the next stage", "an ordinary scan of them"))
body.append(note(680, 526, "The rows stay on the worker that computed them."))
body.append(note(680, 544, "A multi-join query never round-trips an"))
body.append(note(680, 562, "intermediate through the driver."))

write("distributed_stages", svg(W, H, "".join(body)))
print("wrote distributed_stages.svg")
