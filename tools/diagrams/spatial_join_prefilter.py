#!/usr/bin/env python3
"""Draw `spatial_join_prefilter.svg` -- a bounding-box filter in front of the exact predicate.

Source of truth: `docs/user-guide/analyze/geospatial.md`, section "Spatial joins, and the
filter that makes them affordable". `st_intersects_extent` compares four numbers and is
exact in the negative direction, so a false means the geometries certainly do not intersect:
it yields false positives the exact test removes and never false negatives. `st_intersects`
decodes both geometries and walks their segments. The counts are the page's own example:
three points crossed with two regions is six pairs; (1 1) falls in the west box, (7 3) in
the east box, and (20 20) in neither, so the extent test keeps two pairs and both survive
`st_intersects`, giving `{'pid': [1, 2], 'region': ['west', 'east']}`. The footer is the
page's tip: `st_xmin`/`st_ymin`/`st_xmax`/`st_ymax` are plain Float64, so a range predicate
on them pushes down to the scan and to Parquet statistics.

Drawn as a flow with two exits because the point is where rows leave, and which test is
allowed to be wrong in which direction.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, tint, write

W, H = 980, 452

ROW = 80
RH = 76
MID = ROW + RH / 2

body: list[str] = [
    band(20, 20, 940, 318, "ONE SPATIAL JOIN, TWO FILTERS, CHEAPEST FIRST", "blue"),
    card(40, ROW, 150, RH, "cross join", "3 points, 2 regions"),
    arrow(190, MID, 250, MID),
    label(220, MID - 12, "6 pairs", anchor="middle", size=12),
    tint(254, ROW, 206, RH, "st_intersects_extent", "compares 4 numbers"),
    arrow(460, MID, 528, MID),
    label(494, MID - 12, "true: 2", anchor="middle", size=12),
    tint(532, ROW, 190, RH, "st_intersects", "decodes, walks segments"),
    arrow(722, MID, 780, MID),
    label(751, MID - 12, "true: 2", anchor="middle", size=12),
    card(784, ROW, 158, RH, "2 hits", "pid 1 west, pid 2 east"),
    # Exits.
    arrow(357, ROW + RH, 357, 214, "amber"),
    label(369, 190, "false: 4"),
    card(254, 218, 206, 70, "certainly disjoint", "never a false negative"),
    note(357, 310, "pid 3 with both, 1 east, 2 west", anchor="middle"),
    arrow(627, ROW + RH, 627, 214, "grey"),
    label(639, 190, "false: 0 here"),
    card(532, 218, 190, 70, "false positives", "removed by the exact test"),
    band(20, 354, 940, 82, "FASTER STILL", "grey"),
    note(
        38,
        398,
        "Materialize st_xmin, st_ymin, st_xmax and st_ymax once, beside the geometry.",
    ),
    note(
        38,
        418,
        "They are plain Float64, so a range predicate on them pushes down to the scan and to "
        "Parquet statistics.",
    ),
]

write("spatial_join_prefilter", svg(W, H, "".join(body)))
print("wrote spatial_join_prefilter.svg")
