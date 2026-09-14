#!/usr/bin/env python3
"""Draw `metrics_as_aggregates.svg` - why a metric is an aggregate, and when it is not.

Source of truth: `python/batcher/plan/functions/metrics/model/classification.py` (accuracy
and the four confusion counts, each a `count_if`) and `python/batcher/ml/metrics/ranked.py`
(ROC AUC, average precision, KS). The first is an `Expr` you put inside `agg()`; the second
is a Dataset function, because it is built on `rank()` and `cume_dist()` over the score.

The split is stated in both modules' docstrings as "a rank is not an aggregate", and it is
the whole point of the picture: the top row merges in any order, the bottom row needs a
sort first. Keep the two in step.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 490

body = [
    # The mergeable case.
    band(20, 20, 940, 200, "COMPOSES  -  ds.agg(m=bt.accuracy('y', 'p'))", "blue"),
    label(314, 74, "partial", anchor="middle", size=12),
    label(645, 74, "combine: add", anchor="middle", size=12),
    card(44, 86, 230, 84, "scored rows", "one partition per worker"),
    card(354, 86, 250, 84, "count_if(...)", "matched, compared"),
    card(694, 86, 242, 84, "accuracy", "matched / compared"),
    arrow(274, 128, 348, 128, "blue"),
    arrow(604, 128, 688, 128, "blue"),
    note(490, 196, "No row depends on another, so the partial counts merge in any order, on one core or a hundred.", anchor="middle"),
    # The case that needs a global ordering.
    band(20, 250, 940, 200, "DOES NOT COMPOSE  -  roc_auc(ds, 'y', 's')", "amber"),
    label(314, 304, "sort by score", anchor="middle", size=12),
    label(645, 304, "then one agg pass", anchor="middle", size=12),
    card(44, 316, 230, 84, "scored rows", "one partition per worker"),
    card(354, 316, 250, 84, "rank, cume_dist", "a window over every score"),
    card(694, 316, 242, 84, "ROC AUC", "the rank identity"),
    arrow(274, 358, 348, 358, "amber"),
    arrow(604, 358, 688, 358, "amber"),
    note(490, 426, "A rank depends on every other row, so this one adds a distributed sort. That is the whole difference.", anchor="middle"),
    note(490, 474, "Both take by= or group_by, so per-segment scoring is the same query with a grouping added.", anchor="middle"),
]

write("metrics_as_aggregates", svg(W, H, "".join(body)))
print("wrote metrics_as_aggregates.svg")
