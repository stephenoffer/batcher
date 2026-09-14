#!/usr/bin/env python3
"""Draw `dq_actions.svg` - one failing row, three outcomes.

Source of truth: `python/batcher/api/dataset/dq/accessor.py` (`fail`, `drop`,
`quarantine`, `annotate`) and `python/batcher/api/dataset/dq/apply.py`, which all four
terminals share so they cannot disagree about what a violation is.

Every constraint is a boolean `Expr` that is TRUE for a valid row, and validity is forced
to a non-null boolean, so the split is a total partition: valid and invalid together are
exactly the input. That is what lets `quarantine` promise a dead-letter side that loses
nothing. The chain lowers to FILTER, a keyless AGGREGATE, `count() OVER (PARTITION BY
keys)` and a LEFT JOIN - no new IR, so no separate distributed semantics.
"""

from __future__ import annotations

from _authoring import arrow, band, card, label, note, svg, write

W, H = 980, 610

body = [
    band(20, 20, 940, 96, "ONE CHAIN, ONE VALIDITY PREDICATE", "grey"),
    card(
        210,
        44,
        560,
        56,
        'ds.dq.not_null("id").in_range("age", 0, 120)',
        "each constraint is a boolean Expr, TRUE for a valid row",
    ),
    arrow(490, 116, 490, 152, "blue"),
    label(504, 140, "lowers to FILTER, count() OVER, LEFT JOIN", size=12),
    card(300, 152, 380, 64, "valid = every constraint", "non-null, so valid and invalid are the whole input"),
    # The three terminals.
    arrow(430, 218, 200, 300, "amber"),
    label(286, 250, "any violation", anchor="middle", size=12),
    arrow(490, 218, 490, 300, "blue"),
    label(504, 266, "the passing rows", size=12),
    arrow(550, 218, 780, 300, "blue"),
    label(700, 250, "both sides", anchor="middle", size=12),
    band(20, 288, 940, 302, "THREE TERMINALS, THREE OUTCOMES", "blue"),
    card(44, 318, 272, 70, "fail()", "the data-contract gate"),
    card(354, 318, 272, 70, "drop()", "keep the rows that pass"),
    card(664, 318, 272, 70, "quarantine()", "returns (clean, rejected)"),
    note(180, 414, "Output: none. It raises", anchor="middle"),
    note(180, 432, "DataQualityError, carrying a", anchor="middle"),
    note(180, 450, "count per constraint.", anchor="middle"),
    note(180, 474, "Dead-letter sink: nothing", anchor="middle"),
    note(180, 492, "is written. The run stops.", anchor="middle"),
    note(490, 414, "Output: the valid rows,", anchor="middle"),
    note(490, 432, "as one lazy Dataset.", anchor="middle"),
    note(490, 456, "Dead-letter sink: none.", anchor="middle"),
    note(490, 474, "The violating rows are gone", anchor="middle"),
    note(490, 492, "and nothing records them.", anchor="middle"),
    note(800, 414, "Output: the valid rows, and", anchor="middle"),
    note(800, 432, "the violating ones as a", anchor="middle"),
    note(800, 450, "second lazy Dataset.", anchor="middle"),
    note(800, 474, "Dead-letter sink: write that", anchor="middle"),
    note(800, 492, "second Dataset to it.", anchor="middle"),
    note(490, 532, "annotate() is the fourth: it keeps every row and names what each one failed, so a quarantined row carries its reason.", anchor="middle"),
    note(490, 560, "NULL is not a violation. mostly=0.99 passes while 1% violate, and severity='warn' reports without enforcing anywhere.", anchor="middle"),
]

write("dq_actions", svg(W, H, "".join(body)))
print("wrote dq_actions.svg")
