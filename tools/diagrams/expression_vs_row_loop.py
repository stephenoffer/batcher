#!/usr/bin/env python3
"""Draw `expression_vs_row_loop.svg`: a Python per-row loop against a Batcher expression.

Source of truth: `docs/getting-started/concepts/expressions.md`. An `Expr` built from
`bt.col` and `bt.lit` is a description of a computation. Python ships the expression tree
to the engine as part of the plan, the Rust data plane evaluates it over whole Arrow
batches with vectorized kernels, numeric filters and projections compile with Cranelift
(falling back to the interpreter otherwise), and the optimizer can push a filter into a
scan, drop an unread column, or fold a constant. A Python lambda is opaque to the
optimizer. The left column is the contrast the page draws, not something Batcher does.

Layout: two columns of equal width whose rows line up stage for stage, so the reader
compares across: what you write, what the plan holds, what the optimizer sees, what runs.
"""

from __future__ import annotations

from _authoring import arrow, band, card, code, hero, label, note, svg, write

W, H = 980, 600

LX, RX = 250, 730  # column centre lines

body = [
    # --- left: the per-row loop ---------------------------------------------------
    band(20, 20, 460, 560, "A PYTHON LOOP OVER ROWS", "grey"),
    code(52, 62, ["for row in rows:", '    out.append(row["x"] * 10)'], 396, size=13),
    arrow(LX, 146, LX, 184, "grey"),
    label(LX + 14, 170, "becomes"),
    card(64, 190, 372, 70, "Python function", "a loop, not a description"),
    arrow(LX, 260, LX, 304, "grey"),
    label(LX + 14, 287, "opaque to the plan"),
    card(64, 310, 372, 70, "Optimizer", "can't see inside it"),
    arrow(LX, 380, LX, 424, "grey"),
    label(LX + 14, 407, "runs in Python"),
    card(64, 430, 372, 86, "Python interpreter", "called once per row"),
    note(LX, 552, "Nothing to push down, prune, or fold.", anchor="middle"),
    # --- right: the expression --------------------------------------------------------
    band(500, 20, 460, 560, "AN EXPRESSION", "blue"),
    code(532, 62, ['total = bt.col("x") * bt.lit(10)', "ds.select(scaled=total)"], 396, size=13),
    arrow(RX, 146, RX, 184),
    label(RX + 14, 170, "builds"),
    card(544, 190, 372, 70, "Expression tree", "a description, not a loop"),
    arrow(RX, 260, RX, 304),
    label(RX + 14, 287, "shipped in the plan"),
    card(544, 310, 372, 70, "Optimizer", "pushes filters, prunes, folds"),
    arrow(RX, 380, RX, 424, "amber"),
    label(RX + 14, 407, "runs in Rust"),
    hero(
        544, 430, 372, 86, "Whole Arrow batches", "vectorized kernels, Cranelift for numeric work"
    ),
    note(RX, 552, "No part of it walks rows in Python.", anchor="middle"),
]

write("expression_vs_row_loop", svg(W, H, "".join(body)))
print("wrote expression_vs_row_loop.svg")
