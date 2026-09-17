#!/usr/bin/env python3
"""Draw `corpus_prep_order.svg` -- the order a training corpus is prepared in.

Source of truth: the page `docs/ml/training/training-corpus.md`, section "The order to run
them in", and the functions it names in `batcher.ml`: `quality_filter`, `mix_corpora` (which
tags every row with a `source` column), `decontaminate` (verbatim span overlap, ``n`` of 13
tokens by default), and `length_grouped_order`.

The order is the page's: filter before mixing when the sources differ in quality, decontaminate
after mixing and filtering because the check only means something over the corpus you will
train on, and order last of all, after tokenization, which lives on the Tokenization page and
is drawn here without a step number for that reason.
"""

from __future__ import annotations

from _authoring import arrow, card, label, note, step, svg, tint, write

W, H = 980, 470

CW, CH = 240, 84
XS = (40, 370, 700)
R1, R2 = 44, 224

body: list[str] = [
    # Row 1, left to right.
    card(XS[0], R1, CW, CH, "Sources", "web, code, books"),
    tint(XS[1], R1, CW, CH, "Filter each source", "quality_filter"),
    tint(XS[2], R1, CW, CH, "Mix at weights", "mix_corpora adds source"),
    step(XS[1] + 4, R1 + 2, 1),
    step(XS[2] + 4, R1 + 2, 2),
    arrow(XS[0] + CW + 6, R1 + CH / 2, XS[1] - 8, R1 + CH / 2),
    label((XS[0] + CW + XS[1]) / 2, R1 + CH / 2 - 12, "raw text", "middle", 12),
    arrow(XS[1] + CW + 6, R1 + CH / 2, XS[2] - 8, R1 + CH / 2),
    label((XS[1] + CW + XS[2]) / 2, R1 + CH / 2 - 12, "prose kept", "middle", 12),
    note(XS[1] + CW / 2, R1 + CH + 24, "first, when source quality differs", anchor="middle"),
    # Down to row 2.
    arrow(XS[2] + CW / 2, R1 + CH + 8, XS[2] + CW / 2, R2 - 8),
    label(XS[2] + CW / 2 + 12, (R1 + CH + R2) / 2 + 4, "mixed corpus", size=12),
    # Row 2, right to left.
    tint(XS[2], R2, CW, CH, "Decontaminate", "n = 13 tokens by default", kind="amber"),
    step(XS[2] + 4, R2 + 2, 3, "amber"),
    card(XS[1], R2, CW, CH, "Tokenize", "on the Tokenization page"),
    tint(XS[0], R2, CW, CH, "Order", "length_grouped_order"),
    step(XS[0] + 4, R2 + 2, 4),
    arrow(XS[2] - 6, R2 + CH / 2, XS[1] + CW + 8, R2 + CH / 2),
    label((XS[1] + CW + XS[2]) / 2, R2 + CH / 2 - 12, "clean text", "middle", 12),
    arrow(XS[1] - 6, R2 + CH / 2, XS[0] + CW + 8, R2 + CH / 2),
    label((XS[0] + CW + XS[1]) / 2, R2 + CH / 2 - 12, "token ids", "middle", 12),
    # The evaluation sets feed the contamination check from below.
    card(XS[2], R2 + CH + 70, CW, 60, "Evaluation sets"),
    arrow(XS[2] + CW / 2, R2 + CH + 64, XS[2] + CW / 2, R2 + CH + 8, "amber"),
    label(XS[2] + CW / 2 + 12, R2 + CH + 42, "verbatim spans", size=12),
    # What comes out.
    note(XS[0] + CW / 2, R2 + CH + 24, "last: consume in this order", anchor="middle"),
    note(XS[0] + CW / 2, R2 + CH + 42, "and do not re-shuffle", anchor="middle"),
]

write("corpus_prep_order", svg(W, H, "".join(body)))
print("wrote corpus_prep_order.svg")
