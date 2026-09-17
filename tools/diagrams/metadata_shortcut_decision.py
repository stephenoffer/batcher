#!/usr/bin/env python3
"""Draw `metadata_shortcut_decision.svg` -- when a question is answered from metadata, not a scan.

Source of truth: `docs/user-guide/analyze/metadata-shortcuts.md`. Kyber answers only from a
statistic whose provenance is exact (a footer bound, a manifest count, or an immutable
relation's own measurement); if the statistic is missing or merely estimated it declines and
the query that computes the answer runs instead, with an identical result. A filter, a join,
a computed column or `map_batches` turns a recorded minimum into a bound on the surviving
rows, and a bound is not an answer, so it takes the shortcut away; but a bound can still
refute a predicate outright (a comparison past the recorded maximum, key ranges that cannot
overlap), which is why a refuted filter leaves the files unread and a disjoint join does no
build, no probe and no shuffle. `ds.meta.approx` is the named exception: it never executes.

Drawn as a decision tree because the page's argument is that the outcome is decided by two
properties of the statistic, not by which API was called, and that every "no" lands on the
same answer at a higher cost.
"""

from __future__ import annotations

from _authoring import arrow, band, card, hero, label, note, svg, tint, write

W, H = 980, 560

A_X, B_X, COL_W = 96, 564, 320
A_MID, B_MID = A_X + COL_W / 2, B_X + COL_W / 2
Q_Y, D1_Y, D2_Y, OUT_Y = 34, 130, 256, 388
DH = 72

body: list[str] = [
    band(20, 14, 940, 460, "", "grey"),
    card(
        A_X + 30, Q_Y, COL_W - 60, 56, "a question about the data", "count, min, a filter, a join"
    ),
    arrow(A_MID, Q_Y + 56, A_MID, D1_Y - 2),
    label(A_MID + 12, Q_Y + 80, "any terminal", size=12),
    tint(
        A_X,
        D1_Y,
        COL_W,
        DH,
        "Is an exact statistic recorded?",
        "footer bound, manifest count, immutable relation",
    ),
    # D1: no -> scan
    arrow(A_X + COL_W, D1_Y + DH / 2, B_X - 2, D1_Y + DH / 2, "grey"),
    label((A_X + COL_W + B_X) / 2, D1_Y + DH / 2 - 12, "no", "middle", 12),
    note((A_X + COL_W + B_X) / 2, D1_Y + DH / 2 + 22, "or only estimated", "middle"),
    card(B_X, D1_Y, COL_W, DH, "run the query", "the same answer, at full cost"),
    # D1: yes -> D2
    arrow(A_MID, D1_Y + DH, A_MID, D2_Y - 2),
    label(A_MID + 12, D1_Y + DH + 34, "yes", size=12),
    tint(
        A_X,
        D2_Y,
        COL_W,
        DH,
        "Rows unchanged since measured?",
        "no filter, join, computed column or map_batches",
    ),
    # D2: yes -> answer
    arrow(A_MID, D2_Y + DH, A_MID, OUT_Y - 2),
    label(A_MID + 12, D2_Y + DH + 36, "yes", size=12),
    hero(A_X, OUT_Y, COL_W, 72, "answered from metadata", "the same value, no scan"),
    # D2: no -> D3
    arrow(A_X + COL_W, D2_Y + DH / 2, B_X - 2, D2_Y + DH / 2),
    label((A_X + COL_W + B_X) / 2, D2_Y + DH / 2 - 12, "no", "middle", 12),
    note((A_X + COL_W + B_X) / 2, D2_Y + DH / 2 + 22, "only a bound now", "middle"),
    tint(
        B_X,
        D2_Y,
        COL_W,
        DH,
        "Does a bound refute it?",
        "past the recorded max, disjoint key ranges",
    ),
    # D3: no -> scan (upwards)
    arrow(B_MID, D2_Y, B_MID, D1_Y + DH + 2, "grey"),
    label(B_MID + 12, D1_Y + DH + 34, "no: a bound is not an answer", size=12),
    # D3: yes -> provably empty
    arrow(B_MID, D2_Y + DH, B_MID, OUT_Y - 2, "amber"),
    label(B_MID + 12, D2_Y + DH + 36, "yes", size=12),
    tint(
        B_X,
        OUT_Y,
        COL_W,
        72,
        "provably empty",
        "a filter reads no files, a join no shuffle",
        "amber",
    ),
    note(
        490,
        504,
        "Which path ran is invisible: every branch returns what executing would return, "
        "and only the cost moves.",
        anchor="middle",
    ),
    note(
        490,
        526,
        "The one named exception is ds.meta.approx, which never executes and answers from a "
        "sketch or returns None.",
        anchor="middle",
    ),
]

write("metadata_shortcut_decision", svg(W, H, "".join(body)))
print("wrote metadata_shortcut_decision.svg")
