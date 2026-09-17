#!/usr/bin/env python3
"""Draw `embedding_compaction.svg`: normalize, binarize and shrink an embedding, side by side.

Source of truth: `docs/ml/retrieval/embeddings.md` ("Normalize once, at write time",
"Binarize for cheap Hamming search", "Shrink and screen the vectors") and
`python/batcher/api/dataset/ml.py` (`truncate_embeddings`, `binarize_embeddings`).
Normalizing makes every vector unit length, so a dot product ranks exactly as cosine
does. `binarize_embeddings` maps each dimension to 1 when it is above zero and 0
otherwise, ranked with `metric="hamming"`, at a small recall loss.
`truncate_embeddings(column, dim)` keeps the leading `dim` values and re-normalizes by
default, for Matryoshka-trained models, at a small recall cost. All three are native
list expressions with no per-row Python.

Layout: a matrix, one row per option and one column per question a reader asks.
The last column pairs a mark with words, so the verdict never rests on color.
"""

from __future__ import annotations

from _authoring import MONO, band, heading, mark, note, svg, tint, write

W, H = 980, 482

COLS = (36, 192, 432, 624, 792)  # left edge of each column
HEAD_Y = 80
ROW_Y0 = 98
ROW_H = 98
ROW_GAP = 12

HEADS = ("OPTION", "CALL", "RESULT", "SEARCH WITH", "RECALL")

ROWS = (
    (
        "Normalize",
        ("normalize=True", ".list.normalize()"),
        ("unit length,", "same dimensions"),
        ("dot product,", "ranks like cosine"),
        (True, "no loss vs cosine"),
    ),
    (
        "Binarize",
        ("ml.binarize_embeddings", "(col)"),
        ("one 0 or 1 per", "dimension, by sign"),
        ("Hamming distance,", 'metric="hamming"'),
        (False, "small recall loss"),
    ),
    (
        "Shrink",
        ("ml.truncate_embeddings", "(col, dim)"),
        ("first dim values,", "re-normalized"),
        ("a smaller, faster", "index"),
        (False, "small recall cost"),
    ),
)


def mono(x: float, y: float, text: str) -> str:
    """A line of code in a table cell."""
    return (
        f'<text x="{x}" y="{y}" font-family="{MONO}" font-size="12.5" class="t-code">{text}</text>'
    )


body = [band(16, 20, 948, 402, "THREE WAYS TO MAKE A VECTOR CHEAPER", "blue")]
for i, head in enumerate(HEADS):
    body.append(
        heading(COLS[i] + (0 if i == 0 else 16 if i == 1 else 6), HEAD_Y, head, kind="grey")
    )

for r, (name, call, becomes, search, (ok, verdict)) in enumerate(ROWS):
    y = ROW_Y0 + r * (ROW_H + ROW_GAP)
    mid = y + ROW_H / 2
    body += [
        f'<rect x="{COLS[1]}" y="{y}" width="{948 + 16 - COLS[1] - 20}" height="{ROW_H}" '
        'rx="10" class="surface" stroke-width="1"/>',
        tint(COLS[0], y, 140, ROW_H, name),
        mono(COLS[1] + 16, mid - 4, call[0]),
        mono(COLS[1] + 16, mid + 16, call[1]),
        note(COLS[2] + 6, mid - 4, becomes[0]),
        note(COLS[2] + 6, mid + 15, becomes[1]),
        note(COLS[3] + 6, mid - 4, search[0]),
        note(COLS[3] + 6, mid + 15, search[1]),
        mark(COLS[4] + 16, mid - 1, ok),
        note(COLS[4] + 34, mid + 3, verdict),
    ]

body.append(
    note(
        490,
        452,
        "Shrink only a Matryoshka-trained model, such as text-embedding-3-*, Nomic or mxbai.",
        anchor="middle",
    )
)

write("embedding_compaction", svg(W, H, "".join(body)))
print("wrote embedding_compaction.svg")
