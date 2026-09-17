#!/usr/bin/env python3
"""Draw `sequence_packing.svg`: padding each document against packing them end to end.

Source of truth: the runnable examples in `docs/ml/preparing/tokenization.md` ("Sequence
packing for pretraining", "Telling the model where the documents end"). Three documents
of 3, 2 and 4 tokens, `[[1, 2, 3], [4, 5], [6, 7, 8, 9]]`, packed with `seq_len=4` and
`eos_token=0`, print `[[1, 2, 3, 0], [4, 5, 0, 6], [7, 8, 9, 0]]`, and with
`boundaries_column="seq_lens"` the segment lengths `[[4], [3, 1], [4]]`. The padded
side is the alternative the page argues against: each document padded up to `seq_len`,
drawn from the same three documents.

Layout: the input documents across the top, then the padded result and the packed result
side by side on identical grids, with the packed rows bracketed by their segment lengths.
"""

from __future__ import annotations

from _authoring import FONT, band, heading, label, note, svg, write

W, H = 980, 520

CELL, GAP = 42, 6
ROW_STEP = 84


def cell(x: float, y: float, value: str, style: str) -> str:
    """One token slot. `style` is doc-a, doc-b, eos or pad; the text says which, too."""
    rect_cls, text_cls, dash = {
        "doc-a": ("pill-blue", "pt-blue", ""),
        "doc-b": ("surface", "t-title", ""),
        "eos": ("pill-amber", "pt-amber", ""),
        "pad": ("band-grey", "t-sub", ' stroke-dasharray="4 3"'),
    }[style]
    size = 11 if style in ("eos", "pad") else 14
    return (
        f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="7" class="{rect_cls}" '
        f'stroke="#94a3b8" stroke-width="1.2"{dash}/>'
        f'<text x="{x + CELL / 2}" y="{y + CELL / 2 + 5}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="{size}" font-weight="700" '
        f'class="{text_cls}">{value}</text>'
    )


def tokens(x: float, y: float, row: list[tuple[str, str]]) -> str:
    """A row of cells starting at `x`."""
    return "".join(cell(x + i * (CELL + GAP), y, v, s) for i, (v, s) in enumerate(row))


def bracket(x1: float, x2: float, y: float, text: str) -> str:
    """A bracket under a run of cells, labeled with the run's length."""
    return (
        f'<path d="M {x1} {y} L {x1} {y + 7} L {x2} {y + 7} L {x2} {y}" fill="none" '
        f'stroke="#d97706" stroke-width="1.6"/>'
        + label((x1 + x2) / 2, y + 23, text, "middle", 11.5)
    )


def span(first: int, count: int, x0: float) -> tuple[float, float]:
    """The left and right edges of `count` cells starting at cell index `first`."""
    left = x0 + first * (CELL + GAP)
    return left + 2, left + count * CELL + (count - 1) * GAP - 2


A, B = "doc-a", "doc-b"

body = [
    # ---- Input -------------------------------------------------------------------
    band(16, 20, 948, 112, "INPUT  ·  THREE DOCUMENTS OF 3, 2 AND 4 TOKENS", "grey"),
    tokens(196, 60, [("1", A), ("2", A), ("3", A)]),
    note(196 + 69, 122, "doc 1", anchor="middle"),
    tokens(388, 60, [("4", B), ("5", B)]),
    note(388 + 45, 122, "doc 2", anchor="middle"),
    tokens(556, 60, [("6", A), ("7", A), ("8", A), ("9", A)]),
    note(556 + 93, 122, "doc 3", anchor="middle"),
    # ---- Padded ------------------------------------------------------------------
    band(16, 150, 466, 350, "PAD EACH DOCUMENT TO 4", "grey"),
    # ---- Packed ------------------------------------------------------------------
    band(498, 150, 466, 350, "pack_sequences(seq_len=4, eos_token=0)", "blue"),
]

LEFT_X, RIGHT_X, ROW_Y = 116, 554, 198

padded = (
    [("1", A), ("2", A), ("3", A), ("pad", "pad")],
    [("4", B), ("5", B), ("pad", "pad"), ("pad", "pad")],
    [("6", A), ("7", A), ("8", A), ("9", A)],
)
pad_counts = ("1 pad", "2 pad", "0 pad")
packed = (
    [("1", A), ("2", A), ("3", A), ("EOS", "eos")],
    [("4", B), ("5", B), ("EOS", "eos"), ("6", A)],
    [("7", A), ("8", A), ("9", A), ("EOS", "eos")],
)
segments = (((0, 4, "4"),), ((0, 3, "3"), (3, 1, "1")), ((0, 4, "4"),))
seq_lens = ("[4]", "[3, 1]", "[4]")

for r in range(3):
    y = ROW_Y + r * ROW_STEP
    body += [
        tokens(LEFT_X, y, padded[r]),
        note(LEFT_X + 4 * (CELL + GAP) + 24, y + CELL / 2 + 4, pad_counts[r]),
        tokens(RIGHT_X, y, packed[r]),
        heading(RIGHT_X + 4 * (CELL + GAP) + 24, y + CELL / 2 + 4, f"seq_lens {seq_lens[r]}"),
    ]
    for first, count, text in segments[r]:
        x1, x2 = span(first, count, RIGHT_X)
        body.append(bracket(x1, x2, y + CELL + 6, text))

body += [
    label(249, 470, "3 of 12 slots are padding", anchor="middle"),
    label(731, 470, "No padding. EOS marks each seam.", anchor="middle"),
]

write("sequence_packing", svg(W, H, "".join(body)))
print("wrote sequence_packing.svg")
