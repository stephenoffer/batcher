#!/usr/bin/env python3
"""Draw `cv_folds.svg` -- a hashed k-fold matrix beside a chronological time-series split.

Source of truth: `python/batcher/ml/splitting.py` and the page
`docs/ml/evaluation/splits-and-resampling.md`. `fold_column` turns a content hash of each row
into a fold index in ``[0, k)``, and `kfold` builds pair *i* as ``fold != i`` for training and
``fold == i`` for validation, so every row validates exactly once. `stratified_kfold` deals
each label's rows round-robin across folds and `group_kfold` hashes the group value instead of
the row.

`time_series_split` is the exception the module docstring names: it cuts the time column at
``n_splits + 1`` quantiles. Split *i* trains on everything before cut *i* and validates on the
window between cut *i* and cut *i + 1*; with ``expanding=False`` the training set is only the
window before cut *i*. The right panel draws ``n_splits = 4``.
"""

from __future__ import annotations

from _authoring import FONT, arrow, band, label, note, svg, write

W, H = 980, 450

CELL_W, CELL_H, GAP = 66, 34, 6


def cell(x: float, y: float, w: float, text: str, kind: str) -> str:
    """One matrix cell. The word inside carries the meaning; the tint only repeats it."""
    if kind == "none":
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{CELL_H}" rx="7" class="band-grey" '
            f'stroke-width="1" stroke-dasharray="4 3"/>'
        )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{CELL_H}" rx="7" class="pill-{kind}"/>'
        f'<text x="{x + w / 2}" y="{y + CELL_H / 2 + 4}" text-anchor="middle" '
        f'font-family="{FONT}" font-size="11" font-weight="700" class="pt-{kind}">{text}</text>'
    )


body: list[str] = [
    band(20, 20, 470, 410, "K-FOLD  ·  HASHED, NOT SHUFFLED", "blue"),
    band(510, 20, 450, 410, "TIME SERIES  ·  SPLIT BY TIME", "amber"),
]

# Left: k = 5. Columns are fold indexes, rows are the (train, validate) pairs.
LX = 124
TOP = 102
for f in range(5):
    body.append(label(LX + f * (CELL_W + GAP) + CELL_W / 2, TOP - 14, f"fold {f}", "middle", 12))
for pair in range(5):
    y = TOP + pair * (CELL_H + GAP)
    body.append(label(44, y + CELL_H / 2 + 4, f"pair {pair}", size=12))
    for f in range(5):
        x = LX + f * (CELL_W + GAP)
        held_out = f == pair
        body.append(
            cell(x, y, CELL_W, "validate" if held_out else "train", "amber" if held_out else "blue")
        )

body += [
    note(44, 66, "hash(row) picks a fold index in [0, k)"),
    note(44, 324, "Each row validates exactly once, however the"),
    note(44, 342, "data is partitioned."),
    label(44, 380, "stratify=", size=12),
    note(116, 380, "deals each label round-robin across folds."),
    label(44, 408, "group=", size=12),
    note(100, 408, "hashes the group, so an entity stays in one fold."),
]

# Right: n_splits = 4 cuts the time range into five windows.
RX = 604
SEG = 62
body.append(note(534, 66, "Cut at quantiles of the time column"))
for s in range(5):
    body.append(label(RX + s * (SEG + GAP) + SEG / 2, TOP - 14, f"{s + 1}", "middle", 12))
body.append(label(534, TOP - 14, "window", size=12))


def ts_row(y: float, name: str, train: range, val: int) -> list[str]:
    """One split: training windows, the validation window, and the unused rest."""
    out = [label(534, y + CELL_H / 2 + 4, name, size=12)]
    for s in range(5):
        x = RX + s * (SEG + GAP)
        if s in train:
            out.append(cell(x, y, SEG, "train", "blue"))
        elif s == val:
            out.append(cell(x, y, SEG, "validate", "amber"))
        else:
            out.append(cell(x, y, SEG, "", "none"))
    return out


for i in range(4):
    body += ts_row(TOP + i * (CELL_H + GAP), f"split {i}", range(i + 1), i + 1)

AXIS_Y = TOP + 4 * (CELL_H + GAP) + 8
body += [
    arrow(RX, AXIS_Y, RX + 5 * (SEG + GAP) - GAP, AXIS_Y, "grey"),
    label(RX + 5 * (SEG + GAP) - GAP, AXIS_Y + 22, "time", anchor="end", size=12),
    note(534, AXIS_Y + 44, "expanding=True, the default, above. With"),
    note(534, AXIS_Y + 62, "expanding=False only the latest window trains:"),
]
body += ts_row(AXIS_Y + 80, "split 3", range(3, 4), 4)

write("cv_folds", svg(W, H, "".join(body)))
print("wrote cv_folds.svg")
