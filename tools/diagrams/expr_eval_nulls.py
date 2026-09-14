#!/usr/bin/env python3
"""Draw `expr_eval_nulls.svg` — the validity bitmap through one expression tree.

Source of truth:

* `crates/bc-arrow/src/lib.rs` — `pub type Morsel = RecordBatch`, the thing every
  expression is evaluated against, `DEFAULT_MORSEL_ROWS = 16_384`.
* `crates/bc-expr/src/eval/dispatch.rs::Expr::eval` — `(&RecordBatch) -> ArrayRef`,
  always a full-length column.
* `crates/bc-expr/src/eval/binary.rs` — comparisons go through
  `arrow::compute::kernels::cmp`, which returns **null**, not false, where an input is
  null, and nothing here overrides that. Arithmetic goes through `numeric::*_wrapping`
  and never touches a null buffer at all: the kernels combine validity.
* `binary.rs:428-434` — `And`/`Or` use `boolean::and_kleene` / `or_kleene`, with the
  comment that says why: "`FALSE AND NULL` is FALSE, `TRUE OR NULL` is TRUE (a known-
  controlling operand wins over an unknown). Arrow's plain `and`/`or` propagate the
  null instead."
* `crates/bc-expr/src/subset.rs:56::truthy` —
  `BooleanArray::new(mask.values() & nulls.inner(), None)`. This is where three values
  become two, and the `None` is the bitmap being dropped on purpose.

Worked example, chosen so the interesting row is visible rather than described: row 3
of `a` is null, so `a > 10` is unknown there, and `and_kleene` with a false operand
makes the result **valid again**. A reader watching the third validity bit go 1, 0, 1
has the whole rule.

Form: three stacked bands, each showing the same four rows as a values strip over a
validity strip, so the bitmap is drawn as a thing that travels rather than described as
one. The two exits at the bottom are the branch that matters: the same boolean column
means something different depending on whether it is kept or used to filter.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, MUTED, band, card, note, svg, write

W, H = 980, 600

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

CELL_W, CELL_H, GAP = 54, 30, 5
BIT_H = 22
STRIP_W = 4 * CELL_W + 3 * GAP  # 231

AX, BX = 120, 420  # left edge of the two column strips
LABEL_X = 112  # right-aligned row labels, left of the first strip


def strip(x: float, y: float, cells: list[str | None]) -> str:
    """Four value slots. `None` is a null slot, which still occupies a slot."""
    out = []
    for i, text in enumerate(cells):
        cx = x + i * (CELL_W + GAP)
        klass = "band-grey" if text is None else "surface"
        shown = "null" if text is None else text
        fill = f' fill="{MUTED}"' if text is None else ""
        cls_t = "" if text is None else ' class="t-title"'
        out.append(
            f'<rect x="{cx}" y="{y}" width="{CELL_W}" height="{CELL_H}" rx="5" '
            f'class="{klass}" stroke-width="1.2"/>'
            f'<text x="{cx + CELL_W / 2}" y="{y + CELL_H / 2 + 4.5}" text-anchor="middle" '
            f'font-family="{FONT}" font-size="12" font-weight="700"{cls_t}{fill}>{shown}</text>'
        )
    return "".join(out)


def bits(x: float, y: float, valid: list[int]) -> str:
    """The validity bitmap beside the values: one bit per slot, drawn as a digit."""
    out = []
    for i, v in enumerate(valid):
        cx = x + i * (CELL_W + GAP)
        klass = "band-blue" if v else "band-amber"
        out.append(
            f'<rect x="{cx}" y="{y}" width="{CELL_W}" height="{BIT_H}" rx="4" '
            f'class="{klass}" stroke-width="1.2"/>'
            f'<text x="{cx + CELL_W / 2}" y="{y + BIT_H / 2 + 4}" text-anchor="middle" '
            f'font-family="{MONO}" font-size="11.5" font-weight="700" '
            f'fill="{BLUE if v else AMBER_DEEP}">{v}</text>'
        )
    return "".join(out)


def rowlabel(y: float, text: str) -> str:
    return (
        f'<text x="{LABEL_X}" y="{y}" text-anchor="end" font-family="{FONT}" font-size="11.5" '
        f'class="t-sub">{text}</text>'
    )


def heading(x: float, y: float, text: str) -> str:
    return (
        f'<text x="{x + STRIP_W / 2}" y="{y}" text-anchor="middle" font-family="{MONO}" '
        f'font-size="12.5" font-weight="700" fill="{BLUE}">{text}</text>'
    )


body: list[str] = []

# ---- Band 1: the inputs ---------------------------------------------------
body += [
    band(20, 20, 940, 150, "AN ARROW COLUMN IS TWO BUFFERS, NOT ONE", "grey"),
    heading(AX, 72, "column a"),
    strip(AX, 80, ["3", "17", None, "24"]),
    bits(AX, 116, [1, 1, 0, 1]),
    heading(BX, 72, "column b"),
    strip(BX, 80, ["9", "2", "9", "8"]),
    bits(BX, 116, [1, 1, 1, 1]),
    rowlabel(100, "values"),
    rowlabel(132, "validity"),
    note(684, 86, "A null slot still holds a payload."),
    note(684, 104, "The bitmap beside it is the only thing"),
    note(684, 122, "that says to ignore that payload."),
]

# ---- Band 2: comparison -------------------------------------------------
body += [
    band(20, 186, 940, 150, "COMPARISON CARRIES THE BITMAP FORWARD", "blue"),
    heading(AX, 238, "a &gt; 10"),
    strip(AX, 246, ["false", "true", None, "true"]),
    bits(AX, 282, [1, 1, 0, 1]),
    heading(BX, 238, "b &lt; 5"),
    strip(BX, 246, ["false", "true", "false", "false"]),
    bits(BX, 282, [1, 1, 1, 1]),
    rowlabel(266, "values"),
    rowlabel(298, "validity"),
    note(684, 252, "Arrow's compare kernels return null,"),
    note(684, 270, "never false, where an input is null."),
    note(684, 288, "Row 3 is unknown, not excluded."),
]

# ---- Band 3: the AND, and the two things it can be used for --------------
body += [
    band(20, 352, 940, 232, "AND IS THREE-VALUED, UNTIL IT IS USED", "amber"),
    heading(AX, 404, "and_kleene"),
    strip(AX, 412, ["false", "true", "false", "false"]),
    bits(AX, 448, [1, 1, 1, 1]),
    rowlabel(432, "values"),
    rowlabel(464, "validity"),
    note(AX, 498, "false AND null is false, so row 3 is"),
    note(AX, 516, "valid again. Arrow's plain and would"),
    note(AX, 534, "have propagated the null instead."),
]

# The two exits.
body += [
    card(420, 396, 500, 76, "Kept as a column", "three values survive: true, false, unknown"),
    card(
        420, 492, 500, 76, "Used as a filter", "truthy() folds unknown to false, and the row goes"
    ),
    f'<path d="M {AX + STRIP_W + 10} 427 L 410 434" fill="none" stroke="{BLUE}" '
    f'stroke-width="2.4" marker-end="url(#arB)"/>',
    f'<path d="M {AX + STRIP_W + 10} 455 L 410 530" fill="none" stroke="{AMBER_DEEP}" '
    f'stroke-width="2.4" marker-end="url(#arA)"/>',
    f'<text x="378" y="418" text-anchor="end" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" class="t-arrow">as a value</text>',
    f'<text x="368" y="478" text-anchor="end" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" fill="{AMBER_DEEP}">as a predicate</text>',
    note(670, 484, "truthy(): value AND validity, and no bitmap on the result.", anchor="middle"),
]

write("expr_eval_nulls", svg(W, H, "".join(body)))
print("wrote expr_eval_nulls.svg")
