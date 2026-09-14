#!/usr/bin/env python3
"""Draw `ir_wire_contract.svg` — the JSON IR as one contract with two statements.

Sources of truth, all three of which must stay in step with this picture:

* `python/batcher/plan/ir_tags.py` — the Python side's tag *strings*, held in one
  module (`Op`, `ExprTag`) rather than scattered as literals across the `to_ir()`
  methods, so a typo is an `AttributeError` rather than a wrong tag.
* `crates/bc-ir/src/lib.rs` — `#[serde(tag = "op", rename_all = "snake_case",
  deny_unknown_fields)]` on `RelOp`; `crates/bc-expr/src/lib.rs` carries the same
  attributes on `Expr` with `tag = "e"`.
* `tools/lint_ir_contract.py` — the checker. It derives the tags serde will accept
  straight from the enum bodies and compares them against the Python vocabularies,
  reporting the two directions separately: `Python emits, Rust rejects` and
  `Rust accepts, Python never emits`.

The counts in the diagram are that checker's own output (`python
tools/lint_ir_contract.py`); re-run it rather than trusting them if the vocabularies
have grown.

The diagram's job is the *asymmetry* at the bottom, which prose keeps flattening into
"they must agree". The two drift directions fail in completely different ways, and the
reason a checker exists at all is that the reconciliation it replaced — round-trip and
differential tests — only ever covers a tag some test happens to name.

Form: one horizontal spine (Python -> JSON -> Rust) over a checker that reaches up into
both ends, with the two failure directions drawn as separate outcomes rather than as one
"mismatch" box.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, GREY, MUTED, band, card, label, note, svg, write

W, H = 980, 560

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

ROW_Y = 92
ROW_H = 104
LEFT_X, MID_X, RIGHT_X = 44, 380, 716
LEFT_W, MID_W, RIGHT_W = 268, 220, 220


def mono(x: float, y: float, text: str, anchor: str = "middle", size: float = 11.5) -> str:
    """A code-voiced line, for a tag string or a type path."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{MONO}" '
        f'font-size="{size}" fill="{BLUE}">{text}</text>'
    )


body = [
    band(20, 20, 940, 244, "ONE DOCUMENT, WRITTEN ONCE AND READ ONCE", "grey"),
    card(LEFT_X, ROW_Y, LEFT_W, ROW_H, "Python to_ir()", "control plane"),
    mono(LEFT_X + LEFT_W / 2, ROW_Y + ROW_H - 10, "plan/ir_tags.py"),
    card(MID_X, ROW_Y, MID_W, ROW_H, "JSON IR", "the wire"),
    mono(MID_X + MID_W / 2, ROW_Y + ROW_H - 10, '{"op": "hash_join"}'),
    card(RIGHT_X, ROW_Y, RIGHT_W, ROW_H, "Rust serde", "data plane"),
    mono(RIGHT_X + RIGHT_W / 2, ROW_Y + ROW_H - 10, "bc_ir::RelOp"),
    # The spine. Both edges carry what the step actually does, not "flows to".
    label(LEFT_X + LEFT_W + 34, ROW_Y + 32, "writes the tag", anchor="middle"),
    note(LEFT_X + LEFT_W + 34, ROW_Y + 50, "string", anchor="middle"),
    label(MID_X + MID_W + 34, ROW_Y + 32, "must accept it", anchor="middle"),
    note(MID_X + MID_W + 34, ROW_Y + 50, "deny_unknown_fields", anchor="middle"),
    note(490, 236, "16 RelOp tags and 55 Expr tags today, plus 19 function vocabularies underneath them.",
         anchor="middle"),
]

# Arrows drawn after the labels so the heads sit above the band fill.
body += [
    f'<path d="M {LEFT_X + LEFT_W + 8} {ROW_Y + ROW_H / 2} L {MID_X - 10} {ROW_Y + ROW_H / 2}" '
    f'fill="none" stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arB)"/>',
    f'<path d="M {MID_X + MID_W + 8} {ROW_Y + ROW_H / 2} L {RIGHT_X - 10} {ROW_Y + ROW_H / 2}" '
    f'fill="none" stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arB)"/>',
]

# ---- The checker, reaching up into both ends ------------------------------
CHK_Y = 288
body += [
    band(20, 276, 940, 264, "WHAT DRIFT LOOKS LIKE, AND WHO NOTICES", "amber"),
    card(340, CHK_Y + 16, 300, 76, "lint_ir_contract.py", "compares whole vocabularies"),
    note(660, CHK_Y + 48, "It runs no query, so a tag that no"),
    note(660, CHK_Y + 66, "differential test names is still checked."),
]

# Edges from each end down into the checker, labelled with what it reads there.
body += [
    f'<path d="M {LEFT_X + 120} {ROW_Y + ROW_H + 10} V {CHK_Y + 54} H {340 - 12}" fill="none" '
    f'stroke="{GREY}" stroke-width="2.2" stroke-dasharray="5 4" marker-end="url(#arG)"/>',
    f'<path d="M {RIGHT_X + 100} {ROW_Y + ROW_H + 10} V {CHK_Y + 54} H {640 + 12}" fill="none" '
    f'stroke="{GREY}" stroke-width="2.2" stroke-dasharray="5 4" marker-end="url(#arG)"/>',
    note(LEFT_X + 130, CHK_Y + 44, "reads the class"),
    note(RIGHT_X + 92, CHK_Y + 44, "reads the enum body", anchor="end"),
]

# ---- The two failure directions ------------------------------------------
FAIL_Y = 428
body += [
    card(52, FAIL_Y, 420, 92, "Python emits, Rust rejects", "the plan fails to deserialize"),
    note(262, FAIL_Y + 78, "Loud, but only for a query that uses that tag.", anchor="middle"),
    card(508, FAIL_Y, 420, 92, "Rust accepts, Python never emits", "an engine capability goes unreachable"),
    note(718, FAIL_Y + 78, "Silent. Nothing raises, nothing is slower to notice.", anchor="middle"),
]

body += [
    f'<path d="M {440} {CHK_Y + 96} L {300} {FAIL_Y - 10}" fill="none" stroke="{AMBER_DEEP}" '
    f'stroke-width="2.4" marker-end="url(#arA)"/>',
    f'<path d="M {540} {CHK_Y + 96} L {680} {FAIL_Y - 10}" fill="none" stroke="{AMBER_DEEP}" '
    f'stroke-width="2.4" marker-end="url(#arA)"/>',
    f'<text x="326" y="{FAIL_Y - 24}" text-anchor="end" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" fill="{AMBER_DEEP}">one side has a tag</text>',
    f'<text x="654" y="{FAIL_Y - 24}" font-family="{FONT}" font-size="11.5" '
    f'font-weight="700" fill="{AMBER_DEEP}">the other side has a tag</text>',
]

body.append(
    f'<text x="490" y="{H - 8}" text-anchor="middle" font-family="{FONT}" font-size="11" '
    f'fill="{MUTED}">Both sides change in one commit, or neither does.</text>'
)

write("ir_wire_contract", svg(W, H, "".join(body)))
print("wrote ir_wire_contract.svg")
