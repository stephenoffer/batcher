#!/usr/bin/env python3
"""Draw `jit_fallback.svg` — where the JIT gives up, and how much it gives up at once.

`execution_tiers.svg` draws *that* the JIT falls back. This one draws the mechanism,
because the single edge in that picture is really two edges with very different blast
radii, and a reader who has only seen the one edge will guess wrong about both.

Source of truth, all in `crates/bc-codegen/` and `crates/bc-interp/`:

* `bc-codegen/src/analyze.rs::analyze` — the subset gate. Columns compile only as
  `Int64`, `Float64`, `Date32` or `Timestamp(Microsecond, None)`; a bool or string
  literal, a string function, a list, a struct, a date function, `Coalesce` and
  `NullIf` are each their own `Unsupported` arm. Integer `Div`/`Mod` compile only
  against a constant non-zero, non-`-1` divisor, because Cranelift's `sdiv` traps.
* `bc-codegen/src/cache.rs::compile_expr_cached` — the process-wide memo, keyed on
  the expression *and* the batch's full schema. `type Entry = Option<Arc<CompiledExpr>>`:
  a refusal is cached too, so an unsupported expression is analyzed once rather than
  once per call. `MAX_ENTRIES = 1024`, and overflow clears the map rather than evicting.
* `bc-interp/src/ops/mod.rs` — `pub(crate) type Jit = Option<Arc<CompiledExpr>>`, the
  compile-once-per-operator handle, `Send + Sync` so one compile is shared across the
  rayon workers. `try_compile`'s own doc: "a per-morsel compile would lose to the
  interpreter."
* `bc-interp/src/ops/mod.rs::eval_jit` — the whole fallback, and it is silent in both
  directions: no compiled body, or an `Err` from this batch, and the interpreter runs.
* `bc-codegen/src/cache.rs` — the 16.6 ms figure: what a 64-row query with one filter
  and two projections paid in Cranelift before the memo existed.

The two blast radii are the point. Failing `analyze` costs the operator its whole fast
path for the life of the query; failing at `eval` costs one morsel and the next one
tries again.

Form: a vertical spine for the compiled path with the two refusals leaving it sideways
at the two different heights they actually leave it at.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, MUTED, card, label, note, svg, write

W, H = 980, 540

MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"

CW, CH = 330, 88
LX, RX = 60, 590
A_Y, B_Y, C_Y = 86, 230, 374


def crate(x: float, y: float, text: str) -> str:
    """Where the box lives, in the engine's own spelling."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{MONO}" font-size="10.5" '
        f'fill="{BLUE}">{text}</text>'
    )


def fall(x1: float, y1: float, x2: float, y2: float) -> str:
    """A refusal edge: dashed and amber, so it reads as the exceptional route."""
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{AMBER_DEEP}" '
        f'stroke-width="2.3" stroke-dasharray="6 4" marker-end="url(#arA)"/>'
    )


def amber(x: float, y: float, text: str, anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="11.5" '
        f'font-weight="700" fill="{AMBER_DEEP}">{text}</text>'
    )


body = [
    # ---- The compiled path ------------------------------------------------
    card(LX, A_Y, CW, CH, "analyze()", "which types and ops can compile"),
    crate(LX + CW / 2, A_Y + CH - 10, "bc-codegen"),
    card(LX, B_Y, CW, CH, "CompiledExpr", "one Arc, shared by every worker"),
    crate(LX + CW / 2, B_Y + CH - 10, "Send + Sync"),
    card(LX, C_Y, CW, CH, "eval(batch)", "16,384 rows at a time"),
    crate(LX + CW / 2, C_Y + CH - 10, "bc-interp drives the loop"),
    # ---- The memo and the interpreter -------------------------------------
    card(RX, A_Y, CW, CH, "Compile cache", "keyed on the expression and the schema"),
    crate(RX + CW / 2, A_Y + CH - 10, "1024 entries"),
    card(RX, B_Y, CW, CH, "Expr::eval", "the interpreter, and the oracle"),
    crate(RX + CW / 2, B_Y + CH - 10, "bc-expr"),
]

SPINE = LX + CW / 2
body += [
    f'<path d="M {SPINE} {A_Y + CH + 8} L {SPINE} {B_Y - 10}" fill="none" stroke="{BLUE}" '
    f'stroke-width="2.4" marker-end="url(#arB)"/>',
    label(SPINE + 16, A_Y + CH + 32, "in the subset: compile,", size=11.5),
    note(SPINE + 16, A_Y + CH + 50, "and compile exactly once"),
    f'<path d="M {SPINE} {B_Y + CH + 8} L {SPINE} {C_Y - 10}" fill="none" stroke="{BLUE}" '
    f'stroke-width="2.4" marker-end="url(#arB)"/>',
    label(SPINE + 16, B_Y + CH + 32, "reused for every morsel,", size=11.5),
    note(SPINE + 16, B_Y + CH + 50, "never recompiled"),
    # analyze asks the memo first, and writes its answer back either way.
    f'<path d="M {LX + CW + 8} {A_Y + CH / 2} L {RX - 10} {A_Y + CH / 2}" fill="none" '
    f'stroke="{BLUE}" stroke-width="2.4" marker-end="url(#arB)"/>',
    label(490, A_Y + CH / 2 - 14, "asked before it compiles", anchor="middle", size=11.5),
    note(490, A_Y + CH / 2 + 26, "and a refusal is remembered too", anchor="middle"),
]

# ---- The two refusals, at their two different heights -------------------
body += [
    fall(LX + CW + 8, A_Y + CH - 4, RX - 10, B_Y + 30),
    amber(486, B_Y - 2, "outside the subset:"),
    note(486, B_Y + 16, "this operator never compiles at all", anchor="middle"),
    fall(LX + CW + 8, C_Y + 22, RX - 10, B_Y + CH - 16),
    amber(486, C_Y + 10, "nulls this body cannot carry:"),
    note(486, C_Y + 28, "this batch only, and the next one tries again", anchor="middle"),
]

body += [
    note(
        490,
        H - 46,
        "The subset is narrow on purpose: numeric, date and timestamp columns, arithmetic and comparison. No strings.",
        anchor="middle",
    ),
    f'<text x="490" y="{H - 24}" text-anchor="middle" font-family="{FONT}" font-size="11.5" '
    f'fill="{MUTED}">A per-morsel compile would lose. One 64-row query paid 16.6 ms of '
    f"Cranelift before the cache existed.</text>",
]

write("jit_fallback", svg(W, H, "".join(body)))
print("wrote jit_fallback.svg")
