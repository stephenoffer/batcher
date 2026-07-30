#!/usr/bin/env python3
"""Draw `execution_tiers.svg` -- one expression type, three tiers, one answer.

The differentiator here is not speed, it is that speed is bought without a second
semantics. There is exactly one scalar expression type (`bc_expr::Expr`) and one
relational plan type (`bc_ir::RelOp`), and all three execution paths consume the same
one. That shared source is what makes parity a structural property rather than a
testing aspiration.

The three tiers, and the contract each owes the one above it:

* **Tier-0 sequential** (`crates/bc-interp`, `execute`) is the correctness oracle. It
  is kept simple and obviously correct, and every other path is checked against it.
* **Tier-0 parallel** (`crates/bc-interp/src/par.rs`) reuses the same operator code
  and changes only the scheduling: morselize, run on a rayon pool, hash-shuffle into
  the breakers. It must compute exactly what the sequential path computes.
* **Tier-1 JIT** (`crates/bc-codegen`) compiles the supported subset of scalar
  expressions with Cranelift, once per operator and reused across morsels. On its
  subset it must be bit-for-bit identical to the interpreter, and on anything it does
  not support it must fall back silently rather than diverge.

The fallback edge is drawn deliberately, because it is the part that is easy to get
wrong and easy to forget: an unsupported expression is not an error and not a slower
compile, it is a return to the oracle.

Form: one source box fanning into three tier cards, with the parity obligations drawn
as labelled edges back toward the oracle rather than as a legend. Every arrow carries
a label, per the diagram rules; the fallback edge is dashed and amber so it reads as
the exceptional path without relying on colour alone to say so.
"""

from __future__ import annotations

from _authoring import AMBER_DEEP, BLUE, FONT, GREY, write

W, H = 980, 430

SRC_X, SRC_Y, SRC_W, SRC_H = 40, 150, 200, 108
TIER_W, TIER_H = 208, 108
TIER_X = (330, 590, 850 - 208 + 130)   # third column right-aligned in the canvas
TIER_Y = (60, 176, 292)

STYLE = """<style>
  .surf { fill: #ffffff; }
  .card { fill: #ffffff; stroke: #cbd5e1; }
  .card-oracle { fill: #eff6ff; stroke: #2563eb; }
  .card-src { fill: #f8fafc; stroke: #94a3b8; }
  .t-head { fill: #1e293b; }
  .t-sub { fill: #5b6675; }
  .t-tag { fill: #2563eb; }
  @media (prefers-color-scheme: dark) {
    .surf { fill: #131c31; }
    .card { fill: #1e293b; stroke: #334155; }
    .card-oracle { fill: #172554; stroke: #3b82f6; }
    .card-src { fill: #0f172a; stroke: #475569; }
    .t-head { fill: #e2e8f0; }
    .t-sub { fill: #94a3b8; }
    .t-tag { fill: #60a5fa; }
  }
</style>"""

DEFS = f"""<defs>{STYLE}
<filter id="sh" x="-20%" y="-30%" width="140%" height="180%">
  <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.16"/>
</filter>
<marker id="aB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{BLUE}"/></marker>
<marker id="aA" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{AMBER_DEEP}"/></marker>
<marker id="aG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{GREY}"/></marker>
</defs>"""


def tier(x: float, y: float, title: str, crate: str, role: str, oracle: bool = False) -> str:
    """One execution tier: what it is, which crate it lives in, what it owes."""
    cls = "card-oracle" if oracle else "card"
    return (
        f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{TIER_W}" height="{TIER_H}" rx="11" '
        f'class="{cls}" stroke-width="1.6"/></g>'
        f'<text x="{x + 18}" y="{y + 30}" font-family="{FONT}" font-size="14.5" font-weight="700" '
        f'class="t-head">{title}</text>'
        f'<text x="{x + 18}" y="{y + 52}" font-size="11" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" class="t-tag">{crate}</text>'
        f'<text x="{x + 18}" y="{y + 76}" font-family="{FONT}" font-size="11.5" '
        f'class="t-sub">{role}</text>'
    )


def edge(x1: float, y1: float, x2: float, y2: float, marker: str, colour: str,
         dashed: bool = False) -> str:
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{colour}" stroke-width="2.2"'
        f'{dash} marker-end="url(#{marker})"/>'
    )


def elabel(x: float, y: float, text: str, colour: str, anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="11.5" '
        f'font-weight="700" fill="{colour}">{text}</text>'
    )


T0X, T1X, T2X = 330, 620, 620
parts = [
    f'<rect x="0" y="0" width="{W}" height="{H}" rx="14" class="surf"/>',
    f'<text x="40" y="48" font-family="{FONT}" font-size="19" font-weight="700" class="t-head">'
    f'One expression type, three ways to run it</text>',
    f'<text x="40" y="72" font-family="{FONT}" font-size="13" class="t-sub">'
    f'Speed comes from scheduling and compilation, never from a second set of semantics.</text>',
    # the single source of truth
    f'<g filter="url(#sh)"><rect x="{SRC_X}" y="{SRC_Y}" width="{SRC_W}" height="{SRC_H}" rx="11" '
    f'class="card-src" stroke-width="1.6"/></g>',
    f'<text x="{SRC_X + 18}" y="{SRC_Y + 32}" font-family="{FONT}" font-size="14.5" '
    f'font-weight="700" class="t-head">One Expr, one RelOp</text>',
    f'<text x="{SRC_X + 18}" y="{SRC_Y + 56}" font-family="{FONT}" font-size="11" '
    f'class="t-tag">bc-expr / bc-ir</text>',
    f'<text x="{SRC_X + 18}" y="{SRC_Y + 80}" font-family="{FONT}" font-size="11.5" '
    f'class="t-sub">every tier consumes this</text>',
]

# the three tiers
parts += [
    tier(T0X, TIER_Y[0], "Tier-0 sequential", "bc-interp::execute",
         "the correctness oracle", oracle=True),
    tier(T0X, TIER_Y[1], "Tier-0 parallel", "bc-interp::par",
         "same operators, morselized"),
    tier(T0X, TIER_Y[2], "Tier-1 JIT", "bc-codegen (Cranelift)",
         "compiled once per operator"),
]

# fan-out from the shared source
src_mid_y = SRC_Y + SRC_H / 2
for ty in TIER_Y:
    parts.append(edge(SRC_X + SRC_W + 8, src_mid_y, T0X - 10, ty + TIER_H / 2, "aB", BLUE))
parts.append(elabel(SRC_X + SRC_W + 52, src_mid_y - 62, "the same", BLUE, "start"))
parts.append(elabel(SRC_X + SRC_W + 52, src_mid_y - 46, "tree", BLUE, "start"))

# Parity obligations and the fallback both run right of the tier column, on two
# separate spines, so neither crosses the fan-out or the shared-source card.
SPINE = T0X + TIER_W + 26          # parity spine
FALLBACK = T0X + TIER_W + 232      # fallback spine, well clear of the parity labels
right = T0X + TIER_W
oracle_mid = TIER_Y[0] + TIER_H / 2

parts += [
    # parallel -> oracle
    f'<path d="M {right + 8} {TIER_Y[1] + TIER_H / 2} H {SPINE} V {oracle_mid + 14} '
    f'H {right + 8}" fill="none" stroke="{BLUE}" stroke-width="2.2" '
    f'marker-end="url(#aB)"/>',
    # jit -> oracle
    f'<path d="M {right + 8} {TIER_Y[2] + TIER_H / 2} H {SPINE} V {oracle_mid + 14}" '
    f'fill="none" stroke="{BLUE}" stroke-width="2.2"/>',
    elabel(SPINE + 12, TIER_Y[1] + TIER_H / 2 - 10, "must equal the oracle", BLUE, "start"),
    elabel(SPINE + 12, TIER_Y[2] + TIER_H / 2 - 10, "bit-for-bit on its subset", BLUE, "start"),
]

# The fallback edge: an unsupported expression is not an error and not a slow
# compile, it is a return to the oracle.
parts += [
    f'<path d="M {right + 8} {TIER_Y[2] + TIER_H - 22} H {FALLBACK} V {oracle_mid - 14} '
    f'H {right + 8}" fill="none" stroke="{AMBER_DEEP}" stroke-width="2.2" '
    f'stroke-dasharray="6 4" marker-end="url(#aA)"/>',
    elabel(FALLBACK + 12, TIER_Y[1] + 30, "unsupported expression:", AMBER_DEEP, "start"),
    elabel(FALLBACK + 12, TIER_Y[1] + 46, "falls back, never diverges", AMBER_DEEP, "start"),
]

parts.append(
    f'<text x="40" y="{H - 24}" font-family="{FONT}" font-size="11.5" class="t-sub">'
    f'The parity edges are enforced by tests, not convention: seq == par == JIT on every '
    f'supported input, and the interpreter is the reference for both.</text>'
)

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
    f'width="{W}" height="{H}">{DEFS}{"".join(parts)}</svg>'
)
write("execution_tiers", svg)
print("wrote execution_tiers.svg")
