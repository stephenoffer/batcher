#!/usr/bin/env python3
"""Shared drawing primitives for the Batcher documentation diagrams.

The diagrams are SVG (the committed source of truth) drawn in one visual language:
a blue primary, an amber accent, slate text, white cards with a soft shadow, and
labeled arrows. This module holds that language as functions so a new diagram
inherits it rather than re-deriving it by hand, which is how a diagram set drifts
into looking like several documents.

Every diagram emitted here is **theme-aware**: the ``STYLE`` block below restates
the surface and text colors under ``prefers-color-scheme: dark``, so a diagram sits
on the dark page instead of glaring out of it as a white slab.

Run a diagram module directly to regenerate its ``.svg`` under
``docs/_static/diagrams/``.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

#: The scripts live in `tools/diagrams/`; the SVGs they emit live in the docs static
#: tree. They are deliberately separated: Sphinx copies `html_static_path` wholesale,
#: so anything beside the SVGs ships as a published website asset.
HERE = Path(__file__).resolve().parents[2] / "docs" / "_static" / "diagrams"

BLUE = "#2563eb"
BLUE_MID = "#3b82f6"
AMBER = "#f59e0b"
AMBER_DEEP = "#d97706"
SLATE = "#1e293b"
MUTED = "#5b6675"
GREY = "#94a3b8"

FONT = "Helvetica,Arial,sans-serif"
MONO = "Menlo,Consolas,DejaVu Sans Mono,monospace"

#: Restates every surface color for dark mode. `prefers-color-scheme` works inside an
#: SVG referenced by <img>, so this adapts with the OS theme without any page script.
STYLE = """<style>
  .surface { fill: #ffffff; stroke: #cbd5e1; }
  .band-blue { fill: #eff6ff; stroke: #3b82f6; }
  .band-amber { fill: #fffbeb; stroke: #f59e0b; }
  .band-grey { fill: #f8fafc; stroke: #cbd5e1; }
  .t-title { fill: #1e293b; }
  .t-sub { fill: #5b6675; }
  .t-arrow { fill: #475569; }
  .t-code { fill: #1e40af; }
  .code-bg { fill: #f1f5f9; stroke: #e2e8f0; }
  .pill-blue { fill: #dbeafe; } .pill-amber { fill: #fef3c7; } .pill-grey { fill: #e2e8f0; }
  .pt-blue { fill: #1d4ed8; } .pt-amber { fill: #b45309; } .pt-grey { fill: #334155; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1e293b; stroke: #334155; }
    .band-blue { fill: #172554; stroke: #3b82f6; }
    .band-amber { fill: #2c1f06; stroke: #f59e0b; }
    .band-grey { fill: #131c31; stroke: #334155; }
    .t-title { fill: #e2e8f0; }
    .t-sub { fill: #94a3b8; }
    .t-arrow { fill: #cbd5e1; }
    .t-code { fill: #93c5fd; }
    .code-bg { fill: #0f172a; stroke: #334155; }
    .pill-blue { fill: #1e3a8a; } .pill-amber { fill: #78350f; } .pill-grey { fill: #334155; }
    .pt-blue { fill: #bfdbfe; } .pt-amber { fill: #fde68a; } .pt-grey { fill: #e2e8f0; }
    #tintBlue stop[offset="0"] { stop-color: #16233d; }
    #tintBlue stop[offset="1"] { stop-color: #1e3a8a; }
    #tintAmber stop[offset="0"] { stop-color: #2a1e07; }
    #tintAmber stop[offset="1"] { stop-color: #78350f; }
  }
</style>"""

DEFS = f"""<defs>{STYLE}
<filter id="sh" x="-20%" y="-30%" width="140%" height="180%">
  <feDropShadow dx="0" dy="2.5" stdDeviation="4.5" flood-color="#0f172a" flood-opacity="0.18"/>
</filter>
<filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
  <feDropShadow dx="0" dy="4" stdDeviation="9" flood-color="#2563eb" flood-opacity="0.35"/>
</filter>
<linearGradient id="hero" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#60a5fa"/><stop offset="0.5" stop-color="#2563eb"/><stop offset="1" stop-color="#4f46e5"/></linearGradient>
<linearGradient id="heroAmber" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#fbbf24"/><stop offset="1" stop-color="#d97706"/></linearGradient>
<linearGradient id="tintBlue" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f5f9ff"/><stop offset="1" stop-color="#dbeafe"/></linearGradient>
<linearGradient id="tintAmber" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fffdf5"/><stop offset="1" stop-color="#fdecc8"/></linearGradient>
<linearGradient id="ribbonBlue" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#93c5fd"/><stop offset="1" stop-color="#2563eb"/></linearGradient>
<linearGradient id="ribbonAmber" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#f59e0b"/><stop offset="1" stop-color="#fbbf24"/></linearGradient>
<marker id="arB" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{BLUE}"/></marker>
<marker id="arA" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{AMBER_DEEP}"/></marker>
<marker id="arG" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{GREY}"/></marker>
</defs>"""


def svg(width: int, height: int, body: str) -> str:
    """Wrap `body` in an SVG root with the shared defs and a viewBox."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" width="{width}" height="{height}">{DEFS}{body}</svg>'
    )


def band(x: float, y: float, w: float, h: float, label: str, kind: str = "blue") -> str:
    """A titled region grouping related cards. `kind` is blue, amber, or grey."""
    color = {"blue": BLUE_MID, "amber": AMBER, "grey": GREY}[kind]
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" class="band-{kind}" stroke-width="1.5"/>'
        f'<text x="{x + 18}" y="{y + 24}" font-family="{FONT}" font-size="12.5" font-weight="700" '
        f'letter-spacing="1.5" fill="{color}">{label}</text>'
    )


def card(x: float, y: float, w: float, h: float, title: str, sub: str = "") -> str:
    """A white card with a soft shadow, a bold title, and an optional subtitle."""
    cx = x + w / 2
    ty = y + (h / 2 + 5) if not sub else y + h / 2
    out = (
        f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'class="surface" stroke-width="1.2"/></g>'
        f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="{FONT}" font-size="13.5" '
        f'font-weight="700" class="t-title">{title}</text>'
    )
    if sub:
        out += (
            f'<text x="{cx}" y="{y + h / 2 + 19}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="11" class="t-sub">{sub}</text>'
        )
    return out


def arrow(x1: float, y1: float, x2: float, y2: float, kind: str = "blue") -> str:
    """A straight connector. Every arrow in a diagram should carry a label."""
    stroke, marker = {
        "blue": (BLUE, "arB"),
        "amber": (AMBER_DEEP, "arA"),
        "grey": (GREY, "arG"),
    }[kind]
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{stroke}" stroke-width="2.4" '
        f'marker-end="url(#{marker})"/>'
    )


def curve(
    x1: float, y1: float, cx: float, cy: float, x2: float, y2: float, kind: str = "amber"
) -> str:
    """A quadratic connector, for feedback edges that must not overlap the forward path."""
    stroke, marker = {
        "blue": (BLUE, "arB"),
        "amber": (AMBER_DEEP, "arA"),
        "grey": (GREY, "arG"),
    }[kind]
    return (
        f'<path d="M {x1} {y1} Q {cx} {cy} {x2} {y2}" fill="none" stroke="{stroke}" '
        f'stroke-width="2.4" stroke-dasharray="6 4" marker-end="url(#{marker})"/>'
    )


def label(x: float, y: float, text: str, anchor: str = "start", size: float = 12.5) -> str:
    """An arrow or region label. Arrows without labels say only 'related'."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="700" class="t-arrow">{text}</text>'
    )


def note(x: float, y: float, text: str, anchor: str = "start") -> str:
    """Secondary explanatory text, lighter than a label."""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="11.5" '
        f'class="t-sub">{text}</text>'
    )


def write(name: str, content: str) -> Path:
    """Write `content` to ``<name>.svg`` beside this module and return the path."""
    path = HERE / f"{name}.svg"
    path.write_text(content, encoding="utf-8")
    return path


# --- Richer primitives ----------------------------------------------------------------
# The set above draws an explanatory figure. The functions below draw the parts a reader
# notices first on a landing or concept page: a gradient hero for the subject, tinted
# accent cards, numbered steps, pills, code snippets, and check/cross marks for a matrix.
# Every one of them restyles under dark mode through the classes and gradient ids in
# `STYLE`, so none of them reintroduces the white-slab problem.


def hero(
    x: float, y: float, w: float, h: float, title: str, sub: str = "", kind: str = "blue"
) -> str:
    """A gradient card with white text and a soft glow, for the one subject of a figure."""
    grad = {"blue": "hero", "amber": "heroAmber"}[kind]
    cx = x + w / 2
    ty = y + h / 2 + (7 if not sub else -2)
    out = (
        f'<g filter="url(#glow)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
        f'fill="url(#{grad})"/></g>'
        f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="{FONT}" font-size="19" '
        f'font-weight="800" fill="#ffffff">{title}</text>'
    )
    if sub:
        out += (
            f'<text x="{cx}" y="{y + h / 2 + 20}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="12" fill="#ffffff" fill-opacity="0.88">{sub}</text>'
        )
    return out


def tint(
    x: float, y: float, w: float, h: float, title: str, sub: str = "", kind: str = "blue"
) -> str:
    """A tinted card with a colored accent bar on its left edge, for peers in a group."""
    grad, bar, stroke = {
        "blue": ("tintBlue", BLUE_MID, "#bfdbfe"),
        "amber": ("tintAmber", AMBER, "#fcd34d"),
    }[kind]
    cx = x + w / 2 + 3
    ty = y + (h / 2 + 5) if not sub else y + h / 2 - 1
    out = (
        f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="11" '
        f'fill="url(#{grad})" stroke="{stroke}" stroke-width="1.2"/></g>'
        f'<rect x="{x}" y="{y + 12}" width="4" height="{h - 24}" rx="2" fill="{bar}"/>'
        f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="{FONT}" font-size="13.5" '
        f'font-weight="700" class="t-title">{title}</text>'
    )
    if sub:
        out += (
            f'<text x="{cx}" y="{y + h / 2 + 17}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="11" class="t-sub">{sub}</text>'
        )
    return out


def step(x: float, y: float, n: int | str, kind: str = "blue") -> str:
    """A numbered circle centred on (x, y), for the order of a procedure."""
    fill = {"blue": BLUE, "amber": AMBER_DEEP, "grey": GREY}[kind]
    return (
        f'<circle cx="{x}" cy="{y}" r="14" fill="{fill}"/>'
        f'<text x="{x}" y="{y + 4.5}" text-anchor="middle" font-family="{FONT}" font-size="13" '
        f'font-weight="800" fill="#ffffff">{n}</text>'
    )


def pill(x: float, y: float, text: str, kind: str = "blue", anchor: str = "start") -> str:
    """A small rounded tag. `x` is the left edge, or the centre when `anchor` is middle."""
    w = 14 + 6.7 * len(text)
    left = x - w / 2 if anchor == "middle" else x
    return (
        f'<rect x="{left}" y="{y - 13}" width="{w}" height="20" rx="10" class="pill-{kind}"/>'
        f'<text x="{left + w / 2}" y="{y + 1}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="11" font-weight="700" letter-spacing="0.4" class="pt-{kind}">{text}</text>'
    )


def code(x: float, y: float, lines: list[str], w: float, size: float = 12.5) -> str:
    """A snippet of code on a quiet panel. Keep it to what a reader would type."""
    h = 20 + len(lines) * (size + 7)
    out = (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" class="code-bg" stroke-width="1"/>'
    )
    for i, line in enumerate(lines):
        out += (
            f'<text x="{x + 14}" y="{y + 20 + i * (size + 7) + size / 2}" font-family="{MONO}" '
            f'font-size="{size}" class="t-code" xml:space="preserve">{escape(line)}</text>'
        )
    return out


def mark(x: float, y: float, ok: bool) -> str:
    """A check or a cross in a circle, for a capability matrix. Shape carries the meaning."""
    if ok:
        return (
            f'<circle cx="{x}" cy="{y}" r="10" fill="{BLUE}"/>'
            f'<path d="M {x - 4.5} {y} L {x - 1} {y + 3.8} L {x + 5} {y - 3.8}" fill="none" '
            f'stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return (
        f'<circle cx="{x}" cy="{y}" r="10" fill="none" stroke="{GREY}" stroke-width="1.8"/>'
        f'<path d="M {x - 3.8} {y - 3.8} L {x + 3.8} {y + 3.8} M {x + 3.8} {y - 3.8} L {x - 3.8} {y + 3.8}" '
        f'stroke="{GREY}" stroke-width="2" stroke-linecap="round"/>'
    )


def ribbon(x1: float, y1: float, x2: float, y2: float, kind: str = "blue") -> str:
    """A thick S-curve with a gradient, for a fan-in or fan-out without arrowheads."""
    mx = (x1 + x2) / 2
    grad = {"blue": "ribbonBlue", "amber": "ribbonAmber"}[kind]
    return (
        f'<path d="M {x1} {y1} C {mx} {y1}, {mx} {y2}, {x2} {y2}" fill="none" '
        f'stroke="url(#{grad})" stroke-width="3.5" stroke-linecap="round" opacity="0.85"/>'
    )


def heading(x: float, y: float, text: str, anchor: str = "start", kind: str = "blue") -> str:
    """A letter-spaced caps label over a column or a region, without a band behind it."""
    color = {"blue": BLUE, "amber": AMBER_DEEP, "grey": MUTED}[kind]
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" font-size="12" '
        f'font-weight="800" letter-spacing="1.8" fill="{color}">{text}</text>'
    )
