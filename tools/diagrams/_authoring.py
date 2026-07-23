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
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1e293b; stroke: #334155; }
    .band-blue { fill: #172554; stroke: #3b82f6; }
    .band-amber { fill: #2c1f06; stroke: #f59e0b; }
    .band-grey { fill: #131c31; stroke: #334155; }
    .t-title { fill: #e2e8f0; }
    .t-sub { fill: #94a3b8; }
    .t-arrow { fill: #cbd5e1; }
  }
</style>"""

DEFS = f"""<defs>{STYLE}
<filter id="sh" x="-20%" y="-30%" width="140%" height="180%">
  <feDropShadow dx="0" dy="2.5" stdDeviation="4.5" flood-color="#0f172a" flood-opacity="0.18"/>
</filter>
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


def curve(x1: float, y1: float, cx: float, cy: float, x2: float, y2: float, kind: str = "amber") -> str:
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
