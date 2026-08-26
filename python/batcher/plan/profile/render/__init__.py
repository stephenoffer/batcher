"""How a `QueryProfile` becomes something a person can read.

The rendering half of the profile package, split from `types` so the value objects stay
about *what was measured* and this stays about *how it is shown*. It is the text behind
``ds.explain()`` and ``ds.explain(analyze=True)``, and it is the output a user pastes into
an issue when a query is slow — which is the whole design constraint.

**Four things this fixes about the shape that came before.**

*It draws the tree.* The previous form indented by two spaces per level, so at depth four
a reader counted columns to work out what fed what — and a wide join tree, where the
question is exactly "which side is which", was the case it failed hardest on. The box-drawing
spine here is the same one the dashboard's plan view uses, and for the same reason every
other engine's ``EXPLAIN`` has one.

*It aligns the numbers.* Values were interpolated into a sentence, so a column of times
started at a different position on every row and could not be scanned down. Fixed-width,
right-aligned cells make "which operator is the slow one" a glance instead of a read. The
widths are computed from the data, and the measurement is display-width-correct
(`_internal.humanize.fit`), so a source named in Japanese does not shear the table.

*It stays readable on a large plan.* A 300-operator plan printed 300 undifferentiated
lines. Here the hot operators are named up front, the critical path is marked, and cold
subtrees are folded away with a line saying how many and why — so the output grows with
the *interesting* part of a plan rather than with the plan.

*It accounts for the whole clock.* The previous summary reported a bottleneck as a share
of `total_ms`, which is the whole terminal operation, while the operators only cover the
engine call inside it. A query that spent 1 ms in operators and 170 ms in planning and
result assembly reported "bottleneck: filter, 1% of wall time" and said nothing about the
other 99% — the reader was left to conclude the profile was broken, which was closer to
the truth than the number was. The unaccounted remainder is now a line of its own.

Styling is injected, never imported: `plan` is layer 1 and may not reach `observe`, so
color arrives as a `Styler` callable the caller supplies and defaults to the identity. That
is what lets ``explain(format="ansi")`` be colored by `api` while the same code path
produces byte-identical plain text for a log file."""

from __future__ import annotations

from batcher.plan.profile.render.layout import render_profile
from batcher.plan.profile.render.options import (
    DEFAULT_WIDTH,
    RenderOptions,
    Styler,
    plain_styler,
)

__all__ = [
    "DEFAULT_WIDTH",
    "RenderOptions",
    "Styler",
    "plain_styler",
    "render_profile",
]
