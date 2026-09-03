"""The knobs and the presentation vocabulary the profile renderer draws with.

Kept apart from the drawing so "what may vary" is one short file: the glyph tables, the
fold thresholds, the `Styler` seam that lets `api` add color to a layer that may not
import `observe`, and the capability probes for width and Unicode.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "BLOCKS",
    "DEFAULT_WIDTH",
    "ESTIMATE_MISS_FACTOR",
    "FOLD_ABOVE_OPS",
    "FOLD_BELOW_SHARE",
    "HOTSPOTS",
    "MAX_SPINE_DEPTH",
    "RenderOptions",
    "Styler",
    "plain_styler",
    "terminal_width",
    "unicode_ok",
]

DEFAULT_WIDTH = 108
MIN_WIDTH, MAX_WIDTH = 72, 160

#: Above this many operators the tree folds cold subtrees away. Below it, everything is
#: printed: a reader of a 30-operator plan wants all 30, and eliding would only hide.
FOLD_ABOVE_OPS = 24
#: A subtree whose total time is under this share of the run is foldable. Not zero, because
#: the point is to remove rows that cannot contain the answer, and 1% cannot.
FOLD_BELOW_SHARE = 0.01
#: How many operators the hot-spot table names. Enough to see a pattern, few enough to read.
HOTSPOTS = 5

#: Ancestor levels the tree spine draws before it stops indenting.
#:
#: A deep *linear* plan -- thirty chained `with_columns`/`filter` calls, which is the shape
#: any long pipeline has -- pushed the operator's own name out of its own column. The
#: indent is three characters per level, so depth 20 costs 60; the operator column is
#: capped at 48; and `fit` keeps the *front* of a string. So the forty deepest operators of
#: a 61-operator plan rendered as a column of bare ellipsis: correct indentation, and not
#: one operator name anywhere on the page. Past this depth the indentation is not carrying
#: information a reader can act on, and it was displacing the information that is.
MAX_SPINE_DEPTH = 10

GLYPHS = {
    "tee": "├─ ",
    "last": "└─ ",
    "pipe": "│  ",
    "gap": "   ",
    "rule": "─",
    "mark": "▶",
    #: Stands in for the ancestor bars `MAX_SPINE_DEPTH` elides. Same width as `pipe`, so
    #: clamped rows stay aligned with unclamped ones.
    "elide": "⋯  ",
}
ASCII = {
    "tee": "|- ",
    "last": "`- ",
    "pipe": "|  ",
    "gap": "   ",
    "rule": "-",
    "mark": ">",
    "elide": "~  ",
}
#: Eighth-blocks, so a five-cell share bar resolves 40 steps rather than 5.
BLOCKS = "▏▎▍▌▋▊▉█"


class Styler(Protocol):
    """Applies a named presentation role to a piece of text.

    Roles rather than colors, so a caller that cannot color (a log file, a captured
    stream) supplies the identity and every call site stays unchanged. The roles used
    here are ``head``, ``dim``, ``muted``, ``accent``, ``good``, ``warn``, and
    ``critical``.
    """

    def __call__(self, role: str, text: str) -> str:
        """Return `text` presented in `role`."""
        ...


def plain_styler(role: str, text: str) -> str:  # noqa: ARG001 - the Styler signature
    """A `Styler` that adds nothing — the default, and what a non-terminal must get.

    Args:
        role: The presentation role, ignored.
        text: The text.

    Returns:
        `text`, unchanged.

    Examples:
        .. doctest::

            >>> from batcher.plan.profile.render import plain_styler
            >>> plain_styler("critical", "spilled")
            'spilled'
    """
    return text


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """Everything the renderer needs that is not the profile itself.

    Attributes:
        analyze: Show the measured columns rather than the planned ones.
        width: Total line width to lay the table out in.
        unicode: Draw with box-drawing and block glyphs; `False` selects the ASCII forms.
        style: The `Styler` to present roles with.
        fold: Fold cold subtrees on a large plan.
    """

    analyze: bool = False
    width: int = DEFAULT_WIDTH
    unicode: bool = True
    style: Styler = plain_styler
    fold: bool = True

    @property
    def glyphs(self) -> dict[str, str]:
        """The glyph table for this option set's Unicode fidelity."""
        return GLYPHS if self.unicode else ASCII


def terminal_width() -> int:
    """The width to lay out in, clamped so the table is neither cramped nor sprawling."""
    import shutil

    columns = shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns
    return max(MIN_WIDTH, min(MAX_WIDTH, columns))


def unicode_ok() -> bool:
    """Whether stdout can encode the box-drawing glyphs, taken from the stream itself."""
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in encoding


#: How far an estimate must miss before it is worth a reader's attention. Below 4x the
#: optimizer's choices are mostly unaffected, and flagging every 2x miss trains people to
#: ignore the section that flags them.
ESTIMATE_MISS_FACTOR = 4.0
