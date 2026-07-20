"""Terminal capability detection, the color ramp, and the glyph set — the console's look.

Split from the renderer so "what the terminal can do" and "what we draw" are decided in one
place and consumed everywhere, instead of each call site re-deriving whether it may emit a
truecolor escape or a block-drawing character.

**Capability, not assumption.** Terminals disagree about color depth and about which Unicode
blocks they have glyphs for, so both are detected and both degrade: truecolor → 256-color →
16-color → none, and block-drawing → ASCII. The degraded forms are designed, not accidental
— a 16-color CI log and a truecolor iTerm2 render the same layout, only the fidelity differs.

The palette is the same sequential blue ramp the web dashboard uses (see
`assets/app.css`), for the same reason: elapsed time and progress are *magnitudes*, and one
hue light→dark is how a magnitude is encoded. The retro character comes from the typography
— block-drawing, braille, sparklines — not from a nostalgic amber that would break that
correspondence.
"""

from __future__ import annotations

import os
import sys
from typing import TextIO

__all__ = ["Glyphs", "Palette", "detect"]

# --- glyphs -----------------------------------------------------------------
# Eighth-blocks give the progress bar 8x the resolution of its cell count: a 24-cell bar
# advances in 192 visible steps instead of 24, which is what makes the fill read as motion
# rather than as a row of jumping squares.
_EIGHTHS = "▏▎▍▌▋▊▉█"
# Braille dots — 8 phases, and they occupy one cell in every monospace font that has them.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# Shading ramp for the indeterminate sweep's comet trail, darkest → lightest.
_SHADES = "█▓▒░"
# Vertical bars for the throughput sparkline.
_SPARK = "▁▂▃▄▅▆▇█"

# The blue column of the xterm-256 color cube, light → dark. The 256-color stand-in for
# the truecolor ramp: coarser, but still a gradient rather than a flat fill.
_XTERM_BLUES = (153, 111, 75, 69, 33, 32, 26, 25)


class Glyphs:
    """The character set to draw with, at a given Unicode fidelity."""

    def __init__(self, *, unicode: bool) -> None:
        self.unicode = unicode
        self.eighths = _EIGHTHS if unicode else "#"
        self.full = "█" if unicode else "#"
        self.empty = "░" if unicode else "."
        self.spinner = _SPINNER if unicode else "|/-\\"
        self.shades = _SHADES if unicode else "#+-."
        self.spark = _SPARK if unicode else "_.-~"
        self.ok = "✔" if unicode else "OK"
        self.fail = "✘" if unicode else "XX"
        self.sep = "·" if unicode else "|"
        self.rule = "─" if unicode else "-"
        self.cap_l = "▕" if unicode else "["
        self.cap_r = "▏" if unicode else "]"


class Palette:
    """Color escapes at a given depth, with every role degrading to `""` when uncolored.

    Roles rather than colors: the renderer asks for `dim` or `accent`, so dropping to a
    16-color terminal or to `NO_COLOR` changes this table and nothing else.
    """

    def __init__(self, depth: int) -> None:
        self.depth = depth
        on = depth > 0
        self.reset = "\x1b[0m" if on else ""
        self.dim = "\x1b[2m" if on else ""
        self.bold = "\x1b[1m" if on else ""
        self.muted = "\x1b[38;5;244m" if depth >= 8 else ("\x1b[90m" if on else "")
        if depth >= 24:
            self.accent = self._fg(57, 135, 229)
        elif depth >= 8:
            self.accent = "\x1b[38;5;75m"
        else:
            self.accent = "\x1b[36m" if on else ""
        self.good = self._fg(12, 163, 12) if depth >= 24 else ("\x1b[32m" if on else "")
        self.warn = self._fg(250, 178, 25) if depth >= 24 else ("\x1b[33m" if on else "")
        self.serious = self._fg(236, 131, 90) if depth >= 24 else ("\x1b[33m" if on else "")
        self.critical = self._fg(208, 59, 59) if depth >= 24 else ("\x1b[31m" if on else "")

    def _fg(self, r: int, g: int, b: int) -> str:
        """A 24-bit foreground escape."""
        return f"\x1b[38;2;{r};{g};{b}m"

    def ramp(self, t: float) -> str:
        """A color from the sequential blue ramp at position `t` in [0, 1], light → dark.

        Used to tint the progress bar along its length, so the filled region reads as a
        single gradient object rather than a flat block — the cheapest way to make a bar
        look considered. A 256-color terminal gets a real (coarser) ramp from the xterm
        cube rather than a flat accent; only 16-color and monochrome collapse to one value,
        where there is nothing to interpolate between.
        """
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        if self.depth == 8:
            return (
                f"\x1b[38;5;{_XTERM_BLUES[min(int(t * len(_XTERM_BLUES)), len(_XTERM_BLUES) - 1)]}m"
            )
        if self.depth < 24:
            return self.accent
        # Steps 250 → 550 of the documented blue ramp, interpolated.
        lo = (134, 182, 239)
        hi = (28, 92, 171)
        return self._fg(*(round(lo[i] + (hi[i] - lo[i]) * t) for i in range(3)))


def detect(stream: TextIO | None = None) -> tuple[Palette, Glyphs]:
    """The palette and glyph set appropriate for `stream` and the environment.

    Honors the conventions terminals actually publish: ``NO_COLOR`` disables color
    (no-color.org), ``FORCE_COLOR``/``CLICOLOR_FORCE`` enable it against a pipe,
    ``COLORTERM=truecolor|24bit`` advertises 24-bit, and ``TERM`` carries the 256-color and
    dumb-terminal signals. Unicode is taken from the stream's own encoding rather than
    guessed, so a `LANG=C` terminal gets the ASCII forms instead of mojibake.

    Args:
        stream: The output stream to inspect; defaults to `sys.stderr`.

    Returns:
        A ``(Palette, Glyphs)`` pair for the detected capabilities.
    """
    stream = stream if stream is not None else sys.stderr
    env = os.environ
    term = env.get("TERM", "")
    if env.get("NO_COLOR") is not None or term == "dumb":
        depth = 0
    elif env.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        depth = 24
    elif "256" in term:
        depth = 8
    elif env.get("FORCE_COLOR") or env.get("CLICOLOR_FORCE"):
        depth = 24
    else:
        depth = 4 if term else 0
    encoding = (getattr(stream, "encoding", "") or "").lower()
    unicode_ok = "utf" in encoding and term != "dumb"
    return Palette(depth), Glyphs(unicode=unicode_ok)
