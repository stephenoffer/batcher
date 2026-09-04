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
from collections.abc import Mapping
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

    Honors the conventions terminals actually publish — see `_color_depth` for the
    precedence and for the two specs it follows exactly. Unicode is taken from the stream's
    own encoding rather than guessed, so a `LANG=C` terminal gets the ASCII forms instead of
    mojibake.

    Args:
        stream: The output stream to inspect; defaults to `sys.stderr`.

    Returns:
        A ``(Palette, Glyphs)`` pair for the detected capabilities.
    """
    stream = stream if stream is not None else sys.stderr
    depth = _color_depth(os.environ)
    encoding = (getattr(stream, "encoding", "") or "").lower()
    unicode_ok = "utf" in encoding and os.environ.get("TERM", "") != "dumb"
    return Palette(depth), Glyphs(unicode=unicode_ok)


#: Terminals that advertise nothing but are known to render 24-bit color. Read from
#: ``TERM_PROGRAM``, which is what each of them actually sets.
_TRUECOLOR_PROGRAMS = frozenset(
    {"iterm.app", "wezterm", "vscode", "hyper", "ghostty", "warpterminal"}
)
#: ``TERM`` values that imply truecolor without a ``COLORTERM``.
_TRUECOLOR_TERMS = ("kitty", "alacritty", "contour", "wezterm")
#: What each ``FORCE_COLOR`` level means, following the convention Node's ecosystem set.
_FORCE_LEVELS = {"0": 0, "false": 0, "1": 4, "2": 8, "3": 24, "true": 24, "": 24}


def _color_depth(env: Mapping[str, str]) -> int:
    """Bits of color `env` says the terminal has: 0, 4, 8, or 24.

    Ordered by how *explicit* each signal is, because that is the only ordering that lets a
    user override a wrong guess. ``NO_COLOR`` and ``FORCE_COLOR`` are deliberate statements
    and win; ``COLORTERM`` is the terminal advertising itself; ``TERM``/``TERM_PROGRAM`` are
    inference from what the terminal calls itself.

    Two conventions are followed to the letter rather than approximately, because getting
    them nearly right is worse than not implementing them — a user who sets one and does not
    get what it promises has no way to tell a bug from a policy:

    * ``NO_COLOR`` disables color when set **to a non-empty value** (no-color.org). An empty
      ``NO_COLOR=`` does not, which is what lets a wrapper script unset the variable for a
      child by exporting it empty.
    * ``FORCE_COLOR`` enables color against a pipe, and ``FORCE_COLOR=0`` **disables** it.
      Treating any value as "on" is the common mistake, and it turns the one variable people
      use to *suppress* color in CI into one that forces it.

    Args:
        env: The environment mapping to read.

    Returns:
        The color depth in bits: 0 (none), 4 (16-color), 8 (256-color), or 24 (truecolor).
    """
    term = env.get("TERM", "")
    if term == "dumb":
        return 0
    if env.get("NO_COLOR"):
        return 0
    forced = env.get("FORCE_COLOR")
    if forced is not None:
        return _FORCE_LEVELS.get(forced.lower(), 24)
    if env.get("CLICOLOR_FORCE"):
        return 24
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return 24
    # Windows Terminal and ConEmu render truecolor and set no TERM at all under cmd.exe,
    # so without these the whole Windows story degrades to monochrome.
    if env.get("WT_SESSION") or env.get("ConEmuANSI", "").upper() == "ON":
        return 24
    if env.get("TERM_PROGRAM", "").lower() in _TRUECOLOR_PROGRAMS:
        return 24
    if any(name in term for name in _TRUECOLOR_TERMS):
        return 24
    if "256" in term:
        return 8
    if env.get("TERM_PROGRAM", "").lower() == "apple_terminal":
        return 8
    return 4 if term else 0
