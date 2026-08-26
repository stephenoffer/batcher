"""The terminal face of the engine — a live progress bar plus structured status lines.

**Design.** Sleek and technical, with the retro register coming from typography rather than
from color: block-drawing bars, a braille spinner, a throughput sparkline, box-rule
separators. Layout is fixed-column and tabular so numbers line up as they change — the
thing that separates an instrument from a status dump. Colors are the same sequential blue
ramp the dashboard uses, because progress is a magnitude.

**Animation is real, not decorative.** The bar advances in eighth-cells (8x the resolution
of its width), so it reads as motion instead of stepping. Throughput is measured over a
short trailing window rather than as an average since the query began, which is what lets
the sparkline show the stall it claims to show. Nothing is invented: with no row estimate
the bar shows an honest indeterminate sweep rather than a fabricated percentage, and the
ETA is omitted rather than guessed.

Capability detection and the glyph/color tables live in `observe.theme`; degradation to
16-color or ASCII is designed there. The split here is `state` (what is true about a run),
`paint` (how one line is drawn), and `reporter` (the bus sink that owns the terminal).
"""

from __future__ import annotations

from batcher.observe.console.paint import bar, compose, sparkline, sweep
from batcher.observe.console.reporter import ConsoleReporter, should_render
from batcher.observe.console.state import RunState

__all__ = [
    "ConsoleReporter",
    "RunState",
    "bar",
    "compose",
    "should_render",
    "sparkline",
    "sweep",
]
