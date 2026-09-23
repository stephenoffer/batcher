"""The engine's one rendering vocabulary — units and display-width-correct text.

`units` answers "how long, how big, how many, what share" with exactly the rules the web
dashboard's `ui.js` uses, so a number never reads one way in the terminal and another in
the browser. `text` answers "how wide is this on a terminal", correctly for color escapes
and for double-width characters, so an aligned column stays aligned.

Layer 0: `plan`, `observe`, `io`, `dist`, and `api` all render numbers, none of them may
import each other, and copy-pasting a formatter is how the three divergent byte formatters
this package replaced came to exist.
"""

from __future__ import annotations

from batcher._internal.humanize.text import (
    ELLIPSIS,
    display_width,
    fit,
    pad,
    strip_ansi,
    truncate,
    wrap,
)
from batcher._internal.humanize.units import (
    UNKNOWN,
    byte_size,
    count,
    duration_ms,
    duration_s,
    ordinal,
    percent,
    plural,
    rate,
    signed_ratio,
)

__all__ = [
    "ELLIPSIS",
    "UNKNOWN",
    "byte_size",
    "count",
    "display_width",
    "duration_ms",
    "duration_s",
    "fit",
    "ordinal",
    "pad",
    "percent",
    "plural",
    "rate",
    "signed_ratio",
    "strip_ansi",
    "truncate",
    "wrap",
]
