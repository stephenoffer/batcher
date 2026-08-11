"""Window-function translation for the SQL front-end.

Façade over the two halves: `frame` reads a sqlglot window spec into the engine's
`(start, end, units)` frame plus its partition and order keys, and `translate`
groups the SELECT-list windows by that spec and lowers each onto `ds.window(...)`.

The import path `batcher._sql.parser.windowing` is unchanged, so `clauses.py`,
`translator.py` and `grouping.py` reach these names exactly as before.
"""

from __future__ import annotations

from batcher._sql.parser.windowing.frame import _WINDOW_AGGS
from batcher._sql.parser.windowing.translate import (
    _has_window,
    _inline_named_windows,
    _is_window,
    _window,
    hoist_nested_windows,
    hoist_window_args,
    rewrite_aggs_in_windows,
    rewrite_group_keys_in_windows,
    rewrite_offset_defaults,
)

__all__ = [
    "_WINDOW_AGGS",
    "_has_window",
    "_inline_named_windows",
    "_is_window",
    "_window",
    "hoist_nested_windows",
    "hoist_window_args",
    "rewrite_aggs_in_windows",
    "rewrite_group_keys_in_windows",
    "rewrite_offset_defaults",
]
