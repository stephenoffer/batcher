"""Window-function translation for the SQL front-end.

Façade over the two halves: `frame` reads a sqlglot window spec into the engine's
`(start, end, units)` frame plus its partition and order keys, and `translate`
groups the SELECT-list windows by that spec and lowers each onto `ds.window(...)`. `derived`
rewrites the window forms with no operator of their own (``lag``/``lead`` ``IGNORE NULLS``,
frame ``EXCLUDE``) into ones that have one, and `limits` answers ``LIMIT … PERCENT`` and
``FETCH … WITH TIES`` with ranking windows.

The import path `batcher._sql.parser.windowing` is unchanged, so `clauses.py`,
`translator.py` and `grouping.py` reach these names exactly as before.
"""

from __future__ import annotations

from batcher._sql.parser.windowing.derived import (
    rewrite_frame_exclusions,
    rewrite_ignore_nulls_navigation,
)
from batcher._sql.parser.windowing.frame import _WINDOW_AGGS
from batcher._sql.parser.windowing.limits import normalize_limit_modifiers
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
    "normalize_limit_modifiers",
    "rewrite_aggs_in_windows",
    "rewrite_frame_exclusions",
    "rewrite_group_keys_in_windows",
    "rewrite_ignore_nulls_navigation",
    "rewrite_offset_defaults",
]
