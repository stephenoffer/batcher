"""Plan-construction helpers behind the thinner `Dataset` methods.

`Dataset` stays a thin fluent builder (the v2 maintainability contract): its heavier
methods (`window`) and the frame-level convenience sugar (`fill_null`/`drop_nulls`/
`cast`) delegate their bodies here, mirroring how terminal ops live in `terminal.py`.
These functions take the `Dataset` and return a new one via its own public methods,
so they add no new IR — the sugar lowers to existing `select`/`with_columns`/`filter`.

Two modules: `sessions` for session windows, `core` for everything else. The import
path `batcher.api.dataset._build` is unchanged.
"""

from __future__ import annotations

from batcher.api.dataset._build.core import (
    RepartitionSpec,
    _bounded_interval_join,
    build_cast,
    build_distinct,
    build_explode,
    build_pivot,
    build_random_split,
    build_sample,
    build_train_test_split,
    build_unnest,
    build_unpivot,
    build_window,
    build_with_random,
    expand_selector_expr,
    selector_columns,
    split_key,
)
from batcher.api.dataset._build.sessions import (
    build_session_window,
    mark_sessions,
    sessionize,
)

__all__ = [
    "RepartitionSpec",
    "_bounded_interval_join",
    "build_cast",
    "build_distinct",
    "build_explode",
    "build_pivot",
    "build_random_split",
    "build_sample",
    "build_session_window",
    "build_train_test_split",
    "build_unnest",
    "build_unpivot",
    "build_window",
    "build_with_random",
    "expand_selector_expr",
    "mark_sessions",
    "selector_columns",
    "sessionize",
    "split_key",
]
