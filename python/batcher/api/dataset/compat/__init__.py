"""Ecosystem-compatibility bodies behind the `Dataset` migration surface.

`Dataset` stays a thin fluent builder (the v2 maintainability contract), so the
pandas/Polars/Spark-shaped conveniences it exposes delegate their bodies here, the
way the relational sugar delegates to `_build.py`. Nothing in this package adds
IR: every function composes the same public `Dataset` methods a user could have
called, so a migrated script and a native one produce the identical plan.

Three responsibilities, one module each:

* `rows` — row-oriented terminal consumers (`iter_rows`, `first`, `item`). These
  materialize, so they are terminal *consumers*, never hot-path row work.
* `introspect` — the "what am I holding?" surface (`info`, `glimpse`,
  `memory_usage`, `collect_schema`) that a REPL user reaches for.
* `guidance` — the actionable-error table for ecosystem APIs Batcher
  deliberately does not have (anything index-centric, anything per-row-mutable).
  A migrant who types `ds.set_index("k")` gets told why it is absent and what to
  type instead, rather than a bare `AttributeError`.
"""

from __future__ import annotations

from batcher.api.dataset.compat.guidance import attribute_error_for
from batcher.api.dataset.compat.introspect import (
    build_collect_schema,
    build_glimpse,
    build_info,
    build_memory_usage,
)
from batcher.api.dataset.compat.rows import (
    build_first,
    build_item,
    build_iter_rows,
    build_iter_slices,
    build_last,
)

__all__ = [
    "attribute_error_for",
    "build_collect_schema",
    "build_first",
    "build_glimpse",
    "build_info",
    "build_item",
    "build_iter_rows",
    "build_iter_slices",
    "build_last",
    "build_memory_usage",
]
