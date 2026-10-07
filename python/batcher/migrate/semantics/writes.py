"""Save-mode intent for a ported write: carry the source engine's mode, or refuse the rewrite.

Batcher's file writes default to `mode="overwrite"`. A ported `ds.write.<format>(...)` that
names no mode inherits that default, so a Spark job that refused an existing destination, or a
Daft or Ray Data job that appended to one, would silently replace it. A rewrite to a
`Dataset.write.<format>` therefore goes through `carry` and keeps a mode only when it can show
what the source meant:

* an omitted mode becomes the source's default where the registry records it. That is only
  PySpark's `DataFrameWriter`, whose default is `errorifexists` (the `DataFrame.write` row), and
  only on a call made directly on `df.write`: a `.mode(...)` step earlier in the chain is a mode
  the rewrite did not absorb, and a method with its own `overwrite=` (`insertInto`) is not
  governed by the save mode at all;
* an explicit PySpark mode passes through as written, because `ds.write` accepts Spark's
  spellings of it;
* an explicit mode from any other engine passes only as the literal `"append"` or
  `"overwrite"`, which mean the same thing on both sides. A `SaveMode.APPEND` enum or a variable
  proves nothing.

Anything else is refused, and the call is left in the source engine's spelling with a marker,
where it fails on a Batcher `Dataset` rather than writing under a different mode. The check
covers PySpark, Daft and Ray Data; a Polars write ports as its registry row says.
"""

from __future__ import annotations

from typing import Any

from batcher._internal.migration import Mapping
from batcher._internal.optional import require
from batcher.migrate.semantics.base import string
from batcher.migrate.templates import Declined, Signature, literal

cst = require("libcst", feature="batcher.migrate", provides="libcst", extra="migrate")

__all__ = ["REFUSAL", "carry", "is_write", "on_write"]

# The save mode a write uses when it names none, by (engine, surface), where the registry
# records it. Nothing else has a verified default, so nothing else gets one.
_DEFAULTS = {("pyspark", "DataFrameWriter"): "error"}
_PORTABLE = frozenset({"append", "overwrite"})
# The engines the check covers. Polars is outside it: its writes port as their registry rows say.
_GUARDED = frozenset({"pyspark", "daft", "ray_data"})

REFUSAL = (
    "would inherit Batcher's default save mode instead of this write's; pass mode='append' or "
    "mode='overwrite' explicitly, or port it by hand"
)


def is_write(row: Mapping) -> bool:
    """Whether a registry row ports onto a `Dataset.write.<format>` sink.

    Args:
        row: The registry row.

    Returns:
        True for a write onto a named sink (not the bare `Dataset.write` builder).
    """
    return len(row.batcher) == 1 and row.batcher[0].startswith("Dataset.write.")


def on_write(original: Any) -> bool:
    """Whether a call is made directly on a `.write` attribute (`df.write.csv(...)`).

    Args:
        original: The call in the unmodified script.

    Returns:
        True when nothing (a `.mode(...)`, an `.option(...)`) sits between `.write` and it.
    """
    func = getattr(original, "func", None)
    return (
        isinstance(func, cst.Attribute)
        and isinstance(func.value, cst.Attribute)
        and func.value.attr.value == "write"
    )


def carry(engine: str, row: Mapping, source: Signature | None, call: Any, direct: bool) -> Any:
    """The call with a save mode that means what the source's did, or `None` to refuse it.

    Args:
        engine: The source engine.
        row: The registry row the call was looked up under.
        source: The signature the call was written against, when known.
        call: The call as it stands (rewritten, or left as written).
        direct: Whether the original call hangs directly off `.write` (`on_write`).

    Returns:
        The call, with a `mode=` added when the source's default was implied, or `None`.
    """
    if engine not in _GUARDED:
        return call
    default = _DEFAULTS.get((engine, row.surface))
    given = _given_mode(call, source)
    if given is not None:
        if default is not None:
            return call
        try:
            value = literal(given)
        except Declined:
            return None
        return call if value in _PORTABLE else None
    if default is None or not direct or (source is not None and "overwrite" in source.names()):
        return None
    # Append to the call as written, so a call split over lines keeps its layout.
    args = list(call.args)
    if args and args[-1].comma is cst.MaybeSentinel.DEFAULT:
        args[-1] = args[-1].with_changes(
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
        )
    tight = cst.AssignEqual(
        whitespace_before=cst.SimpleWhitespace(""), whitespace_after=cst.SimpleWhitespace("")
    )
    return call.with_changes(
        args=[*args, cst.Arg(string(default), keyword=cst.Name("mode"), equal=tight)]
    )


def _given_mode(call: Any, source: Signature | None) -> Any | None:
    """The mode a call passes, by keyword or in the position the source signature gives it."""
    for arg in call.args:
        if arg.keyword is not None and arg.keyword.value == "mode":
            return arg.value
    names = [n for n, _ in source.positional] if source is not None else []
    plain = [a for a in call.args if a.keyword is None and not a.star]
    if "mode" in names and names.index("mode") < len(plain):
        return plain[names.index("mode")].value
    return None
