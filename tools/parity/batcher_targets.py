"""Resolve a migration-registry target such as `Expr.str.starts_with` on the live surface.

Every registry row that says "this is the Batcher spelling" names a dotted path, and a path
that no longer resolves is a mapping that points users at nothing. The registry package
cannot check that itself: it sits at layer 0 and may not import the engine. So the check is
here, where `tests/unit/test_migration_registry.py` and the census tools can reach it.

A path is a *receiver* followed by one attribute. Receivers are the objects a user holds —
`Dataset`, `Expr`, the accessor namespaces `Expr.str`/`Dataset.ml`, `bt` itself, a public
subpackage such as `batcher.config` — and are matched by longest prefix, so `bt.read.csv`
resolves `csv` on the `Reader` rather than trying `bt.read` as a module. An operator is
written `op:<name>` and resolves when `Expr` defines the matching dunder.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

__all__ = ["OPERATORS", "SURFACE_RECEIVERS", "receivers", "resolve", "unresolved"]

# `op:<name>` → the `Expr` dunder that implements it.
OPERATORS = {
    "add": "__add__",
    "sub": "__sub__",
    "mul": "__mul__",
    "truediv": "__truediv__",
    "floordiv": "__floordiv__",
    "mod": "__mod__",
    "pow": "__pow__",
    "neg": "__neg__",
    "invert": "__invert__",
    "and": "__and__",
    "or": "__or__",
    "xor": "__xor__",
    "eq": "__eq__",
    "ne": "__ne__",
    "lt": "__lt__",
    "le": "__le__",
    "gt": "__gt__",
    "ge": "__ge__",
    "getitem": "__getitem__",
    "lshift": "__lshift__",
    "rshift": "__rshift__",
    "abs": "__abs__",
}

_SUBPACKAGES = ("batcher.config", "batcher.ml", "batcher.io", "batcher.graph", "batcher.governance")

# Which Batcher receiver a user reaches for from each competitor surface. It is runtime data
# (the migration-error guidance reads it), so it lives in the package.
from batcher._internal.migration.hints import SURFACE_RECEIVERS  # noqa: E402


@lru_cache(maxsize=1)
def receivers() -> dict[str, Any]:
    """Every receiver a target path may be rooted at, keyed by its spelling."""
    import batcher as bt
    from batcher.api.merge.builder import MergeBuilder
    from batcher.api.multi_group import MultiLevelGroupBy
    from batcher.api.streaming._query import StreamingQuery
    from batcher.io.manifest import WriteManifest
    from batcher.plan.expr_ir.core import AggExpr, Expr
    from batcher.plan.expr_ir.nodes import CaseBuilder, WindowExpr

    ds = bt.from_pydict({"x": [1]})
    col = bt.col("x")
    out: dict[str, Any] = {
        "bt": bt,
        "bt.read": type(bt.read),
        "Dataset": bt.Dataset,
        "Dataset.write": type(ds.write),
        "Dataset.ml": type(ds.ml),
        "Dataset.dq": type(ds.dq),
        "Dataset.scd": type(ds.scd),
        "Dataset.meta": type(ds.meta),
        "GroupBy": bt.GroupBy,
        "MultiLevelGroupBy": MultiLevelGroupBy,
        "Expr": Expr,
        "AggExpr": AggExpr,
        "WindowExpr": WindowExpr,
        "CaseBuilder": CaseBuilder,
        "Selector": bt.Selector,
        "Session": bt.Session,
        "MergeBuilder": MergeBuilder,
        "StreamingQuery": StreamingQuery,
        "WriteManifest": WriteManifest,
        "Trigger": bt.Trigger,
        "StreamingQueryListener": bt.StreamingQueryListener,
        "Config": bt.Config,
    }
    for ns in ("str", "dt", "list", "struct", "json", "map", "image", "audio", "video", "seq"):
        out[f"Expr.{ns}"] = type(getattr(col, ns))
    for mod in _SUBPACKAGES:
        out[mod] = importlib.import_module(mod)
    return out


def resolve(target: str) -> Any | None:
    """Return the object a target path names, or `None` when it does not resolve.

    Args:
        target: A dotted path rooted at a receiver, or `op:<name>`.

    Returns:
        The attribute, or `None`.
    """
    if target.startswith("op:"):
        dunder = OPERATORS.get(target[3:])
        from batcher.plan.expr_ir.core import Expr

        return getattr(Expr, dunder, None) if dunder else None
    table = receivers()
    head, _, attr = target.rpartition(".")
    if not head:
        return table.get(target)
    root = table.get(head)
    if root is None:
        return None
    # Look on the class dict first so a property or descriptor resolves without calling it.
    for klass in getattr(root, "__mro__", ()):
        if attr in vars(klass):
            return vars(klass)[attr]
    return getattr(root, attr, None)


def unresolved(targets: list[str]) -> list[str]:
    """Return the targets that do not resolve, preserving order.

    Args:
        targets: Target paths to check.

    Returns:
        The subset that `resolve` cannot find.
    """
    return [t for t in targets if resolve(t) is None]
