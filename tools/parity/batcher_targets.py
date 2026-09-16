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

# Which Batcher receiver a user reaches for when they type a name from a competitor surface.
# It is what the no-alias check looks the *competitor's* spelling up on: if `withColumn` were
# ever an attribute of `Dataset`, that would be a second spelling, whatever the row says.
# Surfaces with no Batcher receiver (Spark's `types`, Ray's `DataContext`) are absent.
SURFACE_RECEIVERS: dict[tuple[str, str], str] = {
    **{
        ("pyspark", s): "Dataset"
        for s in ("DataFrame", "DataFrameNaFunctions", "DataFrameStatFunctions")
    },
    ("pyspark", "GroupedData"): "GroupBy",
    ("pyspark", "Column"): "Expr",
    ("pyspark", "functions"): "bt",
    ("pyspark", "WindowSpec"): "WindowExpr",
    ("pyspark", "DataFrameReader"): "bt.read",
    ("pyspark", "DataStreamReader"): "bt.read",
    ("pyspark", "DataFrameWriter"): "Dataset.write",
    ("pyspark", "DataFrameWriterV2"): "Dataset.write",
    ("pyspark", "DataStreamWriter"): "Dataset.write",
    ("pyspark", "StreamingQuery"): "StreamingQuery",
    ("pyspark", "SparkSession"): "Session",
    ("pyspark", "Catalog"): "Session",
    **{("polars", s): "Dataset" for s in ("LazyFrame", "DataFrame")},
    **{("polars", s): "GroupBy" for s in ("GroupBy", "LazyGroupBy")},
    ("polars", "Expr"): "Expr",
    **{("polars", f"Expr.{ns}"): f"Expr.{ns}" for ns in ("str", "dt", "list", "struct")},
    ("polars", "polars"): "bt",
    ("polars", "SQLContext"): "Session",
    ("daft", "DataFrame"): "Dataset",
    ("daft", "Expression"): "Expr",
    ("daft", "functions"): "bt",
    ("daft", "GroupedDataFrame"): "GroupBy",
    ("daft", "daft"): "bt",
    ("daft", "Session"): "Session",
    ("ray_data", "Dataset"): "Dataset",
    ("ray_data", "GroupedData"): "GroupBy",
    ("ray_data", "ray.data"): "bt",
    ("ray_data", "Expr"): "Expr",
    **{("ray_data", f"Expr.{ns}"): f"Expr.{ns}" for ns in ("str", "list", "dt", "struct", "map")},
}


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
