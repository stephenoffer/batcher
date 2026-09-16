"""The migration half of an `AttributeError`: what a removed or foreign spelling is called here.

When an attribute lookup fails on `Dataset`, `Expr`, an accessor, `GroupBy` or `bt` itself, the
traceback is the one piece of documentation a migrating user reads at that moment. The curated
tables beside each hook answer for the idioms they know about (mostly pandas). This module
answers for everything the migration registry knows, so the answer cannot drift from the codemod
and the generated docs, which read the same rows:

* a spelling Batcher removed (`ds.groupby`) names the spelling that replaced it and the codemod
  that rewrites it;
* a PySpark, Polars, Daft or Ray Data spelling typed on the Batcher object a user of that engine
  would reach for (`ds.withColumn`) names the Batcher spelling, the gap, the difference, or the
  reason it is declined, per engine.

It is message-only: nothing here changes what a query computes.
"""

from __future__ import annotations

from functools import lru_cache

from batcher._internal.migration.loader import load_registry
from batcher._internal.migration.renames import OPERATORS, Rename, load_renames
from batcher._internal.migration.schema import ENGINES, Mapping, Status

__all__ = ["SURFACE_RECEIVERS", "migration_hint"]

_ENGINE_LABEL = {"pyspark": "PySpark", "polars": "Polars", "daft": "Daft", "ray_data": "Ray Data"}

# Which Batcher receiver a user of each engine reaches for when they type a name from one of
# that engine's surfaces. Surfaces with no Batcher counterpart (Spark's `types`, Ray's
# `DataContext`) are absent. `tools/parity/batcher_targets.py` re-exports this for the tests.
SURFACE_RECEIVERS: dict[tuple[str, str], str] = {
    ("pyspark", "DataFrame"): "Dataset",
    ("pyspark", "DataFrameNaFunctions"): "Dataset",
    ("pyspark", "DataFrameStatFunctions"): "Dataset",
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
    ("polars", "LazyFrame"): "Dataset",
    ("polars", "DataFrame"): "Dataset",
    ("polars", "GroupBy"): "GroupBy",
    ("polars", "LazyGroupBy"): "GroupBy",
    ("polars", "Expr"): "Expr",
    ("polars", "Expr.str"): "Expr.str",
    ("polars", "Expr.dt"): "Expr.dt",
    ("polars", "Expr.list"): "Expr.list",
    ("polars", "Expr.struct"): "Expr.struct",
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
    ("ray_data", "Expr.str"): "Expr.str",
    ("ray_data", "Expr.list"): "Expr.list",
    ("ray_data", "Expr.dt"): "Expr.dt",
    ("ray_data", "Expr.struct"): "Expr.struct",
    ("ray_data", "Expr.map"): "Expr.map",
}

_CODEMOD = "`python -m batcher.migrate` rewrites existing code"


def _spelled(target: str) -> str:
    if target.startswith("op:"):
        return f"the `{OPERATORS.get(target[3:], target[3:])}` operator"
    return f"`{target}`"


def _removed(rule: Rename) -> str:
    if rule.kind == "operator":
        kept = f"the `{OPERATORS[rule.operator]}` operator"
    elif rule.kind == "call":
        kept = f"`.{rule.to}()`"
    else:
        kept = f"`.{rule.to}`"
    return f"This second spelling was removed: use {kept}; {_CODEMOD}."


def _row_text(row: Mapping) -> str:
    targets = " with ".join(_spelled(t) for t in row.batcher)
    label = _ENGINE_LABEL[row.engine]
    if row.status in (Status.CANONICAL, Status.ALIAS):
        return f"{label}'s `{row.name}` is {targets} here."
    if row.status is Status.PARAM:
        return f"{label}'s `{row.name}` is {targets} here, without: {row.need}."
    if row.status is Status.MISMATCH:
        return f"{label}'s `{row.name}` is closest to {targets}, but: {row.note}."
    if row.status is Status.GAP:
        return f"{label}'s `{row.name}` has no Batcher equivalent yet ({row.need})."
    return f"{label}'s `{row.name}` is not provided: {row.reason}."


@lru_cache(maxsize=1)
def _by_receiver() -> dict[tuple[str, str], tuple[Mapping, ...]]:
    grouped: dict[tuple[str, str], list[Mapping]] = {}
    for row in load_registry().rows.values():
        receiver = SURFACE_RECEIVERS.get((row.engine, row.surface))
        if receiver is not None:
            grouped.setdefault((receiver, row.name), []).append(row)
    order = {engine: i for i, engine in enumerate(ENGINES)}
    return {k: tuple(sorted(v, key=lambda r: order[r.engine])) for k, v in grouped.items()}


def migration_hint(receiver: str, name: str) -> str | None:
    """The guidance for a failed `receiver.name` lookup, from the renames and the registry.

    Args:
        receiver: The receiver key, such as `"Dataset"`, `"Expr.str"` or `"bt"`.
        name: The attribute name that was not found.

    Returns:
        One or more sentences naming what to type instead, or `None` when neither the
        renames nor the registry know the name.

    Examples:
        .. doctest::

            >>> from batcher._internal.migration.hints import migration_hint
            >>> print(migration_hint("Dataset", "zzz_not_a_name"))
            None
    """
    rule = load_renames().get(receiver, {}).get(name)
    parts = [_removed(rule)] if rule is not None else []
    seen: set[str] = set()
    for row in _by_receiver().get((receiver, name), ()):
        text = _row_text(row)
        if text not in seen:
            seen.add(text)
            parts.append(text)
    return " ".join(parts) or None
