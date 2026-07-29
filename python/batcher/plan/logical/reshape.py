"""Row-reshaping logical nodes — `plan`, the neutral contract layer.

`Unnest` (explode a list column into one row per element), `RowId` (append a
sequential row index), and `Unpivot` (wide → long / melt). Each restructures the
*shape* of a row rather than filtering or combining relations, which is why they sit
beside — not inside — `relational`. All three are streaming and stateless; each node
keeps its dataclass, its validation, and its `to_ir()` together, and the IR tags come
from `plan.ir_tags`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.plan.ir_tags import Op
from batcher.plan.logical.base import LogicalPlan
from batcher.plan.schema import SchemaRef
from batcher.plan.types import promote

__all__ = [
    "RowId",
    "Unnest",
    "Unpivot",
]


@dataclass(frozen=True, slots=True)
class Unnest(LogicalPlan):
    """Explode a list/array column into one row per element (SQL ``UNNEST`` /
    DataFrame ``explode``).

    The named `column` is replaced in place by its element values bound to `alias`;
    every other column repeats once per element. Null and empty lists produce no
    output rows (DuckDB ``UNNEST`` semantics) unless `outer` is set. Streaming and
    stateless.

    `outer` keeps a row whose list is null or empty, with a NULL element (Spark
    ``explode_outer``). `index_alias`, when given, adds a column holding each element's
    0-based position in its list (Spark ``posexplode``) — NULL for a row kept only by
    `outer`, which has no element.
    """

    input: LogicalPlan
    column: str
    alias: str
    outer: bool = False
    index_alias: str | None = None

    def __post_init__(self) -> None:
        available = self.input.available_columns()
        if self.column not in available:
            raise PlanError(
                f"unnest column {self.column!r} not found in input columns: {available}"
            )
        # The exploded column is renamed to `alias` in place; if `alias` names a *different*
        # existing column the output would carry two same-named columns and silently drop
        # one. Reject the collision instead of losing data.
        if self.alias != self.column and self.alias in available:
            raise PlanError(
                f"explode alias {self.alias!r} collides with an existing column: {available}"
            )
        # Same reasoning for the index column, which is *appended*: a collision would
        # produce two same-named columns rather than overwriting one.
        if self.index_alias is not None and self.index_alias in (*available, self.alias):
            raise PlanError(
                f"explode index name {self.index_alias!r} collides with an existing column: "
                f"{available}"
            )

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.UNNEST,
            "input": self.input.to_ir(),
            "column": self.column,
            "alias": self.alias,
            "outer": self.outer,
            "index_alias": self.index_alias,
        }

    def available_columns(self) -> list[str]:
        cols = [self.alias if c == self.column else c for c in self.input.available_columns()]
        return [*cols, self.index_alias] if self.index_alias is not None else cols

    def available_schema(self) -> SchemaRef | None:
        inp = self.input.available_schema()
        if inp is None:
            return None
        list_t = inp.field(self.column).type
        if not (
            pa.types.is_list(list_t)
            or pa.types.is_large_list(list_t)
            or pa.types.is_fixed_size_list(list_t)
        ):
            return None  # unnest of a non-list: leave it to the engine
        fields = [
            pa.field(self.alias, list_t.value_type) if f.name == self.column else f
            for f in inp.arrow
        ]
        if self.outer:
            # An `outer` row carries a NULL element, so the exploded column is nullable
            # even when the list's own value type was declared non-nullable.
            fields = [f.with_nullable(True) if f.name == self.alias else f for f in fields]
        if self.index_alias is not None:
            fields.append(pa.field(self.index_alias, pa.int64(), nullable=True))
        return SchemaRef.from_arrow(pa.schema(fields))


@dataclass(frozen=True, slots=True)
class RowId(LogicalPlan):
    """Append a 0-based (plus `offset`) sequential row-index column (Polars
    ``with_row_index``).

    The index numbers rows in arrival order across the whole input via one sequential
    counter, so it is identical on the single-node and parallel paths for an
    order-preserving pipeline. Streaming.
    """

    input: LogicalPlan
    alias: str
    offset: int = 0

    def __post_init__(self) -> None:
        if self.alias in self.input.available_columns():
            raise PlanError(f"with_row_index name {self.alias!r} collides with an existing column")

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.ROW_ID,
            "input": self.input.to_ir(),
            "alias": self.alias,
            "offset": self.offset,
        }

    def available_columns(self) -> list[str]:
        return [self.alias, *self.input.available_columns()]

    def available_schema(self) -> SchemaRef | None:
        inp = self.input.available_schema()
        if inp is None:
            return None
        fields = [pa.field(self.alias, pa.int64()), *inp.arrow]
        return SchemaRef.from_arrow(pa.schema(fields))


@dataclass(frozen=True, slots=True)
class Unpivot(LogicalPlan):
    """Reshape wide → long (SQL ``UNPIVOT`` / pandas ``melt`` / Polars ``unpivot``).

    Each input row becomes one row per `on` column: the `index` columns repeat, the
    `variable_name` column holds the melted column's name, and `value_name` holds its
    value. The `on` columns must share a type. Streaming and stateless.
    """

    input: LogicalPlan
    index: tuple[str, ...]
    on: tuple[str, ...]
    variable_name: str
    value_name: str

    def __post_init__(self) -> None:
        available = self.input.available_columns()
        # `index` carries every column the unpivot keeps, so this checks O(width) names
        # against O(width) columns — quadratic against a list, linear against a set. The
        # list itself is kept for the error message, which reports the columns in order.
        present = set(available)
        missing = [c for c in (*self.index, *self.on) if c not in present]
        if missing:
            raise PlanError(f"unpivot columns {missing} not found in input columns: {available}")
        if not self.on:
            raise PlanError("unpivot requires at least one column in `on`")
        # The output columns are [*index, variable_name, value_name]; a collision among
        # them (e.g. value_name == an index column) would produce two columns of the same
        # name and silently drop one on the way out. Reject it, like Polars does.
        out = [*self.index, self.variable_name, self.value_name]
        # Counted in one pass: `index` is every column the unpivot carries through, so a
        # `list.count()` per element made validating a wide unpivot quadratic in its width.
        seen = Counter(out)
        dups = sorted(c for c, n in seen.items() if n > 1)
        if dups:
            raise PlanError(
                f"unpivot output columns collide: {dups} — variable_name/value_name must "
                "differ from each other and from every index column"
            )
        # The melted `on` columns stack into one `value` column, so they must share a
        # promotable type. Reject an unmergeable mix (e.g. Utf8 + Int64) here with a clear
        # plan-time error — else native `concat` fails deep in execution with an opaque
        # "cannot concatenate arrays of different data types". DuckDB rejects it at bind
        # time too. Best-effort: only when the input schema (column types) is known.
        schema = self.input.available_schema()
        if schema is not None and self.available_schema() is None:
            raise PlanError(
                f"unpivot cannot merge `on` columns of incompatible types "
                f"({[str(schema.field(c).type) for c in self.on]}); cast them to a "
                "common type before unpivoting"
            )

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": Op.UNPIVOT,
            "input": self.input.to_ir(),
            "index": list(self.index),
            "on": list(self.on),
            "variable_name": self.variable_name,
            "value_name": self.value_name,
        }

    def available_columns(self) -> list[str]:
        return [*self.index, self.variable_name, self.value_name]

    def available_schema(self) -> SchemaRef | None:
        inp = self.input.available_schema()
        if inp is None:
            return None
        value_t = inp.field(self.on[0]).type
        for c in self.on[1:]:  # the melted columns share a (promotable) type
            value_t = promote(value_t, inp.field(c).type)
            if value_t is None:
                return None
        fields = [inp.field(c) for c in self.index]
        fields.append(pa.field(self.variable_name, pa.string()))
        fields.append(pa.field(self.value_name, value_t))
        return SchemaRef.from_arrow(pa.schema(fields))
