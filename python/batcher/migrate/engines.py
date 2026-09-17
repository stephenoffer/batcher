"""What the foreign-engine codemod directions know about each engine besides the registry.

The registry says what every PySpark, Polars, Daft and Ray Data name *means* in Batcher. To act
on a row the codemod also needs a little about each engine's shape: how its objects enter a
script (`Seeds`), which of its surfaces are modules or sessions whose value can be dropped when
the Batcher spelling hangs off `bt`, which surfaces hold column expressions, and what to import
when writing code *for* it. That is this module, one `EngineSpec` per engine, plus `Tables`,
which joins a spec with the registry rows and the generated signature tables in
`data/codemod/`.

It is data, not rules: nothing here decides a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from batcher._internal.migration import Mapping, load_codemod_tables, load_registry, load_returns
from batcher._internal.migration.hints import SURFACE_RECEIVERS
from batcher.migrate.receivers import Seeds

__all__ = ["ENGINE_LABELS", "SPECS", "EngineSpec", "Tables", "tables"]

ENGINE_LABELS = {"pyspark": "PySpark", "polars": "Polars", "daft": "Daft", "ray_data": "Ray Data"}


@dataclass(frozen=True)
class EngineSpec:
    """One foreign engine's shape, as the codemod needs it.

    Attributes:
        name: The registry engine name.
        seeds: Where the engine's objects enter a script.
        modules: Module surfaces (called as functions), to the expression that reaches them
            when writing code for the engine (`{"functions": "F"}`).
        droppable: Surfaces whose value carries no data (a module, a session), so a Batcher
            spelling rooted at `bt` may replace the whole expression.
        expressions: Surfaces whose values are column expressions.
        imports: The import lines code written for the engine starts with.
        objects: The surfaces whose methods a Batcher method may become when writing code for
            the engine, most preferred first. `df.na.fill` hangs off a `DataFrame`, so
            `DataFrameNaFunctions` is absent, and a tie between `DataFrame` and `LazyFrame`
            goes to the one listed first.
        session: A statement that binds the engine's session, for spellings that hang off it.
    """

    name: str
    seeds: Seeds
    modules: dict[str, str]
    droppable: frozenset[str]
    expressions: frozenset[str]
    imports: tuple[str, ...]
    objects: tuple[str, ...]
    session: tuple[str, str] | None = None


SPECS: dict[str, EngineSpec] = {
    "pyspark": EngineSpec(
        name="pyspark",
        seeds=Seeds(
            package="pyspark",
            modules={
                "pyspark.sql.functions": "functions",
                "pyspark.sql.SparkSession": "SparkSession.class",
                "pyspark.sql.session.SparkSession": "SparkSession.class",
                "pyspark.sql.Window": "Window",
                "pyspark.sql.window.Window": "Window",
                "pyspark.sql.types": "types",
            },
            classes={
                "DataFrame": "DataFrame",
                "Column": "Column",
                "GroupedData": "GroupedData",
                "WindowSpec": "WindowSpec",
            },
            subscripts={"DataFrame": "Column", "Column": "Column"},
            operands=frozenset({"Column"}),
            operator_result="Column",
            attribute_columns={"DataFrame": "Column"},
        ),
        modules={"functions": "F"},
        droppable=frozenset({"functions", "SparkSession", "SparkSession.class", "Window"}),
        expressions=frozenset({"Column"}),
        imports=("from pyspark.sql import functions as F",),
        objects=("DataFrame", "Column", "GroupedData"),
        session=("SparkSession", "spark = SparkSession.builder.getOrCreate()"),
    ),
    "polars": EngineSpec(
        name="polars",
        seeds=Seeds(
            package="polars",
            modules={"polars": "polars", "polars.selectors": "selectors"},
            classes={"DataFrame": "DataFrame", "LazyFrame": "LazyFrame", "Expr": "Expr"},
            operands=frozenset({"Expr"}),
            operator_result="Expr",
        ),
        modules={"polars": "pl", "selectors": "pl.selectors"},
        droppable=frozenset({"polars", "selectors"}),
        expressions=frozenset({"Expr"}),
        imports=("import polars as pl",),
        objects=(
            "DataFrame",
            "LazyFrame",
            "GroupBy",
            "LazyGroupBy",
            "Expr",
            "Expr.str",
            "Expr.dt",
            "Expr.list",
            "Expr.struct",
        ),
    ),
    "daft": EngineSpec(
        name="daft",
        seeds=Seeds(
            package="daft",
            modules={"daft": "daft", "daft.functions": "functions"},
            classes={"DataFrame": "DataFrame", "Expression": "Expression"},
            subscripts={"DataFrame": "Expression"},
            operands=frozenset({"Expression"}),
            operator_result="Expression",
        ),
        modules={"daft": "daft", "functions": "daft.functions"},
        droppable=frozenset({"daft", "functions", "Session"}),
        expressions=frozenset({"Expression"}),
        imports=("import daft",),
        objects=("DataFrame", "GroupedDataFrame", "Expression"),
    ),
    "ray_data": EngineSpec(
        name="ray_data",
        seeds=Seeds(
            package="ray",
            modules={
                "ray": "ray",
                "ray.data": "ray.data",
                "ray.data.expressions": "expressions",
                "ray.data.aggregate": "aggregate",
            },
            classes={"Dataset": "Dataset"},
            operands=frozenset({"Expr"}),
            operator_result="Expr",
        ),
        modules={"ray.data": "ray.data", "expressions": "ray.data.expressions"},
        droppable=frozenset({"ray", "ray.data", "expressions", "aggregate"}),
        expressions=frozenset({"Expr"}),
        imports=("import ray.data", "import ray.data.expressions"),
        objects=(
            "Dataset",
            "GroupedData",
            "Expr",
            "Expr.str",
            "Expr.dt",
            "Expr.list",
            "Expr.struct",
            "Expr.map",
        ),
    ),
}


@dataclass(frozen=True)
class Tables:
    """Everything the codemod reads for one engine, loaded once.

    Attributes:
        spec: The engine's shape.
        rows: Registry rows by `(surface, name)`.
        returns: The engine's generated returns table.
        params: The engine's generated parameter tokens, `{surface: {member: tokens}}`.
        batcher_params: Batcher's parameter tokens, `{receiver: {member: tokens}}`.
        batcher_returns: Batcher's returns table.
    """

    spec: EngineSpec
    rows: dict[tuple[str, str], Mapping]
    returns: dict[str, dict[str, str]]
    params: dict[str, dict[str, list[str]]]
    batcher_params: dict[str, dict[str, list[str]]]
    batcher_returns: dict[str, dict[str, str]]

    def row(self, surface: str, name: str) -> Mapping | None:
        """The registry row for `name` on `surface`, or `None`.

        Args:
            surface: A registry surface of this engine.
            name: The engine's spelling.

        Returns:
            The row, or `None` when unclassified.
        """
        return self.rows.get((surface, name))

    def batcher_receiver(self, surface: str) -> str | None:
        """The Batcher receiver a user of this engine's `surface` reaches for.

        Args:
            surface: A registry surface of this engine.

        Returns:
            The receiver, such as `Dataset`, or `None` when the surface has no counterpart.
        """
        return SURFACE_RECEIVERS.get((self.spec.name, surface))

    def members(self) -> dict[str, frozenset[str]]:
        """Every member name the engine's generated tables know, per surface.

        Returns:
            `{surface: names}`, which the inference uses to tell a column attribute from a
            method.
        """
        return {s: frozenset(m) for s, m in self.params.items()}

    def split_batcher(self, target: str) -> tuple[str, str] | None:
        """Split a Batcher target path into its receiver and member.

        Args:
            target: A dotted registry target, such as `bt.read.csv`.

        Returns:
            `(receiver, member)` by the longest receiver the parameter table knows, or `None`.
        """
        head, _, member = target.rpartition(".")
        while head:
            if head in self.batcher_params or head in self.batcher_returns:
                return head, member
            head, _, rest = head.rpartition(".")
            member = f"{rest}.{member}"
        return None


@lru_cache(maxsize=8)
def tables(engine: str) -> Tables:
    """Load every table the codemod reads for one engine.

    Args:
        engine: One of the registry engines.

    Returns:
        The joined tables.
    """
    generated = load_codemod_tables(engine)
    rows = {(r.surface, r.name): r for r in load_registry().for_engine(engine)}
    return Tables(
        spec=SPECS[engine],
        rows=rows,
        returns=generated["returns"],  # type: ignore[arg-type]
        params=generated["params"],  # type: ignore[arg-type]
        batcher_params=load_codemod_tables("batcher")["params"],  # type: ignore[arg-type]
        batcher_returns=load_returns(),
    )
