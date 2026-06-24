"""The SQL `Session` — a context binding named tables, Python functions, and a dialect.

A `Session` is the DuckDB ``con`` / SparkSession analogue: it owns the
control-plane metadata a SQL query resolves against — a table catalog, a registry
of Python functions callable from SQL, and the sqlglot read dialect — and nothing
else. Registering never executes; it only records a plan binding. The module-level
``bt.sql`` / ``bt.catalog`` delegate to a hidden default `Session`, so the global,
zero-setup spelling keeps working while `bt.Session(...)` scopes tables and
functions to a single workload.

This is the `api` layer: it builds `Dataset`s and calls the `_sql` translator. It
imports no subsystem (`kyber`/`carbonite`/`core`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset

__all__ = ["Session"]


@dataclass(frozen=True)
class _RegisteredFunction:
    """A Python function registered for use in SQL, plus how it lowers to `map_batches`.

    `table` selects the call form: a table function ``SELECT * FROM f(t)`` (whole
    relation in, relation out) when true, else a scalar function ``SELECT f(x)``
    hoisted into a column-materializing `map_batches`. `vectorized` (scalar form)
    chooses whether `fn` receives whole Arrow arrays or one row at a time; `per_row`
    is the table-form analogue (``ds.ml.map`` vs ``ds.ml.map_batches``).
    """

    name: str
    fn: Callable
    table: bool
    per_row: bool
    vectorized: bool
    result_type: pa.DataType | None
    output_columns: tuple[str, ...] | None
    config: dict[str, Any] = field(default_factory=dict)


def _resolve_type(result_type: str | pa.DataType | None) -> pa.DataType | None:
    """Resolve a declared result type (an Arrow type or its string alias) to a type."""
    if result_type is None or isinstance(result_type, pa.DataType):
        return result_type
    return pa.type_for_alias(result_type)


class Session:
    """A SQL execution context: a table catalog, a Python-function registry, and a dialect.

    Mirrors DuckDB's ``con`` and SparkSession. Build one to scope tables and
    functions to a workload, or use the module-level ``bt.sql`` / ``bt.catalog``,
    which delegate to a shared default `Session`. All state is control-plane
    metadata — registering a table or function never executes anything.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> s = bt.Session()
            >>> _ = s.register("nums", bt.from_pydict({"v": [1, 2, 3]}))
            >>> s.sql("SELECT SUM(v) AS total FROM nums").to_pydict()
            {'total': [6]}
    """

    __slots__ = ("_dialect", "_functions", "_tables")

    def __init__(self, *, dialect: str = "duckdb") -> None:
        """Create an empty session reading SQL in `dialect` (the sqlglot read dialect)."""
        self._tables: dict[str, Dataset] = {}
        self._functions: dict[str, _RegisteredFunction] = {}
        self._dialect = dialect

    # --- tables ------------------------------------------------------------
    def register(self, name: str, dataset: Dataset | pa.Table) -> Dataset:
        """Register `dataset` as the table `name` for this session, replacing any prior.

        The DuckDB ``con.register`` / Spark ``createOrReplaceTempView`` analogue. A
        pyarrow table is lifted to a `Dataset`.

        Args:
            name: The table name SQL queries will refer to.
            dataset: A `Dataset` or pyarrow table to bind.

        Returns:
            The bound `Dataset`.
        """
        ds = self._as_dataset(dataset)
        self._tables[name] = ds
        return ds

    def table(self, name: str) -> Dataset:
        """Return the `Dataset` registered as `name`, raising `PlanError` if absent."""
        if name not in self._tables:
            raise PlanError(f"no table {name!r} in catalog; registered: {self.list()}")
        return self._tables[name]

    def list(self) -> list[str]:
        """The sorted names of all registered tables."""
        return sorted(self._tables)

    def drop(self, name: str) -> None:
        """Remove table `name` from the catalog (no error if absent)."""
        self._tables.pop(name, None)

    def clear(self) -> None:
        """Remove every registered table (registered functions and dialect are kept)."""
        self._tables.clear()

    # --- functions ---------------------------------------------------------
    def register_function(
        self,
        name: str,
        fn: Callable,
        *,
        table: bool = False,
        per_row: bool = False,
        vectorized: bool = True,
        result_type: str | pa.DataType | None = None,
        output_columns: list[str] | None = None,
        batch_format: str = "pyarrow",
        **config: Any,
    ) -> None:
        """Register a Python function callable from SQL (DuckDB ``create_function`` /
        Spark ``udf.register``).

        Python cannot run inside the engine's expression evaluator, so the function
        lowers to a `map_batches` stage. Two call forms are supported:

        * scalar (default) — ``SELECT f(x)`` / ``WHERE f(x)``. `vectorized=True`
          (the fast default) passes whole Arrow arrays to `fn` and expects an array
          back; `vectorized=False` calls ``fn(*scalars)`` per row. Declare
          `result_type` (an Arrow type or alias like ``"int64"``) — required for the
          per-row form, optional for vectorized (inferred from the returned array).
        * table — ``SELECT * FROM f(t)``, set ``table=True``. `fn` follows the
          `map_batches` contract (batch in, batch out) unless ``per_row=True``;
          `output_columns` declares the result schema and `batch_format`/extra
          ``config`` forward to `map_batches`.

        Scalar functions are not supported in ``GROUP BY`` keys, aggregate arguments,
        or ``ORDER BY`` — compute them in a subquery or projected alias first.

        Args:
            name: The SQL name the function is called by.
            fn: The Python callable.
            table: Register as a table function rather than a scalar function.
            per_row: Table form only — apply row-by-row instead of per batch.
            vectorized: Scalar form only — pass Arrow arrays (else per-row scalars).
            result_type: Scalar output Arrow type (or alias).
            output_columns: Table-function result column names.
            batch_format: Table form `map_batches` batch format.
            **config: Extra `map_batches` keyword arguments (table form).
        """
        if table:
            config = {"batch_format": batch_format, **config}
        self._functions[name] = _RegisteredFunction(
            name=name,
            fn=fn,
            table=table,
            per_row=per_row,
            vectorized=vectorized,
            result_type=_resolve_type(result_type),
            output_columns=tuple(output_columns) if output_columns is not None else None,
            config=config,
        )

    def list_functions(self) -> list[str]:
        """The sorted names of all registered functions."""
        return sorted(self._functions)

    # --- execution ---------------------------------------------------------
    def sql(self, query: str, **tables: Dataset | pa.Table) -> Dataset:
        """Run `query` against this session's tables, functions, and dialect.

        Keyword `tables` bind or override names for this call only (they do not
        mutate the catalog). ``CREATE TABLE/VIEW AS`` registers a lazy `Dataset`
        into this session; ``DROP TABLE`` unregisters one. Everything else is a
        ``SELECT``-family query returning a lazy `Dataset`.

        Args:
            query: A SQL statement.
            **tables: Per-call table bindings (a `Dataset` or pyarrow table each).

        Returns:
            A lazy `Dataset` of the result (the registered relation for DDL).
        """
        return self._run(query, tables)

    # --- internals ---------------------------------------------------------
    def _run(self, query: str, tables: dict[str, Dataset | pa.Table]) -> Dataset:
        """Parse and dispatch `query` (tables passed as a dict to allow any name)."""
        import sqlglot
        from sqlglot import expressions as exp

        ast = sqlglot.parse_one(query, read=self._dialect)
        if isinstance(ast, exp.Create):
            return self._create(ast, tables)
        if isinstance(ast, exp.Drop):
            return self._drop(ast)
        return self._translate(ast, tables)

    def _translate(self, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset:
        from batcher._sql import translate_ast

        return translate_ast(ast, functions=self._functions, **{**self._tables, **tables})

    def _create(self, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset:
        """Handle ``CREATE [OR REPLACE] {TABLE|VIEW} name AS <select>`` — register lazily.

        Both forms register a *lazy* `Dataset` (Batcher is lazy throughout, so
        ``CREATE TABLE AS`` does not materialize — a terminal op does).
        """
        name = ast.this.name
        if not bool(ast.args.get("replace")) and name in self._tables:
            raise PlanError(f"table {name!r} already exists; use CREATE OR REPLACE")
        body = ast.expression
        if body is None:
            raise PlanError("CREATE TABLE/VIEW requires an AS <select> body")
        ds = self._translate(body, tables)
        self._tables[name] = ds
        return ds

    def _drop(self, ast: Any) -> Dataset:
        """Handle ``DROP TABLE [IF EXISTS] name`` — unregister the table."""
        name = ast.this.name
        if not bool(ast.args.get("exists")) and name not in self._tables:
            raise PlanError(f"no table {name!r} to drop")
        self._tables.pop(name, None)
        return self._as_dataset(pa.table({"dropped": pa.array([name], pa.string())}))

    def _with_dialect(self, dialect: str) -> Session:
        """A view of this session reading `dialect`, sharing its tables and functions."""
        view = Session.__new__(Session)
        view._tables = self._tables
        view._functions = self._functions
        view._dialect = dialect
        return view

    @staticmethod
    def _as_dataset(dataset: Dataset | pa.Table) -> Dataset:
        if isinstance(dataset, Dataset):
            return dataset
        if isinstance(dataset, pa.Table):
            from batcher.api.session import from_arrow

            return from_arrow(dataset)
        raise PlanError(f"table must be a Dataset or pyarrow.Table, got {type(dataset).__name__}")
