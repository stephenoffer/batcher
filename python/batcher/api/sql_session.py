"""The SQL `Session` — a context binding named tables, Python functions, and a dialect.

A `Session` is the DuckDB ``con`` / SparkSession analogue: it owns the
control-plane metadata a SQL query resolves against — a table catalog, a registry
of Python functions callable from SQL, and the sqlglot read dialect — and nothing
else. Registering never executes; it only records a plan binding. The module-level
``bt.sql`` / ``bt.register_function`` delegate to a hidden default `Session`, so the
global, zero-setup spelling keeps working while `bt.Session(...)` scopes tables and
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
from batcher._internal.sql_errors import parse_sql
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
    functions to a workload, or use the module-level ``bt.sql`` /
    ``bt.register_function``, which delegate to a shared default `Session`. All state
    is control-plane metadata — registering a table or function never executes anything.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> s = bt.Session()
            >>> _ = s.register("nums", bt.from_pydict({"v": [1, 2, 3]}))
            >>> s.sql("SELECT SUM(v) AS total FROM nums").to_pydict()
            {'total': [6]}
    """

    __slots__ = ("_dialect", "_functions", "_generation", "_plan_cache", "_tables")

    def __init__(self, *, dialect: str = "duckdb") -> None:
        """Create an empty session reading SQL in `dialect` (the sqlglot read dialect)."""
        self._tables: dict[str, Dataset] = {}
        self._functions: dict[str, _RegisteredFunction] = {}
        self._dialect = dialect
        # Prepared-statement cache: (dialect, query, bound names) ->
        # (catalog generation, bound objects, Dataset).
        #
        # A repeated SELECT skips the sqlglot parse + AST translation, which measures
        # ~2.1 ms — the dominant fixed cost of a small query. Two things make a hit safe:
        # the generation bumps on every catalog mutation (register / drop / create /
        # clear / register_function), so a plan never outlives the tables or functions it
        # was built against; and for a query with *per-call* bindings (`ds.sql(...)`,
        # `bt.sql(q, a=ds1)`) the entry stores the bound objects and a hit requires each
        # to be the **identical object** (`is`). Structural equality would not do: two
        # different in-memory Datasets can share a plan shape, and serving one's plan for
        # the other's data is a wrong answer, not a slow one.
        #
        # Storing the bound objects pins them alive, so the cache is capped and evicts
        # oldest-first rather than growing with every dataset a caller queries.
        # `_generation` is a one-slot list, not an int, because `_with_dialect` views
        # share it by reference.
        self._plan_cache: dict[
            tuple[str, str, tuple[str, ...]], tuple[int, tuple[object, ...], Dataset]
        ] = {}
        self._generation: list[int] = [0]

    def __repr__(self) -> str:
        """Show the registered table names, e.g. ``Session(tables=['emp', 'dept'])``."""
        return f"Session(tables={list(self._tables)!r})"

    def __len__(self) -> int:
        """The number of registered tables.

        Returns:
            The count of tables in the session catalog.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1]}))
                >>> len(s)
                1
        """
        return len(self._tables)

    def __contains__(self, name: str) -> bool:
        """Whether a table is registered under `name`.

        Args:
            name: The table name to look up.

        Returns:
            True if a table is registered under `name`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1]}))
                >>> "t" in s
                True
        """
        return name in self._tables

    def __getitem__(self, name: str) -> Dataset:
        """Get a registered table by name — sugar for `table`.

        Args:
            name: The registered table name.

        Returns:
            The `Dataset` registered under `name`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1]}))
                >>> s["t"].columns
                ['x']
        """
        return self.table(name)

    def _bump(self) -> None:
        """Invalidate the prepared-statement cache after a catalog mutation."""
        self._generation[0] += 1

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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
                >>> s.list()
                ['t']
        """
        ds = self._as_dataset(dataset)
        self._tables[name] = ds
        self._bump()
        return ds

    def table(self, name: str) -> Dataset:
        """Return the `Dataset` registered as `name`, raising `PlanError` if absent.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
                >>> s.table("t").to_pydict()
                {'x': [1, 2, 3]}

        Args:
            name: The registered table name to look up.

        Returns:
            The `Dataset` bound to `name`.

        Raises:
            PlanError: If no table is registered under `name`.
        """
        if name not in self._tables:
            raise PlanError(f"no table {name!r} in catalog; registered: {self.list()}")
        return self._tables[name]

    def list(self) -> list[str]:
        """The sorted names of all registered tables.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("b", bt.from_pydict({"x": [2]}))
                >>> _ = s.register("a", bt.from_pydict({"x": [1]}))
                >>> s.list()
                ['a', 'b']

        Returns:
            The sorted list of registered table names.
        """
        return sorted(self._tables)

    def drop(self, name: str) -> None:
        """Remove table `name` from the catalog (no error if absent).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("a", bt.from_pydict({"x": [1]}))
                >>> _ = s.register("b", bt.from_pydict({"x": [2]}))
                >>> s.drop("a")
                >>> s.list()
                ['b']

        Args:
            name: The table name to remove from the catalog.
        """
        self._tables.pop(name, None)
        self._bump()

    def clear(self) -> None:
        """Remove every registered table (registered functions and dialect are kept).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
                >>> s.clear()
                >>> s.list()
                []
        """
        self._tables.clear()
        self._bump()

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
        """Register a Python function callable from SQL (a DuckDB/Spark UDF).

        The DuckDB ``create_function`` / Spark ``udf.register`` analogue.
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

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow.compute as pc
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
                >>> s.register_function("dbl", lambda a: pc.multiply(a, 2), result_type="int64")
                >>> s.sql("SELECT dbl(x) AS y FROM t").to_pydict()
                {'y': [2, 4, 6]}
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
        self._bump()

    def list_functions(self) -> list[str]:
        """The sorted names of all registered functions.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow.compute as pc
                >>> s = bt.Session()
                >>> s.register_function("dbl", lambda a: pc.multiply(a, 2), result_type="int64")
                >>> s.list_functions()
                ['dbl']

        Returns:
            The sorted list of registered function names.
        """
        return sorted(self._functions)

    # --- execution ---------------------------------------------------------
    def sql(self, query: str, **tables: Dataset | pa.Table) -> Dataset:
        """Run `query` against this session's tables, functions, and dialect.

        Keyword `tables` bind or override names for this call only (they do not
        mutate the catalog). ``CREATE TABLE/VIEW AS`` registers a lazy `Dataset`
        into this session and ``DROP TABLE`` unregisters one; ``INSERT`` /
        ``DELETE`` / ``UPDATE`` rebind the target table to its new state (a pure
        plan rewrite — union / filter / projected CASE — that runs only on a later
        terminal op). Everything else is a ``SELECT``-family query. Every form
        returns a lazy `Dataset` — the query result, or the table's new state.

        Args:
            query: A SQL statement.
            **tables: Per-call table bindings (a `Dataset` or pyarrow table each).

        Returns:
            A lazy `Dataset` of the result (the registered relation for DDL).

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("nums", bt.from_pydict({"v": [1, 2, 3]}))
                >>> s.sql("SELECT SUM(v) AS total FROM nums").to_pydict()
                {'total': [6]}
        """
        return self._run(query, tables)

    # --- internals ---------------------------------------------------------
    def _run(self, query: str, tables: dict[str, Dataset | pa.Table]) -> Dataset:
        """Parse and dispatch `query` (tables passed as a dict to allow any name)."""
        # Prepared-statement fast path: re-running the same query text against an
        # unchanged catalog reuses its built plan, skipping the sqlglot parse + AST
        # translation (~2.1 ms). Per-call bindings are cached too — `ds.sql(...)` always
        # passes one, and it is the most-repeated SQL entry point there is — but only
        # when every bound object is the identical object the plan was built over.
        # CREATE/DROP/DML mutate the catalog and are never cached (they bump the
        # generation, which invalidates everything anyway).
        names = tuple(sorted(tables))
        bound = tuple(tables[n] for n in names)
        key = (self._dialect, query, names)
        hit = self._plan_cache.get(key)
        if (
            hit is not None
            and hit[0] == self._generation[0]
            and len(hit[1]) == len(bound)
            and all(a is b for a, b in zip(hit[1], bound, strict=True))
        ):
            return hit[2]

        from sqlglot import expressions as exp

        ast = parse_sql(query, dialect=self._dialect)
        if isinstance(ast, exp.Create):
            return self._create(ast, tables)
        if isinstance(ast, exp.Drop):
            return self._drop(ast)
        if isinstance(ast, (exp.Insert, exp.Delete, exp.Update)):
            return self._dml(ast, tables)
        ds = self._translate(ast, tables)
        self._remember(key, bound, ds)
        return ds

    # How many prepared plans to keep. Entries pin their bound datasets alive, so this is
    # a memory bound, not just a lookup bound: a caller that queries a stream of distinct
    # datasets would otherwise retain every one of them.
    _PLAN_CACHE_MAX = 256

    def _remember(
        self, key: tuple[str, str, tuple[str, ...]], bound: tuple[object, ...], ds: Dataset
    ) -> None:
        """Store a built plan, evicting oldest-first past `_PLAN_CACHE_MAX`."""
        cache = self._plan_cache
        if len(cache) >= self._PLAN_CACHE_MAX and key not in cache:
            for stale in list(cache)[: len(cache) - self._PLAN_CACHE_MAX + 1]:
                del cache[stale]
        cache[key] = (self._generation[0], bound, ds)

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
        self._bump()
        return ds

    def _dml(self, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset:
        """Handle ``INSERT`` / ``DELETE`` / ``UPDATE`` — rebind the target table.

        DML is a pure plan rewrite (union / filter / projected CASE) that produces
        the target table's new lazy state; the catalog is rebound to it and the new
        state is returned. Per-call `tables` bindings are visible to the rewrite but
        the rebind lands on the session catalog (matching ``CREATE``).
        """
        from batcher._sql.dml import apply_dml

        registry = {name: Session._as_dataset(t) for name, t in {**self._tables, **tables}.items()}
        name, new_state = apply_dml(ast, registry, self._functions)
        self._tables[name] = new_state
        self._bump()
        return new_state

    def _drop(self, ast: Any) -> Dataset:
        """Handle ``DROP TABLE [IF EXISTS] name`` — unregister the table."""
        name = ast.this.name
        if not bool(ast.args.get("exists")) and name not in self._tables:
            raise PlanError(f"no table {name!r} to drop")
        self._tables.pop(name, None)
        self._bump()
        return self._as_dataset(pa.table({"dropped": pa.array([name], pa.string())}))

    def _with_dialect(self, dialect: str) -> Session:
        """A view of this session reading `dialect`, sharing its tables and functions.

        Everything mutable is shared *by reference* with the owning session — the
        catalog, the function registry, the plan cache, and the generation counter —
        so a table registered on either is visible to both. Only the read dialect
        differs, and the plan cache is keyed by dialect, so the same query text
        parsed as Spark and as DuckDB cannot collide.
        """
        view = Session.__new__(Session)
        view._tables = self._tables
        view._functions = self._functions
        view._dialect = dialect
        view._plan_cache = self._plan_cache
        view._generation = self._generation
        return view

    @staticmethod
    def _as_dataset(dataset: Dataset | pa.Table) -> Dataset:
        if isinstance(dataset, Dataset):
            return dataset
        if isinstance(dataset, pa.Table):
            from batcher.api.session import from_arrow

            return from_arrow(dataset)
        raise PlanError(f"table must be a Dataset or pyarrow.Table, got {type(dataset).__name__}")
