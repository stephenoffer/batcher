"""The SQL `Session` — a context binding named tables, Python functions, and a dialect.

A `Session` is the DuckDB ``con`` / SparkSession analogue: it owns the control-plane
metadata a SQL query resolves against — a table catalog, a registry of Python functions
callable from SQL, and the sqlglot read dialect — and nothing else. Registering never
executes; it only records a plan binding. The module-level ``bt.sql`` /
``bt.register_function`` delegate to a hidden default `Session`, so the global, zero-setup
spelling keeps working while `bt.Session(...)` scopes tables and functions to a workload.

This is the `api` layer: it builds `Dataset`s and calls the `_sql` translator. It imports
no subsystem (`kyber`/`carbonite`/`core`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextvars import ContextVar
from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError, SQLSyntaxError, SQLUnsupportedError
from batcher._internal.sql_errors import check_dialect, parse_sql
from batcher.api.catalog import SessionCatalog
from batcher.api.dataset import Dataset
from batcher.api.sql_session import catalog_sql, statements, views
from batcher.api.sql_session import params as bindings
from batcher.api.sql_session.registry import (
    RegisteredFunction,
    resolve_type,
    validate_null_handling,
    validate_options,
)

__all__ = ["Session"]

#: The session `Session.activate` installed for the current context, or None. A
#: `ContextVar`, so a scope is local to its thread and to its asyncio task.
_ACTIVE: ContextVar[Session | None] = ContextVar("batcher_active_session", default=None)


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

    __slots__ = (
        "_catalog",
        "_dialect",
        "_engines",
        "_functions",
        "_generation",
        "_max_recursion",
        "_models",
        "_plan_cache",
        "_read_only",
        "_tables",
        "_views",
    )

    def __init__(
        self, *, dialect: str = "duckdb", read_only: bool = False, max_recursion: int = 1024
    ) -> None:
        """Create an empty session.

        Args:
            dialect: The sqlglot read dialect. It picks the SQL *grammar* and function
                spellings only: every dialect runs on Batcher's semantics, which follow
                DuckDB's, so ``7 / 2`` is ``3.5`` under ``"postgres"`` too.
            read_only: Refuse statements that create, drop or change a table, view or
                schema. A guard on SQL statements, not a sandbox: see `Session.sql`.
            max_recursion: The most iterations a ``WITH RECURSIVE`` CTE may run before
                the query is refused (DuckDB's default is also 1024).

        Raises:
            PlanError: If `dialect` is unknown or `max_recursion` is not a positive int.
        """
        if isinstance(max_recursion, bool) or not isinstance(max_recursion, int):
            raise PlanError(f"max_recursion must be an int, got {type(max_recursion).__name__}")
        if max_recursion < 1:
            raise PlanError(f"max_recursion must be at least 1, got {max_recursion}")
        self._dialect = check_dialect(dialect)
        self._read_only = bool(read_only)
        self._max_recursion = max_recursion
        self._tables: dict[str, Dataset] = {}
        # `CREATE VIEW` definitions, translated afresh by every query that names one. They
        # share one case-insensitive namespace with `_tables`; see `sql_session.views`.
        self._views: dict[str, views.View] = {}
        self._functions: dict[str, RegisteredFunction] = {}
        self._models: dict[str, Any] = {}
        self._engines: dict[str, Any] = {}
        self._catalog = SessionCatalog()
        # Prepared-statement cache: (dialect, query, bound names, bound values) ->
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
        self._plan_cache: dict[tuple, tuple[int, tuple[object, ...], Dataset]] = {}
        self._generation: list[int] = [0]

    def __repr__(self) -> str:
        """Show the registered table names, e.g. ``Session(tables=['emp', 'dept'])``."""
        return f"Session(tables={[*self._tables, *self._views]!r})"

    def __len__(self) -> int:
        """The number of registered tables and views.

        Returns:
            The count of names in the session catalog.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1]}))
                >>> len(s)
                1
        """
        return len(self._tables) + len(self._views)

    def __contains__(self, name: str) -> bool:
        """Whether a table or view is registered under `name`, compared case-insensitively.

        Args:
            name: The table name to look up.

        Returns:
            True if a table or view is registered under `name`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1]}))
                >>> "t" in s
                True
        """
        return self._key(name) is not None

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

    @property
    def catalog(self) -> SessionCatalog:
        """The catalogs this session resolves table names against (Spark ``spark.catalog``).

        Holds the attached `Catalog`s and the current catalog and namespace. A fresh session
        has one in-memory catalog, ``memory``, with the namespace ``main``.

        Returns:
            This session's `SessionCatalog`.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.catalog.create_table("orders", bt.from_pydict({"id": [1, 2]}))
                >>> s.catalog.list_tables()
                ['main.orders']
        """
        return self._catalog

    def _bump(self) -> None:
        """Invalidate the prepared-statement cache after a catalog mutation."""
        self._generation[0] += 1

    # --- tables ------------------------------------------------------------
    def register(self, name: str, dataset: Dataset | pa.Table, *, replace: bool = True) -> Dataset:
        """Register `dataset` as the session table `name`, replacing any prior by default.

        The DuckDB ``con.register`` / Spark ``createOrReplaceTempView`` analogue, and with
        ``replace=False`` Spark's ``createTempView``. The name is bound to a lazy plan in
        this session only; it shadows a catalog table of the same name and stores nothing.
        A pyarrow table is lifted to a `Dataset`. Names are case-insensitive, as in SQL:
        registering ``MyTab`` replaces an existing ``mytab``.

        Args:
            name: The table name SQL queries will refer to.
            dataset: A `Dataset` or pyarrow table to bind.
            replace: Replace an existing view of the same name; when False, raise instead.

        Raises:
            PlanError: `name` is already registered and `replace` is False.

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
        existing = self._key(name)
        if not replace and existing is not None:
            raise PlanError(
                f"a table or view named {existing!r} is already registered",
                hint="Pass replace=True to replace it, or Session.drop it first.",
            )
        ds = self._as_dataset(dataset)
        self._forget(name)
        self._tables[name] = ds
        self._bump()
        return ds

    def table(self, name: str) -> Dataset:
        """Return the table or view registered as `name`, or else the catalog table it names.

        A session name (`register`, ``CREATE TABLE AS``, ``CREATE VIEW``) shadows a catalog
        table of the same name, and is matched case-insensitively. A view is translated
        now, against the session's current tables. A catalog name resolves as
        ``session.catalog`` describes: ``"t"``, ``"ns.t"`` or ``"catalog.ns.t"``.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
                >>> s.table("t").to_pydict()
                {'x': [1, 2, 3]}

        Args:
            name: The view or catalog table name to look up.

        Returns:
            The view's `Dataset`, or a lazy read of the catalog table.

        Raises:
            PlanError: If neither a view nor a catalog table has that name.
        """
        key = self._key(name)
        if key in self._views:
            return views.expand(self, key)
        if key is not None:
            return self._tables[key]
        if self._catalog.has_table(name):
            return self._catalog.get_table(name).read()
        raise PlanError(
            f"no table {name!r}: not a registered view, nor a catalog table; views: {self.list()}",
            hint="List catalog tables with session.catalog.list_tables().",
        )

    def list(self) -> list[str]:
        """The sorted names of all registered tables and views.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("b", bt.from_pydict({"x": [2]}))
                >>> _ = s.register("a", bt.from_pydict({"x": [1]}))
                >>> s.list()
                ['a', 'b']

        Returns:
            The sorted list of registered table and view names.
        """
        return sorted([*self._tables, *self._views])

    def drop(self, name: str) -> None:
        """Remove the table or view `name` from the session (no error if absent).

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
            name: The table or view name to remove, matched case-insensitively.
        """
        self._forget(name)
        self._bump()

    def clear(self) -> None:
        """Remove every registered table and view (functions and dialect are kept).

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
        self._views.clear()
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
        null_handling: str = "special",
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

        `null_handling` is DuckDB's switch of the same name, for the scalar form. The default
        ``"special"`` passes NULL arguments to `fn` (``None`` per row, nulls inside the
        arrays when vectorized). ``"default"`` is NULL-in, NULL-out: a row with any NULL
        argument gets NULL without `fn` seeing it, so `fn` need not guard against ``None``.

        A registered function is callable from SQL (`sql`, `Dataset.sql`), not from
        `bt.call_function`, which names the built-in function library. In the DataFrame API
        apply the same callable with `Dataset.map_batches`.

        Scalar functions are not supported in ``GROUP BY`` keys, aggregate arguments,
        or ``ORDER BY`` — compute them in a subquery or projected alias first. There is no
        aggregate form at all: an aggregate needs a mergeable partial/combine/finalize
        implementation in the engine, which a Python callable over one batch cannot provide.
        Use ``ds.group_by(...).agg(...)``, or ``map_groups`` for arbitrary Python per group.

        An option the chosen call form cannot honour is rejected rather than ignored, so a
        misspelled keyword or a `map_batches` option on the scalar form fails at registration
        instead of quietly doing nothing.

        Args:
            name: The SQL name the function is called by.
            fn: The Python callable.
            table: Register as a table function rather than a scalar function.
            per_row: Table form only — apply row-by-row instead of per batch.
            vectorized: Scalar form only — pass Arrow arrays (else per-row scalars).
            result_type: Scalar output Arrow type (or alias).
            output_columns: Table-function result column names.
            batch_format: Batch table form only — the `map_batches` batch format.
            null_handling: Scalar form only — ``"special"`` (NULLs reach `fn`) or
                ``"default"`` (a NULL argument gives NULL without calling `fn`).
            **config: Extra `map_batches` (or, with `per_row`, `map`) keyword arguments,
                forwarded by the table form. Anything the call form cannot honour raises.

        Raises:
            PlanError: If an option cannot take effect for the chosen call form.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> import pyarrow.compute as pc
                >>> s = bt.Session()
                >>> _ = s.register("t", bt.from_pydict({"x": [1, 2, 3]}))
                >>> s.register_function("dbl", lambda a: pc.multiply(a, 2), result_type="int64")
                >>> s.sql("SELECT dbl(x) AS y FROM t").to_pydict()
                {'y': [2, 4, 6]}

                >>> _ = s.register("n", bt.from_pydict({"x": [1, None]}))
                >>> s.register_function(
                ...     "inc", lambda v: v + 1, vectorized=False, result_type="int64",
                ...     null_handling="default",
                ... )
                >>> s.sql("SELECT inc(x) AS y FROM n").to_pydict()
                {'y': [2, None]}
        """
        # `batch_format` is a named parameter rather than part of `**config`, so it bypasses
        # the check below — and both forms that cannot honour it dropped it in silence.
        if batch_format != "pyarrow" and (not table or per_row):
            form = "a per-row table function" if per_row else "a scalar function"
            raise PlanError(
                f"register_function({name!r}): batch_format={batch_format!r} has no effect on "
                f"{form}, which receives {'one row dict' if per_row else 'Arrow arrays'} at a "
                f"time. Drop it, or register a batch table function (table=True)."
            )
        validate_options(name, config, table=table, per_row=per_row)
        validate_null_handling(name, null_handling, table=table)
        if table and not per_row:
            # `batch_format` is a `map_batches` option; the per-row form goes through
            # `Dataset.map`, which has no such thing, so injecting it there would fail the very
            # validation above at the point of use rather than at the point of the mistake.
            config = {"batch_format": batch_format, **config}
        self._functions[name] = RegisteredFunction(
            name=name,
            fn=fn,
            table=table,
            per_row=per_row,
            vectorized=vectorized,
            result_type=resolve_type(result_type),
            output_columns=tuple(output_columns) if output_columns is not None else None,
            config=config,
            null_handling=null_handling,
        )
        self._bump()

    def register_model(self, name: str, model: Any) -> None:
        """Register a fitted model that SQL can score with ``ML_PREDICT``.

        The BigQuery ``CREATE MODEL`` analogue for a model that already exists: it binds a
        name in this session's model catalog so a query can name it, the way `register` binds
        a table. Registering never scores anything — the prediction happens when the query
        that names the model runs.

        `model` is a fitted model object (XGBoost, LightGBM, CatBoost, scikit-learn, ONNX) or
        a path to a saved one. A query can also name a saved model by quoted path without
        registering it at all; the catalog exists for the case a path cannot express, which is
        a model fitted in this process and never written to storage.

        Args:
            name: The SQL name the model is scored by.
            model: A fitted model object, or a path/URI to a saved one.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml import LinearRegression
                >>> train = bt.from_pydict({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
                >>> fitted = LinearRegression(features=["x"], target="y").fit(train)
                >>> s = bt.Session()
                >>> s.register_model("doubler", fitted)
                >>> s.list_models()
                ['doubler']
        """
        if not isinstance(name, str) or not name:
            raise PlanError(f"a model name must be a non-empty string, got {name!r}")
        self._models[name] = model
        self._bump()

    def list_models(self) -> list[str]:
        """The sorted names of all registered models.

        Returns:
            The sorted list of registered model names.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.register_model("scorer", "s3://models/churn.onnx")
                >>> s.list_models()
                ['scorer']
        """
        return sorted(self._models)

    def register_engine(self, name: str, engine: Any) -> None:
        """Register an LLM engine that SQL can call with ``AI_GENERATE`` and friends.

        The generative counterpart to `register_model`. An engine is a callable holding an
        endpoint, credentials and sampling settings — `batcher.ml.http_engine`,
        `vllm_engine`, `anthropic_engine` or any zero-argument callable returning a
        ``list[str] -> list[str]`` function — so unlike a model it has no path spelling and
        must be built in Python and bound to a name here. Putting an endpoint and an API key
        in query text is the thing this avoids.

        Registering never calls the model; generation happens when a query naming the engine
        runs.

        Args:
            name: The SQL name the engine is called by.
            engine: An `EngineFactory` — a zero-argument callable returning the engine.

        Raises:
            PlanError: If `name` is not a non-empty string.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> shouty = lambda: (lambda prompts: [p.upper() for p in prompts])
                >>> s.register_engine("shouty", shouty)
                >>> s.list_engines()
                ['shouty']
        """
        if not isinstance(name, str) or not name:
            raise PlanError(f"an engine name must be a non-empty string, got {name!r}")
        self._engines[name] = engine
        self._bump()

    def list_engines(self) -> list[str]:
        """The sorted names of all registered engines.

        Returns:
            The registered engine names, sorted.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.register_engine("a", lambda: (lambda p: p))
                >>> s.list_engines()
                ['a']
        """
        return sorted(self._engines)

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

    def has_function(self, name: str) -> bool:
        """Whether a Python function is registered for SQL under `name`.

        Args:
            name: The SQL function name.

        Returns:
            True if registered.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.register_function("dbl", lambda a: a, result_type="int64")
                >>> s.has_function("dbl")
                True
        """
        return name in self._functions

    def drop_function(self, name: str, *, if_exists: bool = False) -> None:
        """Unregister the SQL function `name`.

        Args:
            name: The SQL function name.
            if_exists: Do nothing when it is not registered, instead of raising.

        Raises:
            PlanError: No function is registered under `name` and `if_exists` is False.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> s.register_function("dbl", lambda a: a, result_type="int64")
                >>> s.drop_function("dbl")
                >>> s.list_functions()
                []
        """
        if name not in self._functions:
            if if_exists:
                return
            raise PlanError(
                f"no function {name!r} is registered; registered: {self.list_functions()}"
            )
        del self._functions[name]
        self._bump()

    # --- execution ---------------------------------------------------------
    def sql(
        self,
        query: str,
        tables: Mapping[str, Any] | None = None,
        *,
        dialect: str | None = None,
        params: Sequence[Any] | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dataset:
        """Run `query` against this session's tables, functions, and dialect.

        Tables passed here bind or override names for this call only (they do not mutate
        the catalog): a ``{name: table}`` mapping as `tables`, keywords, or both. Each is a
        `Dataset`, a pyarrow table, or anything a ``bt.from_*`` constructor accepts. A
        table named ``tables``, ``dialect`` or ``params`` is bound through the mapping.

        `params` fills the query's parameter placeholders with values: ``?`` (or ``$1``,
        ``$2``) from a sequence, ``$name`` (``:name`` in dialects that read it) from a
        mapping. Each value becomes a typed literal in the parsed query, never text spliced
        into it, so a value cannot change what the query means.

        ``CREATE TABLE/VIEW AS`` registers a lazy `Dataset` into this session and ``DROP
        TABLE`` unregisters one; ``INSERT`` / ``DELETE`` / ``UPDATE`` rebind the target table
        to its new state (a plan rewrite — union / filter / projected CASE — that runs only
        on a later terminal op). ``INSERT`` aligns the new rows to the table's column types,
        so over a Python callback stage with no declared output schema it probes that stage
        on an empty batch or a single row (see `Dataset.schema`). ``CREATE VIEW`` stores the
        query, and every later query naming the view translates it again, so a view sees the
        base tables as they are when it is queried. Everything else is a ``SELECT``-family
        query. A session built with ``read_only=True`` refuses every statement that writes;
        registered Python functions still run whatever they run, so it is a statement guard
        and not a sandbox.

        A ``SELECT`` returns a lazy `Dataset` and does no work until a terminal op, with
        these exceptions, each evaluated while the statement is translated: a
        ``WITH RECURSIVE`` CTE (run to its fixpoint), an uncorrelated scalar subquery
        (inlined as a literal), an uncorrelated ``EXISTS`` (a ``LIMIT 1`` probe) or ``NOT IN``
        (two ``LIMIT 1`` probes, for an empty set and a NULL), and the membership set of an
        uncorrelated ``IN (SELECT ...)`` under ``OR`` or read as a value. Statements that
        write a *catalog* table (``CREATE TABLE ns.t AS``, ``INSERT``/``DELETE``/``UPDATE`` on
        one) write it immediately.

        Args:
            query: A SQL statement.
            tables: A ``{name: table}`` mapping, merged with the keyword bindings.
            dialect: Read this call in another sqlglot dialect than the session's.
            params: Values for the query's ``?``/``$1`` (a sequence) or ``$name`` (a
                mapping) placeholders.
            **kwargs: Per-call table bindings by name.

        Returns:
            A lazy `Dataset` of the result (the registered relation for DDL).

        Raises:
            SQLSyntaxError: If `query` does not parse in the dialect.
            SQLUnsupportedError: If it uses a construct Batcher does not translate.
            PlanError: For a bad binding, or a write on a read-only session.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> _ = s.register("nums", bt.from_pydict({"v": [1, 2, 3]}))
                >>> s.sql("SELECT SUM(v) AS total FROM nums").to_pydict()
                {'total': [6]}

                >>> s.sql("SELECT v FROM nums WHERE v > ? ORDER BY v", params=[1]).to_pydict()
                {'v': [2, 3]}
        """
        if not isinstance(query, str):
            raise PlanError(
                f"sql() expects a SQL string as its first argument, got {type(query).__name__}"
            )
        session = self if dialect is None else self._with_dialect(dialect)
        return session._run(query, bindings.bind_tables(tables, kwargs), values=params)

    def execute_script(
        self, script: str, tables: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> list[Dataset]:
        """Run a script of ``;``-separated statements in order, one result per statement.

        The DuckDB ``con.execute`` of a migration script: each statement runs as
        `Session.sql` would run it, so a ``CREATE TABLE`` is visible to the statements
        after it. The script is **not atomic**. Each statement's effect stays in place as
        soon as it has run, so when one fails, the ones before it remain applied; the
        error raised is the failing statement's own, with a note saying how many statements
        completed. Parameter binding is per statement, so it is not offered here: run a
        statement that needs values with `Session.sql`.

        Args:
            script: One or more SQL statements separated by ``;``.
            tables: A ``{name: table}`` mapping visible to every statement.
            **kwargs: Table bindings by name, visible to every statement.

        Returns:
            One lazy `Dataset` per statement, in order.

        Raises:
            SQLSyntaxError: If the script does not parse; nothing has run.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> results = s.execute_script(
                ...     "CREATE TABLE t AS SELECT 1 AS x; INSERT INTO t VALUES (2); "
                ...     "SELECT SUM(x) AS total FROM t"
                ... )
                >>> results[-1].to_pydict()
                {'total': [3]}
        """
        from sqlglot import expressions as exp

        bound = bindings.bind_tables(tables, kwargs)
        ast = self._guarded(lambda: parse_sql(script, dialect=self._dialect))
        statements_ = list(ast.expressions) if isinstance(ast, exp.Block) else [ast]
        results: list[Dataset] = []
        for index, statement in enumerate(statements_):
            try:
                results.append(self._guarded(lambda s=statement: self._dispatch(s, bound)))
            except Exception as exc:
                exc.add_note(
                    f"Statement {index + 1} of {len(statements_)} failed; the {index} before "
                    f"it completed and remain applied (execute_script is not atomic)."
                )
                raise
        return results

    @contextlib.contextmanager
    def activate(self) -> Iterator[Session]:
        """Make this session the one `bt.sql` and `ds.write.table` use, inside a ``with``.

        The scope is a `contextvars.ContextVar`, so it is local to the code running inside
        it: a nested ``activate`` wins until it exits, and two asyncio tasks that each
        activate their own session never see each other's. On exit the previous session is
        back. `bt.set_session` still sets the process default, which applies wherever no
        scope is active.

        Returns:
            A context manager yielding this session.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> s = bt.Session()
                >>> with s.activate():
                ...     _ = bt.sql("CREATE TABLE scratch AS SELECT 1 AS x")
                ...     print(bt.current_session() is s)
                True
                >>> s.list(), "scratch" in bt.current_session()
                (['scratch'], False)
        """
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    @staticmethod
    def _active() -> Session | None:
        """The session the innermost `activate` scope installed here, or None."""
        return _ACTIVE.get()

    # --- internals ---------------------------------------------------------
    def _run(
        self,
        query: str,
        tables: dict[str, Dataset | pa.Table],
        values: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Dataset:
        """Parse, bind and dispatch `query`, with the SQL error types applied."""
        return self._guarded(lambda: self._run_unguarded(query, tables, values))

    @staticmethod
    def _guarded(step: Callable[[], Any]) -> Any:
        """Run `step`, turning a refusal or a stray sqlglot parse error into the SQL types.

        `SQLUnsupportedError` subclasses `NotImplementedError`, so a refusal raised before
        it existed keeps its type for ``except NotImplementedError`` and gains `PlanError`.
        """
        from sqlglot.errors import ParseError, TokenError

        try:
            return step()
        except SQLUnsupportedError:
            raise
        except NotImplementedError as exc:
            raise SQLUnsupportedError(str(exc)) from exc
        except (ParseError, TokenError) as exc:
            raise SQLSyntaxError(f"could not parse SQL: {exc}") from exc

    def _run_unguarded(
        self,
        query: str,
        tables: dict[str, Dataset | pa.Table],
        values: Sequence[Any] | Mapping[str, Any] | None,
    ) -> Dataset:
        # Prepared-statement fast path: re-running the same query text against an
        # unchanged catalog reuses its built plan, skipping the sqlglot parse + AST
        # translation (~2.1 ms). Per-call bindings are cached too — `ds.sql(...)` always
        # passes one, and it is the most-repeated SQL entry point there is — but only
        # when every bound object is the identical object the plan was built over. Bound
        # values are part of the key, each tagged with its type.
        # CREATE/DROP/DML mutate the catalog and are never cached (they bump the
        # generation, which invalidates everything anyway).
        names = tuple(sorted(tables))
        bound = tuple(tables[n] for n in names)
        key = (self._dialect, query, names, bindings.params_key(values))
        hit = self._plan_cache.get(key)
        if (
            hit is not None
            and hit[0] == self._generation[0]
            and len(hit[1]) == len(bound)
            and all(a is b for a, b in zip(hit[1], bound, strict=True))
        ):
            return hit[2]

        ast = bindings.bind_params(parse_sql(query, dialect=self._dialect), values)
        ds, cacheable = self._dispatch_cacheable(ast, tables)
        if cacheable:
            self._remember(key, bound, ds)
        return ds

    def _dispatch(self, ast: Any, tables: dict[str, Dataset | pa.Table]) -> Dataset:
        """Run one parsed statement: a catalog statement, DDL, DML, or a query."""
        return self._dispatch_cacheable(ast, tables)[0]

    def _dispatch_cacheable(
        self, ast: Any, tables: dict[str, Dataset | pa.Table]
    ) -> tuple[Dataset, bool]:
        """Run one parsed statement, and say whether its plan may be served again."""
        from sqlglot import expressions as exp

        if self._read_only:
            _refuse_writes(ast, self._dialect)
        handled = catalog_sql.catalog_statement(self, ast, tables)
        if handled is not None:
            return handled, False
        if isinstance(ast, exp.Create):
            return statements.create(self, ast, tables), False
        if isinstance(ast, exp.Drop):
            return statements.drop(self, ast), False
        if isinstance(ast, (exp.Insert, exp.Delete, exp.Update, exp.Merge)):
            return statements.dml(self, ast, tables), False
        ast, resolved, dynamic = catalog_sql.bind(self, ast, tables)
        ds = self._translate_bound(ast, {**tables, **resolved})
        # A view is re-translated on every reference, so a plan over one is never reused.
        dynamic = dynamic or bool(views.referenced_views(self, ast, tables))
        # A plan over a catalog table, or one that inlined session state such as
        # `current_catalog()`, is rebuilt every call: the table's storage and the session's
        # position both change without the query text changing.
        return ds, not dynamic

    # How many prepared plans to keep. Entries pin their bound datasets alive, so this is
    # a memory bound, not just a lookup bound: a caller that queries a stream of distinct
    # datasets would otherwise retain every one of them.
    _PLAN_CACHE_MAX = 256

    def _remember(self, key: tuple, bound: tuple[object, ...], ds: Dataset) -> None:
        """Store a built plan, evicting oldest-first past `_PLAN_CACHE_MAX`."""
        cache = self._plan_cache
        if len(cache) >= self._PLAN_CACHE_MAX and key not in cache:
            for stale in list(cache)[: len(cache) - self._PLAN_CACHE_MAX + 1]:
                del cache[stale]
        cache[key] = (self._generation[0], bound, ds)

    # --- the seam `statements` reaches through --------------------------------
    # These stay underscore-private: `statements` is a sibling module inside this
    # package, so it reads them the same way the pre-split single file read its own
    # attributes. Publishing them to widen a within-package seam would enlarge the
    # documented public API, which is a commitment we don't make for plumbing.

    def _translate(
        self, ast: Any, tables: dict[str, Dataset | pa.Table], active: tuple[str, ...] = ()
    ) -> Dataset:
        """Lower a parsed ``SELECT``-family AST to a `Dataset` against this catalog.

        Args:
            ast: The parsed statement.
            tables: Per-call bindings, which shadow the catalog for this call.
            active: Views being expanded around this translation (cycle guard).

        Returns:
            The lazy result relation.
        """
        ast, resolved, _ = catalog_sql.bind(self, ast, tables)
        return self._translate_bound(ast, {**tables, **resolved}, active)

    def _translate_bound(
        self, ast: Any, tables: dict[str, Dataset | pa.Table], active: tuple[str, ...] = ()
    ) -> Dataset:
        """Lower an AST whose catalog references `catalog_sql.bind` already resolved.

        Every view the AST names is translated here, now, against the current session.
        """
        from batcher._sql import translate_ast

        registry: dict[str, Dataset | pa.Table] = {**self._tables}
        listing = views.lists_catalog(ast)
        for key in views.referenced_views(self, ast, tables):
            try:
                registry[key] = views.expand(self, key, active)
            except PlanError:
                if not listing:
                    raise  # a query naming a broken view fails, as in DuckDB
        return translate_ast(
            ast,
            functions=self._functions,
            models=self._models,
            engines=self._engines,
            max_recursion=self._max_recursion,
            **{**registry, **tables},
        )

    def _key(self, name: str) -> str | None:
        """The stored spelling of the session table or view `name` names, if any."""
        return views.find(self._tables, name) or views.find(self._views, name)

    def _forget(self, name: str) -> None:
        """Remove every entry `name` names, whatever its case, from tables and views."""
        for store in (self._tables, self._views):
            key = views.find(store, name)
            if key is not None:
                del store[key]

    def _define_view(self, name: str, view: views.View) -> None:
        """Bind `name` to a view definition, replacing any table or view of that name.

        Args:
            name: The view name as written.
            view: The definition.
        """
        key = self._key(name) or name
        self._forget(name)
        self._views[key] = view
        self._bump()

    def _rebind(self, name: str, dataset: Dataset) -> None:
        """Point `name` at `dataset` and invalidate the prepared-statement cache.

        Args:
            name: The catalog name to bind.
            dataset: The relation to bind it to.
        """
        key = self._key(name) or name
        self._forget(name)
        self._tables[key] = dataset
        self._bump()

    def _unbind(self, name: str) -> None:
        """Drop `name` from the catalog if present, invalidating prepared statements.

        Args:
            name: The catalog name to remove.
        """
        self._forget(name)
        self._bump()

    def _with_dialect(self, dialect: str) -> Session:
        """A view of this session reading `dialect`, sharing its tables, functions and models.

        Everything mutable is shared *by reference* with the owning session — the
        catalog, the function and model registries, the plan cache, and the generation
        counter — so a table registered on either is visible to both. Only the read dialect
        differs, and the plan cache is keyed by dialect, so the same query text
        parsed as Spark and as DuckDB cannot collide. The read-only guard and the recursion
        cap carry over.
        """
        check_dialect(dialect)
        view = Session.__new__(Session)
        view._tables = self._tables
        view._views = self._views
        view._functions = self._functions
        view._models = self._models
        view._engines = self._engines
        view._catalog = self._catalog
        view._dialect = dialect
        view._plan_cache = self._plan_cache
        view._generation = self._generation
        view._read_only = self._read_only
        view._max_recursion = self._max_recursion
        return view

    @staticmethod
    def _as_dataset(dataset: Dataset | pa.Table) -> Dataset:
        if isinstance(dataset, Dataset):
            return dataset
        if isinstance(dataset, pa.Table):
            from batcher.api.session import from_arrow

            return from_arrow(dataset)
        raise PlanError(f"table must be a Dataset or pyarrow.Table, got {type(dataset).__name__}")


def _refuse_writes(ast: Any, dialect: str) -> None:
    """Raise `PlanError` when `ast` creates, drops or changes a table, view or schema.

    ``EXPLAIN`` reaches here as an opaque command, so the statement it explains is parsed
    and checked in turn. Any other command the session cannot classify (an `exp.Command`
    other than ``SHOW``) is refused: a read-only guard that waves through what it does not
    recognize guards nothing.
    """
    from sqlglot import expressions as exp

    if isinstance(ast, exp.Command) and str(ast.this).upper() == "EXPLAIN":
        inner = str(getattr(ast.expression, "this", ast.expression) or "").strip()
        if inner.upper().startswith("ANALYZE"):
            inner = inner[len("ANALYZE") :]
        _refuse_writes(parse_sql(inner, dialect=dialect), dialect)
        return
    writes = (
        exp.Create,
        exp.Drop,
        exp.Insert,
        exp.Delete,
        exp.Update,
        exp.Merge,
        exp.Alter,
        exp.TruncateTable,
        exp.Copy,
    )
    found = ast.find(*writes)
    unknown = isinstance(ast, exp.Command) and str(ast.this).upper() != "SHOW"
    if found is None and not unknown:
        return
    what = (found or ast).key.upper()
    raise PlanError(
        f"this session is read-only and refuses the {what} statement",
        hint="Run it on a session built without read_only=True.",
    )
