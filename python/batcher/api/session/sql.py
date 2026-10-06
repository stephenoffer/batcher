"""The default session: `bt.sql`, `bt.register_function`, `bt.register_model` and its accessors.

A process-global `Session` backs all three, so ``CREATE TABLE AS`` in one call is
visible to the next, and `ds.write.table` resolves table names against it when no session
is passed. `bt.current_session` returns it and `bt.set_session` replaces it; `bt.Session`
builds an isolated one, and ``with session.activate():`` makes one current for the code
inside the block only.

`bt.sql_expr` and `bt.call_function` sit beside them: they reach the same SQL function
table for a single expression, read in the default session's dialect unless told otherwise.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api.dataset import Dataset
from batcher.api.sql_session import Session

if TYPE_CHECKING:
    from batcher.plan.expr_ir import Expr

__all__ = [
    "call_function",
    "current_session",
    "register_function",
    "register_model",
    "set_session",
    "sql",
    "sql_expr",
]

# The process-global default session, backing the module-level `sql` / `register_function`
# below wherever no `Session.activate()` scope is active. A one-slot list so `set_session`
# can replace it without a `global` statement.
_default: list[Session] = [Session()]


def current_session() -> Session:
    """The `Session` that `bt.sql` and `ds.write.table` use here and now.

    Spark's ``SparkSession.active()`` and Daft's ``current_session()``. Inside a
    ``with session.activate():`` block it is that session; everywhere else it is the
    process default, so tables created with ``bt.sql("CREATE TABLE ...")`` and catalogs
    attached to it are visible to every later `bt.sql` call in the process.

    Returns:
        The active session, or the process default.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> bt.current_session().catalog.current_catalog()
            'memory'
    """
    active = Session._active()
    return _default[0] if active is None else active


def set_session(session: Session) -> None:
    """Make `session` the process default that `bt.sql` and `ds.write.table` use.

    A ``with other.activate():`` block still uses `other` inside it; the default applies
    wherever no such block is active.

    Args:
        session: The session to install.

    Raises:
        PlanError: `session` is not a `bt.Session`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> previous = bt.current_session()
            >>> s = bt.Session()
            >>> bt.set_session(s)
            >>> bt.current_session() is s
            True
            >>> bt.set_session(previous)
    """
    if not isinstance(session, Session):
        raise PlanError(f"set_session() takes a bt.Session, got {type(session).__name__}")
    _default[0] = session


def sql(
    query: str,
    tables: Mapping[str, Any] | None = None,
    *,
    dialect: str | None = None,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Dataset:
    """Run a SQL query over named tables, returning a lazy `Dataset`.

    Each keyword binds a table name used in the query to a `Dataset`, a pyarrow
    table, or any object a ``bt.from_*`` constructor accepts — a pandas or Polars
    frame, a dict of columns, a list of row dicts, a DuckDB relation. Pass a
    ``{name: table}`` mapping positionally when the names are not valid Python
    identifiers, are computed, or are ``dialect`` or ``params``. The query is parsed and
    optimized through the same engine as the DataFrame API, so the two interoperate freely:
    the result is itself a lazy `Dataset` you can keep building on (``.filter``,
    ``.with_columns``, another ``sql``) before a terminal operation runs the plan.

    `params` binds values to the query's placeholders: ``?`` or ``$1`` take a sequence,
    ``$name`` takes a mapping. A value is substituted into the parsed query as a typed
    literal, never spliced into its text, so it cannot change what the query means.

    Names not passed here resolve from the current session's catalog, which ``CREATE
    TABLE/VIEW AS`` populates and ``DROP TABLE`` clears, so a later ``bt.sql("...
    FROM t")`` can omit the binding. Functions registered with `bt.register_function`
    are callable from the query. For an isolated catalog use `bt.Session`. This is
    `Session.sql` on `bt.current_session`.

    Args:
        query: A SQL statement. Table names refer to the bound names.
        tables: A ``{name: table}`` mapping, merged with the keyword bindings.
        dialect: Override the sqlglot read dialect for this call (default ``duckdb``). It
            changes the grammar only; the semantics are Batcher's in every dialect.
        params: Values for the query's ``?``/``$1`` (a sequence) or ``$name`` (a mapping)
            placeholders.
        **kwargs: Named inputs, each a `Dataset`, pyarrow table, or convertible object.

    Returns:
        A lazy `Dataset` of the query result.

    Raises:
        SQLSyntaxError: If `query` does not parse.
        SQLUnsupportedError: If `query` uses a construct Batcher does not translate.
        PlanError: If `query` is not a string, `tables` is not a mapping, or `params`
            does not match the placeholders.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> sales = bt.from_pydict({"region": ["w", "e", "w"], "amount": [10, 20, 30]})
            >>> out = bt.sql(
            ...     "SELECT region, SUM(amount) AS total "
            ...     "FROM sales GROUP BY region ORDER BY region",
            ...     sales=sales,
            ... )
            >>> out.to_pydict()
            {'region': ['e', 'w'], 'total': [20, 40]}

            >>> bt.sql("SELECT * FROM t", {"t": {"x": [1, 2]}}).to_pydict()
            {'x': [1, 2]}

            >>> bt.sql(
            ...     "SELECT region FROM sales WHERE amount > $min ORDER BY region",
            ...     sales=sales,
            ...     params={"min": 15},
            ... ).to_pydict()
            {'region': ['e', 'w']}
    """
    return current_session().sql(query, tables, dialect=dialect, params=params, **kwargs)


def sql_expr(text: str, *, dialect: str | None = None) -> Expr:
    """Parse one SQL expression into an `Expr` (Polars and Daft ``sql_expr``, Spark ``expr``).

    The text is translated by the same function table `bt.sql` uses, so a function spelled
    in SQL has the meaning it has in a query. A trailing ``AS name`` becomes an alias, which
    makes ``ds.select(bt.sql_expr("a + 1 AS b"))`` the spelling of Spark's ``selectExpr``.
    Column references stay names, resolved when the expression is used. An aggregate call
    becomes an aggregate expression for `agg`.

    Window functions (``OVER``) and functions registered with `bt.register_function` need a
    relation and are refused; use `bt.sql` for those.

    Args:
        text: One SQL expression, optionally ending in ``AS name``.
        dialect: The sqlglot read dialect, such as ``"spark"``; the default catalog's
            dialect (``duckdb``) when omitted.

    Returns:
        The expression, aliased when the text carries ``AS name``.

    Raises:
        PlanError: If `text` is not valid SQL, is a query or statement rather than an
            expression, or uses a construct an expression cannot carry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"x": [1, 2], "s": ["a", "b"]})
            >>> ds.select(bt.sql_expr("x + 1 AS y"), bt.sql_expr("upper(s)").alias("u")).to_pydict()
            {'y': [2, 3], 'u': ['A', 'B']}

            >>> ds.agg(bt.sql_expr("sum(x) AS total")).to_pydict()
            {'total': [3]}
    """
    from batcher._sql.expression import parse_sql_expression

    return parse_sql_expression(text, dialect=dialect or current_session()._dialect)


def call_function(name: str, *args: Any, dialect: str | None = None) -> Expr:
    """Call a SQL function by name on expression arguments (Spark ``call_function``).

    The name is looked up in the SQL function table, the one `bt.sql` and `bt.sql_expr`
    read, so any function a query can call is reachable without its own Python constructor.
    A string argument is a column name, as in Spark. A Python number, or a ``bt.lit``
    constant, is passed as a SQL literal, which the functions that need a constant argument
    require.

    Args:
        name: The SQL function name, such as ``"pmod"`` or ``"find_in_set"``.
        *args: The arguments: expressions, column names, or constants.
        dialect: The sqlglot read dialect whose function names apply; the default
            catalog's dialect (``duckdb``) when omitted.

    Returns:
        The expression the call translates to.

    Raises:
        PlanError: If `name` is not a function name or the call does not translate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"a": [-10, 7], "csv": ["a,b", "b,c"]})
            >>> ds.select(
            ...     m=bt.call_function("pmod", "a", 3, dialect="spark"),
            ...     p=bt.call_function("find_in_set", bt.lit("b"), "csv", dialect="spark"),
            ... ).to_pydict()
            {'m': [2, 1], 'p': [2, 1]}
    """
    from batcher._sql.expression import call_sql_function

    return call_sql_function(name, args, dialect=dialect or current_session()._dialect)


def register_function(name: str, fn: Callable, **options: Any) -> None:
    """Register a Python function callable from `bt.sql` (the default session).

    Registers on the default catalog; see `Session.register_function` for the call
    forms (scalar ``SELECT f(x)`` vs table ``SELECT * FROM f(t)``) and options. For an
    isolated registry use `bt.Session`.

    Args:
        name: The SQL name the function is called by.
        fn: The Python callable.
        **options: Forwarded to `Session.register_function`.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> import pyarrow.compute as pc
            >>> bt.register_function("dbl", lambda a: pc.multiply(a, 2), result_type="int64")
            >>> t = bt.from_pydict({"x": [1, 2, 3]})
            >>> bt.sql("SELECT dbl(x) AS y FROM t", t=t).to_pydict()
            {'y': [2, 4, 6]}
    """
    current_session().register_function(name, fn, **options)


def register_model(name: str, model: Any) -> None:
    """Register a fitted model that `bt.sql` can score with ``ML_PREDICT`` (default session).

    Registers on the default catalog; see `Session.register_model`. A query can also name a
    saved model by quoted path without registering anything. For an isolated registry use
    `bt.Session`.

    Args:
        name: The SQL name the model is scored by.
        model: A fitted model object, or a path/URI to a saved one.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml import LinearRegression
            >>> train = bt.from_pydict({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
            >>> fitted = LinearRegression(features=["x"], target="y").fit(train)
            >>> bt.register_model("doubler", fitted)
            >>> scored = bt.sql(
            ...     "SELECT x, prediction FROM ML_PREDICT(t, doubler) ORDER BY x",
            ...     t=bt.from_pydict({"x": [5.0]}),
            ... )
            >>> [round(v, 6) for v in scored.to_pydict()["prediction"]]
            [10.0]
    """
    current_session().register_model(name, model)
