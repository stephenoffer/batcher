"""Vendor adjustments applied where a PEP 249 connection is opened — on the worker.

`dbapi._dsn` turns a URI into ``connect()`` keyword arguments that are pickled onto every
split, so they must stay plain values and secret *references*. Two drivers need something a
plain keyword cannot carry, and both are built here, after the secret has been resolved on
the machine that dials the connection:

* **Trino** takes a password only inside an authentication object,
  ``trino.auth.BasicAuthentication(user, password)``, and refuses one over plain http.
  `adapt_connect_kwargs` builds the object from the resolved password and defaults
  ``http_scheme`` to https, so a ``trino://user:pw@host/...`` URI authenticates the way the
  client documents rather than failing on an unexpected ``password`` keyword.
* **python-oracledb** (2.0 or later) returns a NUMBER with a fractional scale as a float. With
  ``oracle_numbers="decimal"`` `prepare_connection` installs the output type handler the
  driver documents for fetching numbers as `decimal.Decimal`, so a ``NUMBER(38, 10)`` keeps
  every digit.
"""

from __future__ import annotations

import decimal
from typing import Any

from batcher._internal.errors import BackendError

__all__ = ["adapt_connect_kwargs", "prepare_connection"]

#: Drivers whose output type handler takes ``(cursor, metadata)`` — python-oracledb 2 and later.
#: cx_Oracle's handler has a six-argument signature, so it is refused rather than half-served.
_ORACLE_DRIVERS = frozenset({"oracledb"})


def _root(module_name: str) -> str:
    return module_name.split(".", maxsplit=1)[0]


def adapt_connect_kwargs(module_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """The ``connect()`` kwargs `module_name` actually accepts, built from resolved ones.

    Args:
        module_name: The driver module being connected through.
        kwargs: The keyword arguments, secrets already resolved.

    Returns:
        The kwargs to pass to ``connect()``; unchanged for every driver but Trino.

    Examples:
        .. doctest::

            >>> from batcher.io.formats.sql.vendors.connect import adapt_connect_kwargs
            >>> adapt_connect_kwargs("sqlite3", {"database": ":memory:"})
            {'database': ':memory:'}
    """
    if _root(module_name) != "trino" or "password" not in kwargs:
        return kwargs
    adapted = dict(kwargs)
    password = adapted.pop("password")
    user = adapted.get("user")
    if not user:
        raise BackendError(
            "a Trino password needs a user to authenticate as: put it in the URI "
            "(trino://user@host:443/catalog) or pass connect_kwargs={'user': ...}."
        )
    from batcher.io.formats.sql._common import require_module

    auth = require_module("trino.auth", extra="trino")
    adapted["auth"] = auth.BasicAuthentication(user, password)
    adapted.setdefault("http_scheme", "https")
    return adapted


def prepare_connection(conn: Any, module_name: str, *, oracle_numbers: str = "native") -> None:
    """Configure a freshly opened connection for the declared vendor type rules.

    Args:
        conn: The open connection.
        module_name: The driver module it came from.
        oracle_numbers: ``"decimal"`` fetches every Oracle NUMBER as `decimal.Decimal`.

    Raises:
        BackendError: If ``oracle_numbers="decimal"`` is asked of a non-Oracle driver.

    Examples:
        .. doctest::

            >>> import sqlite3
            >>> from batcher.io.formats.sql.vendors.connect import prepare_connection
            >>> prepare_connection(sqlite3.connect(":memory:"), "sqlite3")
    """
    if oracle_numbers == "native":
        return
    if _root(module_name) not in _ORACLE_DRIVERS:
        raise BackendError(
            f"oracle_numbers={oracle_numbers!r} applies to python-oracledb only; "
            f"{module_name!r} is not it. Use module='oracledb', or CAST the column in the query."
        )
    import importlib

    number_type = importlib.import_module("oracledb").DB_TYPE_NUMBER

    def _numbers_as_decimal(cursor: Any, metadata: Any) -> Any:
        if metadata.type_code is number_type:
            return cursor.var(decimal.Decimal, arraysize=cursor.arraysize)
        return None

    conn.outputtypehandler = _numbers_as_decimal
