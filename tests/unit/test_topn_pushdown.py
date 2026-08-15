"""A top-N reaches the server as `ORDER BY ... LIMIT n`, or does not reach it at all.

A plain `LIMIT` is a positional prefix, so a sort between it and the scan blocks it: the
first n rows of a sorted relation are not the first n of its input. Pushing the *ordering*
alongside the cap is what makes that case sound again, and it is the shape behind every
"latest 20" and "top 10 by revenue" query.

The whole risk lives in one clause. Servers disagree about where a null sorts by default —
measured on sqlite 3.52, ``ORDER BY k LIMIT 2`` over ``[3, NULL, 1, NULL, 2]`` returns
``[NULL, NULL]`` there and ``[1, 2]`` in DuckDB — so an ordering pushed without an explicit
``NULLS FIRST``/``NULLS LAST`` asks the server for a different top-N than the engine would
compute. That is a *wrong rows* failure, not a slow one, and it is why a dialect with no
such clause (MySQL, SQL Server) must decline the cap as well as the ordering.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.io.formats.sql._common import push_down
from batcher.io.formats.sql.uri import quote_identifier, supports_nulls_ordering
from batcher.kyber.rules.source_limits import (
    required_limits_per_source,
    required_orderings_per_source,
)

pytestmark = pytest.mark.unit

_DATA = {"k": [3, 1, 2], "v": [1, 2, 3]}


def _ds():
    return bt.from_pydict(_DATA)


def test_a_top_n_pushes_both_the_cap_and_its_ordering():
    plan = _ds().sort("k").limit(2)._plan
    assert required_limits_per_source(plan) == {0: 2}
    assert required_orderings_per_source(plan) == {0: (("k", False, False),)}


@pytest.mark.parametrize(
    ("descending", "nulls_first"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_the_direction_and_null_placement_travel_with_the_key(descending, nulls_first):
    plan = _ds().sort("k", descending=descending, nulls_first=nulls_first).limit(2)._plan
    assert required_orderings_per_source(plan) == {0: (("k", descending, nulls_first),)}


def test_a_sort_without_a_cap_pushes_nothing():
    # Ordering a source without capping it makes the server sort rows it would otherwise
    # have streamed, for no saving at all.
    plan = _ds().sort("k")._plan
    assert required_limits_per_source(plan) == {}
    assert required_orderings_per_source(plan) == {}


def test_a_plain_limit_still_pushes_without_an_ordering():
    plan = _ds().limit(2)._plan
    assert required_limits_per_source(plan) == {0: 2}
    assert required_orderings_per_source(plan) == {}


def test_a_computed_sort_key_declines():
    # The server can only order by something it can name; re-deriving an expression in the
    # pushed SQL risks a different top-N rather than extra rows.
    plan = _ds().sort(bt.col("k") * 2).limit(2)._plan
    assert required_limits_per_source(plan) == {}


def test_a_filter_between_the_sort_and_the_scan_declines():
    # The server's top-2 of the unfiltered relation is not the top-2 of the filtered one.
    plan = _ds().filter(bt.col("k") > 0).sort("k").limit(2)._plan
    assert required_limits_per_source(plan) == {}


def test_a_projection_between_the_sort_and_the_scan_passes_through():
    plan = _ds().select("k", "v").sort("k").limit(2)._plan
    assert required_limits_per_source(plan) == {0: 2}


@pytest.mark.parametrize("scheme", ["postgresql", "duckdb", "sqlite", "snowflake", "clickhouse"])
def test_dialects_that_can_express_null_placement(scheme):
    assert supports_nulls_ordering(scheme) is True


@pytest.mark.parametrize("scheme", ["mysql", "mariadb", "tidb", "mssql", "sqlserver", "unknown"])
def test_dialects_that_cannot_must_decline(scheme):
    assert supports_nulls_ordering(scheme) is False


def test_the_generated_sql_states_the_null_placement_explicitly():
    quote = lambda name: quote_identifier(name, "postgresql")  # noqa: E731
    sql = push_down(
        None, None, None, table="t", limit=2, order_by=(("k", True, False),), quote=quote
    )
    assert 'ORDER BY "k" DESC NULLS LAST LIMIT 2' in sql


def test_the_ordering_precedes_the_cap():
    quote = lambda name: quote_identifier(name, "postgresql")  # noqa: E731
    sql = push_down(
        None, None, None, table="t", limit=2, order_by=(("k", False, False),), quote=quote
    )
    assert sql.index("ORDER BY") < sql.index("LIMIT")


def test_a_backend_that_cannot_order_drops_the_cap_as_well(tmp_path):
    """The soundness rule: an ordered cap is all-or-nothing.

    A backend that takes `LIMIT` but cannot spell `NULLS LAST` would otherwise return its
    own first n — silently the wrong rows.
    """
    from batcher.io.formats.sql.connectorx import ConnectorXSource

    order = (("k", False, False),)
    mysql = ConnectorXSource(query="SELECT * FROM t", conn_uri="mysql://h/db")
    assert mysql.supports_limit is True, "MySQL does take a LIMIT"
    assert mysql.supports_ordering is False, "but cannot state null placement"
    assert "LIMIT" not in mysql._split(None, None, 5, order).query

    postgres = ConnectorXSource(query="SELECT * FROM t", conn_uri="postgresql://h/db")
    pushed = postgres._split(None, None, 5, order).query
    assert "ORDER BY" in pushed and pushed.endswith("LIMIT 5")


def test_an_unordered_cap_still_pushes_to_a_backend_that_cannot_order():
    from batcher.io.formats.sql.connectorx import ConnectorXSource

    mysql = ConnectorXSource(query="SELECT * FROM t", conn_uri="mysql://h/db")
    assert mysql._split(None, None, 5, None).query.endswith("LIMIT 5")
