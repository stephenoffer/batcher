"""A query-based read is governed by the table it declares, and refused when it declares none.

A warehouse read defined by SQL (Snowflake, a BigQuery or ``bt.read.sql`` ``query=``) names
no table, so a policy keyed on a table name cannot match it. Inside a `security()` block that
used to mean ``SELECT * FROM users`` returned every column of a table whose grant withheld
half of them, silently: the catalog was never consulted because there was no name to ask it
about.

Two things close it. ``governed_as=`` declares the table the query reads, and the policy on
that table is then applied to the result. And an *undeclared* query that references a
governed table is refused rather than passed through.

The reads here go through the DB-API path over an in-memory DuckDB connection, which takes
the same `_read_table` -> `_scan` -> `govern_scan` route as every warehouse connector and
needs no service.
"""

from __future__ import annotations

import dataclasses

import duckdb
import pytest

import batcher as bt
from batcher._internal.errors import AccessDeniedError
from batcher.api.security._binding import _referenced_governed_tables
from batcher.config import active_config, set_config

pytestmark = pytest.mark.unit

ANALYST = bt.Principal("ana", roles=["analyst"])


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("CREATE TABLE users (id INTEGER, ssn TEXT)")
    c.execute("CREATE TABLE events (id INTEGER, kind TEXT)")
    c.execute("INSERT INTO users VALUES (1, '111'), (2, '222')")
    c.execute("INSERT INTO events VALUES (1, 'a')")
    return c


@pytest.fixture
def strict():
    original = active_config()
    set_config(original.replace(governance=dataclasses.replace(original.governance, mode="strict")))
    yield
    set_config(original)


def _catalog() -> bt.SecurityCatalog:
    return bt.SecurityCatalog().grant("analyst", on="MAIN.USERS", select=["id"])


class TestAnUndeclaredQuery:
    def test_outside_security_the_read_is_unchanged(self, con):
        out = bt.read.sql("SELECT * FROM users", connection=con).to_pydict()
        assert out == {"id": [1, 2], "ssn": ["111", "222"]}

    def test_a_query_touching_a_governed_table_is_refused(self, con):
        with (
            bt.security(_catalog(), ANALYST),
            pytest.raises(AccessDeniedError, match="governed_as"),
        ):
            bt.read.sql("SELECT * FROM users", connection=con)

    @pytest.mark.parametrize(
        "sql",
        [
            "select id, ssn from USERS",
            'SELECT * FROM "users"',
            "SELECT * FROM main.users",
            "SELECT u.ssn FROM events e JOIN users u ON e.id = u.id",
        ],
    )
    def test_every_spelling_of_the_reference_is_refused(self, con, sql):
        with bt.security(_catalog(), ANALYST), pytest.raises(AccessDeniedError):
            bt.read.sql(sql, connection=con)

    def test_a_query_touching_only_ungoverned_tables_proceeds(self, con):
        with bt.security(_catalog(), ANALYST):
            out = bt.read.sql("SELECT * FROM events", connection=con).to_pydict()
        assert out == {"id": [1], "kind": ["a"]}

    def test_a_longer_identifier_is_not_the_governed_table(self):
        tables = frozenset({"MAIN.USERS"})
        assert _referenced_governed_tables("select * from users_archive", tables) == []
        assert _referenced_governed_tables("select * from users", tables) == ["MAIN.USERS"]


class TestADeclaredQuery:
    def test_the_declared_tables_policy_is_applied(self, con):
        with bt.security(_catalog(), ANALYST):
            ds = bt.read.sql("SELECT * FROM users", connection=con, governed_as="MAIN.USERS")
            out = ds.to_pydict()
        assert out == {"id": [1, 2]}

    def test_a_declaration_differing_only_in_case_is_refused(self, con):
        with bt.security(_catalog(), ANALYST), pytest.raises(AccessDeniedError, match="case"):
            bt.read.sql("SELECT * FROM users", connection=con, governed_as="main.users")

    def test_an_empty_declaration_is_refused(self, con):
        with pytest.raises(AccessDeniedError, match="non-empty"):
            bt.read.sql("SELECT * FROM users", connection=con, governed_as="  ")

    def test_a_named_source_cannot_be_redeclared(self, tmp_path):
        path = str(tmp_path / "t.parquet")
        bt.from_pydict({"x": [1]}).write(path, format="parquet")
        with pytest.raises(AccessDeniedError, match="already names its table"):
            bt.read.table("parquet", path, governed_as="/somewhere/else")

    def test_declaring_a_named_sources_own_name_is_accepted(self, tmp_path):
        path = str(tmp_path / "t.parquet")
        bt.from_pydict({"x": [1]}).write(path, format="parquet")
        assert bt.read.table("parquet", path, governed_as=path).count() == 1

    def test_strict_mode_accepts_a_declared_query(self, con, strict):
        with bt.security(_catalog(), ANALYST):
            # The positive control: strict mode refuses the same shape undeclared.
            with pytest.raises(AccessDeniedError, match="no durable name"):
                bt.read.sql("SELECT * FROM events", connection=con)
            out = bt.read.sql(
                "SELECT * FROM users", connection=con, governed_as="MAIN.USERS"
            ).to_pydict()
        assert out == {"id": [1, 2]}
