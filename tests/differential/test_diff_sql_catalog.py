"""`SHOW TABLES` and `DESCRIBE` against DuckDB — the two statements a client opens with.

Every SQL client issues one or both of these before it issues a query. A BI tool populates
its table picker from `SHOW TABLES`; a SQLAlchemy reflection and every schema browser reads
`DESCRIBE`. Without them a session that can answer any query at all still looks empty, and
the failure is at the *connection*, before the user has typed anything.

Neither needed a catalog: `Session` already holds `{name: Dataset}` and a `Dataset` already
knows its schema, so both statements read what was there and shape it as a relation. That is
the same thing `EXPLAIN` does one branch above them in the translator.

**The one deliberate divergence is pinned here rather than left to be discovered.**
`DESCRIBE`'s `column_type` carries Batcher's type names, not DuckDB's: an integer column
says `int64` where DuckDB says `INTEGER`. The engine stores Arrow and `Dataset.schema`
reports Arrow, so rendering DuckDB's spelling would tell a client something untrue about the
storage. The shape is borrowed; the content is this engine's. Everything else about the two
statements is asserted *equal* to DuckDB.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential


@pytest.fixture
def two_tables(duck):
    """The same two tables registered on a `Session` and created in DuckDB."""
    orders = pa.table({"order_id": [1, 2], "total": [10.5, 20.0]})
    people = pa.table({"name": ["a"], "age": [30]})
    session = bt.Session()
    session.register("orders", bt.from_arrow(orders))
    session.register("people", bt.from_arrow(people))
    duck.register("orders", orders)
    duck.register("people", people)
    return session, duck


class TestShowTables:
    def test_it_lists_every_registered_table(self, two_tables):
        session, duck = two_tables
        got = sorted(session.sql("SHOW TABLES").to_pydict()["name"])
        expected = sorted(r[0] for r in duck.sql("SHOW TABLES").fetchall())
        assert got == expected == ["orders", "people"]

    def test_the_column_is_named_as_duckdb_names_it(self, two_tables):
        """A client parses this by column name, so the name is the contract."""
        session, duck = two_tables
        assert session.sql("SHOW TABLES").columns == list(duck.sql("SHOW TABLES").columns)

    def test_an_empty_session_lists_nothing_rather_than_failing(self):
        """The first thing a client does on a fresh connection."""
        assert bt.sql("SHOW TABLES").to_pydict() == {"name": []}

    def test_the_result_is_a_relation_and_can_be_queried(self, two_tables):
        """It is a `Dataset`, so the ordinary verbs work on it -- which is what makes it
        useful to a client that filters the picker."""
        session, _ = two_tables
        listed = session.sql("SHOW TABLES").filter(bt.col("name") == "orders")
        assert listed.to_pydict()["name"] == ["orders"]


class TestDescribe:
    def test_the_columns_are_duckdbs(self, two_tables):
        session, duck = two_tables
        assert session.sql("DESCRIBE orders").columns == list(duck.sql("DESCRIBE orders").columns)

    def test_it_names_every_column_in_schema_order(self, two_tables):
        session, duck = two_tables
        got = session.sql("DESCRIBE orders").to_pydict()["column_name"]
        expected = [r[0] for r in duck.sql("DESCRIBE orders").fetchall()]
        assert got == expected == ["order_id", "total"]

    def test_nullability_matches(self, two_tables):
        session, duck = two_tables
        got = session.sql("DESCRIBE orders").to_pydict()["null"]
        expected = [r[2] for r in duck.sql("DESCRIBE orders").fetchall()]
        assert got == expected

    def test_the_three_columns_batcher_has_nothing_to_put_in_are_null(self, two_tables):
        """Batcher has no primary keys, column defaults or storage attributes. Inventing
        values for them would be a plausible answer, which is worse than an empty one."""
        session, _ = two_tables
        rows = session.sql("DESCRIBE orders").to_pydict()
        assert rows["key"] == rows["default"] == rows["extra"] == [None, None]

    def test_the_type_spelling_is_batchers_and_that_is_deliberate(self, two_tables):
        """The pinned divergence. Asserted in both directions so a later change that
        started rendering DuckDB's spelling fails here and has to argue for itself."""
        session, duck = two_tables
        got = session.sql("DESCRIBE orders").to_pydict()["column_type"]
        duck_types = [r[1] for r in duck.sql("DESCRIBE orders").fetchall()]
        assert got == ["int64", "double"]
        assert duck_types == ["BIGINT", "DOUBLE"]
        assert got != duck_types

    def test_describing_an_unregistered_table_names_the_ones_that_exist(self, two_tables):
        session, _ = two_tables
        with pytest.raises(Exception, match="unknown table"):
            session.sql("DESCRIBE nope")

    def test_a_dataset_can_describe_itself(self):
        """`ds.sql(...)` registers the dataset as `self`, so this is the no-session form."""
        rows = bt.from_pydict({"a": [1], "b": ["x"]}).sql("DESCRIBE self").to_pydict()
        assert rows["column_name"] == ["a", "b"]
        assert rows["column_type"] == ["int64", "string"]
