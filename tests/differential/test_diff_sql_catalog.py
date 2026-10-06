"""`SHOW TABLES`, `DESCRIBE` and `information_schema` against DuckDB.

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
from _harness import assert_same
from batcher._internal.errors import PlanError

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


class TestInformationSchema:
    """The ANSI spelling of the same two questions, which is what a reflection reads.

    It is answered from the same registry `SHOW TABLES` reads, so the three cannot disagree
    about what exists -- which is the property worth having, and the reason this is not a
    second catalog.
    """

    def test_tables_lists_what_show_tables_lists(self, two_tables):
        """The two spellings must never disagree; asserting one against the other is
        cheaper and stricter than asserting each against a literal."""
        session, _ = two_tables
        via_show = sorted(session.sql("SHOW TABLES").to_pydict()["name"])
        via_schema = sorted(
            session.sql("SELECT table_name FROM information_schema.tables").to_pydict()[
                "table_name"
            ]
        )
        assert via_show == via_schema == ["orders", "people"]

    def test_tables_matches_duckdb_on_the_table_names(self, two_tables):
        session, duck = two_tables
        got = session.sql(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).to_pydict()["table_name"]
        expected = [
            r[0]
            for r in duck.sql(
                "SELECT table_name FROM information_schema.tables ORDER BY table_name"
            ).fetchall()
        ]
        assert got == expected

    def test_a_registered_dataset_is_a_base_table(self, two_tables):
        """`Session.register` binds a relation, which is a table to a reflection.

        Not compared against DuckDB here on purpose. DuckDB answers `VIEW` for an Arrow
        object registered with `register()` and `BASE TABLE` for one created with `CREATE
        TABLE`, so the comparison would be measuring how the *fixture* loaded the data
        rather than anything about either engine. `TestViewsAndSchemata` compares the
        statement-created forms, which both engines create the same way.
        """
        session, _ = two_tables
        types = session.sql("SELECT table_type FROM information_schema.tables").to_pydict()[
            "table_type"
        ]
        assert set(types) == {"BASE TABLE"}

    def test_columns_matches_duckdb_on_everything_but_the_type_name(self, two_tables):
        session, duck = two_tables
        q = (
            "SELECT table_name, column_name, ordinal_position, is_nullable "
            "FROM information_schema.columns ORDER BY table_name, ordinal_position"
        )
        got = session.sql(q).to_pydict()
        expected = duck.sql(q).fetchall()
        assert (
            list(
                zip(
                    got["table_name"],
                    got["column_name"],
                    got["ordinal_position"],
                    got["is_nullable"],
                    strict=True,
                )
            )
            == expected
        )

    def test_the_data_type_column_is_batchers_spelling(self, two_tables):
        """Same pinned divergence as `DESCRIBE`, and pinned in the same way: the engine
        stores Arrow, so reporting a DuckDB type name would describe storage that does not
        exist."""
        session, duck = two_tables
        got = session.sql(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'orders' ORDER BY ordinal_position"
        ).to_pydict()["data_type"]
        expected = [
            r[0]
            for r in duck.sql(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'orders' ORDER BY ordinal_position"
            ).fetchall()
        ]
        assert got == ["int64", "double"]
        assert expected == ["BIGINT", "DOUBLE"]

    def test_it_is_a_relation_so_a_reflection_can_filter_it(self, two_tables):
        """A reflection asks for one table's columns, not the whole catalog."""
        session, _ = two_tables
        rows = session.sql(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'people'"
        ).to_pydict()
        assert rows["column_name"] == ["name", "age"]

    def test_an_unserved_view_says_which_are_served(self, two_tables):
        session, _ = two_tables
        with pytest.raises(Exception, match="is not served"):
            session.sql("SELECT * FROM information_schema.routines")

    def test_an_empty_session_returns_no_rows_rather_than_failing(self):
        assert bt.sql("SELECT table_name FROM information_schema.tables").to_pydict() == {
            "table_name": []
        }


_SETUP = (
    "CREATE TABLE base AS SELECT * FROM (VALUES (1, 10), (2, 20)) AS v(k, x)",
    "CREATE TABLE other AS SELECT 1 AS y",
    "CREATE VIEW big AS SELECT k FROM base WHERE x > 10",
)


class TestViewsAndSchemata:
    """A view is reported as a view, and `views`/`schemata` are served (AP-336, AP-350).

    Both engines run the same statements, so the comparison is about the catalog rather than
    how a fixture loaded it.
    """

    @pytest.fixture
    def both(self, duck):
        session = bt.Session()
        for statement in _SETUP:
            session.sql(statement)
            duck.execute(statement)
        return session, duck

    def test_table_type_matches_duckdb(self, both):
        session, duck = both
        q = "SELECT table_name, table_type FROM information_schema.tables ORDER BY table_name"
        got = session.sql(q).to_pydict()
        expected = duck.sql(q).fetchall()
        assert list(zip(got["table_name"], got["table_type"], strict=True)) == expected
        assert ("big", "VIEW") in expected

    def test_views_lists_the_views_by_name_as_duckdb_does(self, both):
        session, duck = both
        q = "SELECT table_name FROM information_schema.views"
        # DuckDB also lists its own system views, in catalog `system`; the user's are in
        # `memory`, Batcher's session views in `batcher`.
        theirs = duck.sql(q + " WHERE table_catalog = 'memory'").fetchall()
        assert session.sql(q).to_pydict()["table_name"] == [r[0] for r in theirs] == ["big"]

    def test_views_columns_are_a_subset_of_duckdbs(self, both):
        session, duck = both
        ours = session.sql("SELECT * FROM information_schema.views").columns
        theirs = list(duck.sql("SELECT * FROM information_schema.views").columns)
        assert ours == ["table_catalog", "table_schema", "table_name", "view_definition"]
        assert set(ours) <= set(theirs)

    def test_the_view_definition_is_the_stored_query_and_runs(self, both):
        """DuckDB stores the whole ``CREATE VIEW ...;`` text and re-renders the body;
        Batcher reports the query the view stores, as Postgres does. The two texts differ,
        so what is asserted is that the definition *is* the view: running it gives the
        view's rows."""
        session, _ = both
        definition = session.sql(
            "SELECT view_definition FROM information_schema.views WHERE table_name = 'big'"
        ).to_pydict()["view_definition"][0]
        assert session.sql(definition).to_pydict() == session.sql("SELECT * FROM big").to_pydict()

    def test_a_view_shadowed_by_a_per_call_table_is_a_table(self, both):
        session, _ = both
        rows = session.sql(
            "SELECT table_type FROM information_schema.tables WHERE table_name = 'big'",
            big=bt.from_pydict({"k": [1]}),
        ).to_pydict()
        assert rows == {"table_type": ["BASE TABLE"]}

    def test_schemata_lists_every_catalog_namespace(self, both):
        """Compared with DuckDB on its user catalog. DuckDB also lists its ``system`` and
        ``temp`` catalogs, which Batcher has no counterpart for, and Batcher lists the
        session's own ``batcher.main``, where session tables and views are reported."""
        session, duck = both
        session.sql("CREATE SCHEMA stg")
        duck.execute("CREATE SCHEMA stg")
        q = "SELECT catalog_name, schema_name FROM information_schema.schemata"
        got = session.sql(q).to_pydict()
        ours = set(zip(got["catalog_name"], got["schema_name"], strict=True))
        theirs = {r for r in duck.sql(q).fetchall() if r[0] == "memory"}
        assert ours - {("batcher", "main")} == theirs == {("memory", "main"), ("memory", "stg")}

    def test_schemata_names_the_schema_the_session_tables_are_reported_in(self, both):
        """`tables` and `schemata` join: every table's schema is a listed schema."""
        session, _ = both
        orphans = session.sql(
            "SELECT t.table_name FROM information_schema.tables AS t "
            "LEFT JOIN information_schema.schemata AS s "
            "ON t.table_catalog = s.catalog_name AND t.table_schema = s.schema_name "
            "WHERE s.schema_name IS NULL"
        ).to_pydict()
        assert orphans == {"table_name": []}

    def test_columns_still_lists_a_views_columns(self, both):
        session, duck = both
        q = (
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'big' ORDER BY ordinal_position"
        )
        assert session.sql(q).to_pydict()["column_name"] == [r[0] for r in duck.sql(q).fetchall()]


class TestTemporaryTables:
    """``CREATE TEMP TABLE`` is session-scoped whatever ``USE`` says (AP-322)."""

    def test_temp_after_use_creates_a_session_table_not_a_catalog_table(self):
        session = bt.Session()
        session.sql("CREATE SCHEMA stg")
        session.sql("USE stg")
        session.sql("CREATE TEMP TABLE t3 AS SELECT 1 AS x")
        assert session.catalog.list_tables() == []
        assert session.sql("SELECT x FROM t3").to_pydict() == {"x": [1]}

    def test_a_plain_create_after_use_still_writes_the_catalog(self):
        """The control: without TEMP the same statement reaches the catalog."""
        session = bt.Session()
        session.sql("CREATE SCHEMA stg")
        session.sql("USE stg")
        session.sql("CREATE TABLE t3 AS SELECT 1 AS x")
        assert session.catalog.list_tables() == ["stg.t3"]

    def test_temporary_spelling_and_temp_view(self):
        session = bt.Session()
        session.sql("CREATE SCHEMA stg")
        session.sql("USE stg")
        session.sql("CREATE TEMPORARY TABLE t4 AS SELECT 2 AS x")
        session.sql("CREATE TEMP VIEW v4 AS SELECT x + 1 AS y FROM t4")
        assert session.catalog.list_tables() == []
        assert session.sql("SELECT y FROM v4").to_pydict() == {"y": [3]}

    def test_a_qualified_temp_name_is_refused(self):
        session = bt.Session()
        session.sql("CREATE SCHEMA stg")
        with pytest.raises(PlanError, match="temporary table is a session table"):
            session.sql("CREATE TEMP TABLE stg.t5 AS SELECT 1 AS x")
        assert session.catalog.list_tables() == []

    def test_a_temp_table_answers_queries_as_duckdbs_does(self, duck):
        statement = "CREATE TEMP TABLE tt AS SELECT * FROM (VALUES (1, 'a'), (2, NULL)) v(k, s)"
        session = bt.Session()
        session.sql("CREATE SCHEMA stg")
        session.sql("USE stg")
        session.sql(statement)
        duck.execute(statement)
        q = "SELECT k, s FROM tt WHERE k >= 1"
        assert_same(session.sql(q).to_arrow(), duck.sql(q))


class TestExplain:
    """``EXPLAIN`` reads the session's dialect and raises the session's parse error (AP-352)."""

    def test_the_session_dialect_parses_the_explained_query(self):
        session = bt.Session(dialect="spark")
        session.register("t", bt.from_pydict({"a": [1]}))
        plan = session.sql("EXPLAIN SELECT `a` FROM t").to_pydict()
        assert plan["explain_key"] == ["plan"]
        assert "scan" in plan["explain_value"][0]

    def test_a_syntax_error_is_a_plan_error_without_terminal_escapes(self):
        session = bt.Session()
        with pytest.raises(PlanError) as raised:
            session.sql("EXPLAIN SELEC 1")
        assert "\x1b[" not in str(raised.value)

    def test_it_explains_a_view_and_a_catalog_table(self):
        session = bt.Session()
        session.sql("CREATE SCHEMA stg")
        session.sql("CREATE TABLE stg.t AS SELECT 1 AS a")
        session.sql("CREATE VIEW v AS SELECT a FROM stg.t")
        assert session.sql("EXPLAIN SELECT * FROM v").to_pydict()["explain_key"] == ["plan"]

    def test_explaining_a_change_is_refused_and_changes_nothing(self):
        session = bt.Session()
        session.register("t", bt.from_pydict({"a": [1]}))
        with pytest.raises(PlanError, match="EXPLAIN explains a query"):
            session.sql("EXPLAIN DELETE FROM t")
        assert session.sql("SELECT a FROM t").to_pydict() == {"a": [1]}
