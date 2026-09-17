"""The `Catalog` contract, held identically over the in-memory and Delta-directory backends.

Every rule about names and modes lives once in `Catalog` / `api.catalog.modes`, and the
backends only store. So the same assertions run against both, parametrized: a backend that
drifts on what ``mode="error"``, ``cascade`` or ``overwrite_partitions`` means fails here by
name. The Iceberg backend has its own file, against a pyiceberg catalog held in a dict
(`tests/io/test_catalog_iceberg_backend.py`), because it needs that fake to exist at all.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit


@pytest.fixture(params=["memory", "directory"])
def catalog(request, tmp_path) -> bt.Catalog:
    if request.param == "memory":
        return bt.Catalog.from_pydict({}, name="cat")
    pytest.importorskip("deltalake")
    return bt.Catalog.from_directory(str(tmp_path / "wh"), name="cat")


@pytest.fixture
def session(catalog) -> bt.Session:
    s = bt.Session()
    s.catalog.attach(catalog)
    s.catalog.use("cat")
    return s


def _sorted_ids(ds: bt.Dataset) -> list:
    return ds.sort("id").to_pydict()["id"]


class TestNamespaces:
    def test_the_default_namespace_exists_and_cannot_be_dropped(self, catalog):
        assert catalog.list_namespaces() == ["main"]
        assert catalog.has_namespace("main")
        with pytest.raises(PlanError, match="default namespace"):
            catalog.drop_namespace("main")

    def test_create_refuses_a_duplicate_unless_if_not_exists(self, catalog):
        catalog.create_namespace("raw")
        with pytest.raises(PlanError, match="already exists"):
            catalog.create_namespace("raw")
        catalog.create_namespace("raw", if_not_exists=True)
        assert catalog.list_namespaces() == ["main", "raw"]

    def test_drop_refuses_a_missing_namespace_unless_if_exists(self, catalog):
        with pytest.raises(PlanError, match="no namespace"):
            catalog.drop_namespace("nope")
        catalog.drop_namespace("nope", if_exists=True)

    def test_drop_refuses_a_namespace_with_tables_unless_cascade(self, catalog):
        catalog.create_namespace("raw")
        catalog.create_table("raw.t", {"id": [1]})
        with pytest.raises(PlanError, match="still holds 1 table"):
            catalog.drop_namespace("raw")
        assert catalog.has_table("raw.t")
        catalog.drop_namespace("raw", cascade=True)
        assert catalog.list_namespaces() == ["main"]
        assert catalog.list_tables() == []

    def test_the_pattern_is_a_glob(self, catalog):
        for name in ("raw", "raw_eu", "gold"):
            catalog.create_namespace(name)
        assert catalog.list_namespaces("raw*") == ["raw", "raw_eu"]


class TestTables:
    def test_create_then_read_back(self, catalog):
        table = catalog.create_table("orders", bt.from_pydict({"id": [1, 2]}))
        assert table.name == "cat.main.orders"
        assert table.schema.names == ["id"]
        assert _sorted_ids(table.read()) == [1, 2]

    def test_a_schema_creates_an_empty_table(self, catalog):
        table = catalog.create_table("e", pa.schema([("id", pa.int64()), ("s", pa.string())]))
        assert table.read().count() == 0
        assert table.schema.names == ["id", "s"]

    def test_create_refuses_a_duplicate_unless_if_not_exists(self, catalog):
        catalog.create_table("t", {"id": [1]})
        with pytest.raises(PlanError, match="already exists"):
            catalog.create_table("t", {"id": [2]})
        kept = catalog.create_table("t", {"id": [2]}, if_not_exists=True)
        assert _sorted_ids(kept.read()) == [1]

    def test_create_needs_the_namespace(self, catalog):
        with pytest.raises(PlanError, match="no namespace 'raw'"):
            catalog.create_table("raw.t", {"id": [1]})

    def test_list_tables_qualifies_by_namespace_and_filters_by_glob(self, catalog):
        catalog.create_namespace("raw")
        catalog.create_table("a", {"id": [1]})
        catalog.create_table("raw.b", {"id": [1]})
        assert catalog.list_tables() == ["main.a", "raw.b"]
        assert catalog.list_tables("raw.*") == ["raw.b"]

    def test_drop_and_if_exists(self, catalog):
        catalog.create_table("t", {"id": [1]})
        catalog.drop_table("main.t")
        assert not catalog.has_table("t")
        with pytest.raises(PlanError, match="no table"):
            catalog.drop_table("t")
        catalog.drop_table("t", if_exists=True)

    def test_a_dropped_name_can_be_created_again_with_new_rows(self, catalog):
        """The Delta directory case restarts the log at version 0 at the same path, which a
        version-keyed snapshot cache would answer with the dropped table's files."""
        catalog.create_table("t", {"id": [1, 2, 3]})
        assert catalog.get_table("t").read().count() == 3
        catalog.drop_table("t")
        catalog.create_table("t", {"id": [7]})
        assert _sorted_ids(catalog.get_table("t").read()) == [7]

    def test_truncate_keeps_schema(self, catalog):
        catalog.create_table("t", {"id": [1, 2], "s": ["a", "b"]})
        catalog.truncate_table("t")
        table = catalog.get_table("t")
        assert table.read().count() == 0
        assert table.schema.names == ["id", "s"]

    def test_get_table_on_a_missing_table_names_what_exists(self, catalog):
        catalog.create_table("orders", {"id": [1]})
        with pytest.raises(PlanError, match="no table 'order'") as err:
            catalog.get_table("order")
        assert "main.orders" in str(err.value)

    def test_an_empty_name_part_is_refused(self, catalog):
        with pytest.raises(PlanError, match="non-empty"):
            catalog.has_table("raw..t")


class TestWriteModes:
    def test_error_creates_then_refuses(self, session):
        bt.from_pydict({"id": [1]}).write.table("t", session=session)
        with pytest.raises(PlanError, match="already exists"):
            bt.from_pydict({"id": [2]}).write.table("t", session=session)
        assert _sorted_ids(session.table("t")) == [1]

    def test_ignore_creates_then_leaves_the_table(self, session):
        bt.from_pydict({"id": [1]}).write.table("t", mode="ignore", session=session)
        manifest = bt.from_pydict({"id": [2]}).write.table("t", mode="ignore", session=session)
        assert manifest.total_rows == 0
        assert _sorted_ids(session.table("t")) == [1]

    def test_append_creates_when_missing_then_adds(self, session):
        bt.from_pydict({"id": [1]}).write.table("t", mode="append", session=session)
        bt.from_pydict({"id": [2, 3]}).write.table("t", mode="append", session=session)
        assert _sorted_ids(session.table("t")) == [1, 2, 3]

    def test_append_matches_by_name_and_fills_missing_columns_with_null(self, session):
        bt.from_pydict({"id": [1], "s": ["a"]}).write.table("t", session=session)
        bt.from_pydict({"s": ["b"], "id": [2]}).write.table("t", mode="append", session=session)
        bt.from_pydict({"id": [3]}).write.table("t", mode="append", session=session)
        got = session.table("t").sort("id").to_pydict()
        assert got == {"id": [1, 2, 3], "s": ["a", "b", None]}

    def test_append_by_position_ignores_names_and_casts(self, session):
        bt.from_pydict({"id": [1], "s": ["a"]}).write.table("t", session=session)
        bt.from_pydict({"x": [2], "y": ["b"]}).write.table(
            "t", mode="append", by_name=False, session=session
        )
        assert session.table("t").sort("id").to_pydict() == {"id": [1, 2], "s": ["a", "b"]}

    def test_by_position_needs_an_existing_table(self, session):
        with pytest.raises(PlanError, match="by_name=False"):
            bt.from_pydict({"id": [1]}).write.table(
                "t", mode="append", by_name=False, session=session
            )

    def test_overwrite_replaces_rows_and_schema(self, session):
        bt.from_pydict({"id": [1, 2]}).write.table("t", session=session)
        bt.from_pydict({"z": ["q"]}).write.table("t", mode="overwrite", session=session)
        assert session.table("t").to_pydict() == {"z": ["q"]}

    def test_overwrite_with_replace_where_keeps_the_rest(self, session):
        """Partitioned, because a Delta table scopes `replace_where` to partition columns."""
        start = bt.from_pydict({"id": [1, 2, 3], "p": ["a", "b", "a"]})
        start.write.table("t", partition_by=["p"], session=session)
        bt.from_pydict({"id": [9], "p": ["a"]}).write.table(
            "t", mode="overwrite", replace_where=bt.col("p") == "a", session=session
        )
        got = session.table("t").sort("id").to_pydict()
        assert got == {"id": [2, 9], "p": ["b", "a"]}

    def test_replace_where_keeps_rows_the_predicate_is_null_on(self):
        """In memory the predicate may be over any column; a NULL is not a match."""
        s = bt.Session()
        bt.from_pydict({"id": [1, 2], "v": [5, None]}).write.table("t", session=s)
        bt.from_pydict({"id": [3], "v": [7]}).write.table(
            "t", mode="overwrite", replace_where=bt.col("v") > 1, session=s
        )
        assert s.table("t").sort("id").to_pydict() == {"id": [2, 3], "v": [None, 7]}

    def test_replace_needs_an_existing_table(self, session):
        with pytest.raises(PlanError, match="needs an existing table"):
            bt.from_pydict({"id": [1]}).write.table("t", mode="replace", session=session)
        bt.from_pydict({"id": [1]}).write.table("t", session=session)
        bt.from_pydict({"id": [5]}).write.table("t", mode="replace", session=session)
        assert _sorted_ids(session.table("t")) == [5]

    def test_overwrite_partitions_replaces_only_the_covered_partitions(self, session):
        start = bt.from_pydict({"id": [1, 2, 3], "p": ["a", "b", "c"]})
        start.write.table("t", partition_by=["p"], session=session)
        bt.from_pydict({"id": [10, 11], "p": ["a", "c"]}).write.table(
            "t", mode="overwrite_partitions", session=session
        )
        got = session.table("t").sort("id").to_pydict()
        assert got == {"id": [2, 10, 11], "p": ["b", "a", "c"]}

    def test_overwrite_partitions_on_an_unpartitioned_table_is_refused(self, session):
        bt.from_pydict({"id": [1], "p": ["a"]}).write.table("t", session=session)
        with pytest.raises(PlanError, match="needs partition columns"):
            bt.from_pydict({"id": [2], "p": ["a"]}).write.table(
                "t", mode="overwrite_partitions", session=session
            )

    def test_an_unknown_mode_lists_the_modes(self, session):
        with pytest.raises(PlanError, match="unknown mode") as err:
            bt.from_pydict({"id": [1]}).write.table("t", mode="upsert", session=session)
        assert "overwrite_partitions" in str(err.value)

    def test_replace_where_only_scopes_an_overwrite(self, session):
        with pytest.raises(PlanError, match="scopes an overwrite"):
            bt.from_pydict({"id": [1]}).write.table(
                "t", mode="append", replace_where=bt.col("id") == 1, session=session
            )

    def test_properties_apply_on_create(self, session):
        bt.from_pydict({"id": [1]}).write.table(
            "t", properties={"delta.appendOnly": "false"}, session=session
        )
        assert session.catalog.get_table("t").properties["delta.appendOnly"] == "false"


class TestSessionResolution:
    def test_names_resolve_through_catalog_and_namespace(self, session, catalog):
        catalog.create_namespace("raw")
        catalog.create_table("raw.t", {"id": [1]})
        for name in ("cat.raw.t", "raw.t"):
            assert session.table(name).count() == 1
        session.catalog.use("raw")
        assert session.table("t").count() == 1

    def test_a_two_part_name_can_name_a_catalog(self, catalog):
        s = bt.Session()
        s.catalog.attach(catalog)
        catalog.create_table("t", {"id": [4]})
        assert _sorted_ids(s.table("cat.t")) == [4]

    def test_a_view_shadows_a_catalog_table_and_cannot_be_written(self, session):
        bt.from_pydict({"id": [1]}).write.table("t", session=session)
        session.register("t", bt.from_pydict({"id": [99]}))
        assert _sorted_ids(session.table("t")) == [99]
        with pytest.raises(PlanError, match="session view"):
            bt.from_pydict({"id": [2]}).write.table("t", mode="append", session=session)
        session.drop("t")
        assert _sorted_ids(session.table("t")) == [1]

    def test_a_missing_name_says_both_places_were_looked_in(self):
        with pytest.raises(
            PlanError, match="no table 'nope': not a registered view, nor a catalog table"
        ):
            bt.Session().table("nope")

    def test_use_refuses_an_unknown_name(self, session):
        with pytest.raises(PlanError, match="cannot use 'nowhere'"):
            session.catalog.use("nowhere")

    def test_attach_detach(self, catalog):
        s = bt.Session()
        s.catalog.attach(catalog, alias="other")
        assert s.catalog.list_catalogs() == ["memory", "other"]
        assert s.catalog.get_catalog("other").name == "other"
        with pytest.raises(PlanError, match="already attached"):
            s.catalog.attach(catalog, alias="other")
        s.catalog.use("other")
        with pytest.raises(PlanError, match="current catalog"):
            s.catalog.detach("other")
        s.catalog.use("memory")
        s.catalog.detach("other")
        assert not s.catalog.has_catalog("other")
        with pytest.raises(PlanError, match="no catalog 'other'"):
            s.catalog.detach("other")

    def test_attach_takes_only_a_catalog(self):
        with pytest.raises(PlanError, match=r"Session\.register"):
            bt.Session().catalog.attach(bt.from_pydict({"x": [1]}))

    def test_the_current_namespace_cannot_be_dropped(self, session):
        session.catalog.create_namespace("raw")
        session.catalog.use("raw")
        with pytest.raises(PlanError, match="current namespace"):
            session.catalog.drop_namespace("raw")

    def test_sessions_do_not_share_catalog_state(self):
        a, b = bt.Session(), bt.Session()
        a.catalog.create_table("t", {"id": [1]})
        assert a.catalog.has_table("t")
        assert not b.catalog.has_table("t")


class TestSessionSurface:
    def test_register_without_replace_refuses_an_existing_view(self):
        s = bt.Session()
        s.register("v", bt.from_pydict({"x": [1]}))
        with pytest.raises(PlanError, match="already registered"):
            s.register("v", bt.from_pydict({"x": [2]}), replace=False)
        assert s.table("v").to_pydict() == {"x": [1]}
        s.register("w", bt.from_pydict({"x": [3]}), replace=False)
        assert s.list() == ["v", "w"]

    def test_function_lifecycle(self):
        s = bt.Session()
        s.register_function("dbl", lambda a: a, result_type="int64")
        assert s.has_function("dbl")
        s.drop_function("dbl")
        assert not s.has_function("dbl")
        with pytest.raises(PlanError, match="no function 'dbl'"):
            s.drop_function("dbl")
        s.drop_function("dbl", if_exists=True)

    def test_set_session_changes_what_bt_sql_and_write_table_use(self):
        previous = bt.current_session()
        fresh = bt.Session()
        try:
            bt.set_session(fresh)
            bt.from_pydict({"id": [1, 2]}).write.table("t")
            assert fresh.catalog.has_table("t")
            assert bt.sql("SELECT count(*) AS n FROM t").to_pydict() == {"n": [2]}
        finally:
            bt.set_session(previous)
        assert not previous.catalog.has_table("t")

    def test_set_session_takes_only_a_session(self):
        with pytest.raises(PlanError, match=r"bt\.Session"):
            bt.set_session(object())

    def test_a_dialect_view_shares_the_catalog(self):
        s = bt.Session()
        s.catalog.create_table("t", {"id": [1]})
        assert s._with_dialect("spark").catalog is s.catalog
