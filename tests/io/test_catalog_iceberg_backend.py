"""`Catalog.from_iceberg` against a real pyiceberg catalog whose metastore is a dict.

pyiceberg's own local catalogs (`SqlCatalog`, `InMemoryCatalog`) both need sqlite, which
cannot be imported in every environment this suite runs in. The Iceberg backend is written
against pyiceberg's `Catalog` class, not a service, so the contract is exercised with the
smallest real subclass: `MetastoreCatalog` with the table-pointer map held in a dict. Every
metadata file, manifest and data file is real and written by pyiceberg, and every read and
write goes through `bt.read.iceberg` / `ds.write.iceberg`. What is faked is only where the
current metadata location of each table is recorded — the one thing a service provides.

It proves the backend speaks pyiceberg's API correctly. It does not prove any particular
service (Glue, REST, Unity) behaves like this dict, which is a property of that service.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytest.importorskip("pyiceberg")

from pyiceberg.catalog import MetastoreCatalog
from pyiceberg.exceptions import (
    NamespaceNotEmptyError,
    NoSuchNamespaceError,
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.io import load_file_io
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.serializers import FromInputFile
from pyiceberg.table import CommitTableResponse, Table
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER

pytestmark = pytest.mark.integration


class DictCatalog(MetastoreCatalog):
    """A pyiceberg metastore catalog whose table pointers live in a dict."""

    def __init__(self, warehouse: str) -> None:
        super().__init__("dict", warehouse=f"file://{warehouse}")
        self._namespaces: set[tuple[str, ...]] = set()
        self._tables: dict[tuple[str, ...], str] = {}

    def create_table(
        self,
        identifier,
        schema,
        location=None,
        partition_spec=UNPARTITIONED_PARTITION_SPEC,
        sort_order=UNSORTED_SORT_ORDER,
        properties=None,
    ):
        ident = self.identifier_to_tuple(identifier)
        if ident[:-1] not in self._namespaces:
            raise NoSuchNamespaceError(ident[:-1])
        if ident in self._tables:
            raise TableAlreadyExistsError(ident)
        staged = self._create_staged_table(
            identifier, schema, location, partition_spec, sort_order, properties or {}
        )
        self._write_metadata(staged.metadata, staged.io, staged.metadata_location)
        self._tables[ident] = staged.metadata_location
        return self.load_table(identifier)

    def load_table(self, identifier):
        ident = self.identifier_to_tuple(identifier)
        if ident not in self._tables:
            raise NoSuchTableError(ident)
        location = self._tables[ident]
        io = load_file_io(self.properties, location)
        metadata = FromInputFile.table_metadata(io.new_input(location))
        return Table(
            identifier=ident,
            metadata=metadata,
            metadata_location=location,
            io=self._load_file_io(metadata.properties, location),
            catalog=self,
        )

    def commit_table(self, table, requirements, updates):
        ident = self.identifier_to_tuple(table.name())
        current = self.load_table(ident) if ident in self._tables else None
        staged = self._update_and_stage_table(current, ident, requirements, updates)
        self._write_metadata(staged.metadata, staged.io, staged.metadata_location)
        self._tables[ident] = staged.metadata_location
        return CommitTableResponse(
            metadata=staged.metadata, metadata_location=staged.metadata_location
        )

    def drop_table(self, identifier):
        if self._tables.pop(self.identifier_to_tuple(identifier), None) is None:
            raise NoSuchTableError(identifier)

    def create_namespace(self, namespace, properties=None):
        self._namespaces.add(self.identifier_to_tuple(namespace))

    def drop_namespace(self, namespace):
        ns = self.identifier_to_tuple(namespace)
        if any(ident[:-1] == ns for ident in self._tables):
            raise NamespaceNotEmptyError(ns)
        self._namespaces.discard(ns)

    def list_tables(self, namespace):
        ns = self.identifier_to_tuple(namespace)
        return [ident for ident in self._tables if ident[:-1] == ns]

    def list_namespaces(self, namespace=()):
        return sorted(self._namespaces)

    def load_namespace_properties(self, namespace):
        if self.identifier_to_tuple(namespace) not in self._namespaces:
            raise NoSuchNamespaceError(namespace)
        return {}

    def update_namespace_properties(self, namespace, removals=None, updates=None):
        raise NotImplementedError

    def register_table(self, identifier, metadata_location, overwrite=False):
        raise NotImplementedError

    def rename_table(self, from_identifier, to_identifier):
        raise NotImplementedError

    def list_views(self, namespace):
        return []

    def load_view(self, identifier):
        raise NotImplementedError

    def view_exists(self, identifier):
        return False

    def drop_view(self, identifier):
        raise NotImplementedError

    def register_view(self, identifier, metadata_location):
        raise NotImplementedError


@pytest.fixture
def lake(tmp_path) -> tuple[bt.Session, bt.Catalog]:
    live = DictCatalog(str(tmp_path))
    live.create_namespace("default")
    catalog = bt.Catalog.from_iceberg(live, name="ice")
    session = bt.Session()
    session.catalog.attach(catalog)
    return session, catalog


def test_the_catalog_name_defaults_to_pyicebergs(tmp_path):
    assert bt.Catalog.from_iceberg(DictCatalog(str(tmp_path))).name == "dict"


def test_namespace_lifecycle(lake):
    _, catalog = lake
    catalog.create_namespace("sales")
    assert catalog.list_namespaces() == ["default", "sales"]
    catalog.drop_namespace("sales")
    assert catalog.list_namespaces() == ["default"]


def test_create_read_append_overwrite_drop(lake):
    session, catalog = lake
    catalog.create_namespace("sales")
    table = catalog.create_table(
        "sales.orders", bt.from_pydict({"id": [1, 2]}), properties={"owner": "ops"}
    )
    assert table.name == "ice.sales.orders"
    assert table.properties["owner"] == "ops"
    assert catalog.list_tables() == ["sales.orders"]

    bt.from_pydict({"id": [3]}).write.table("ice.sales.orders", mode="append", session=session)
    assert session.table("ice.sales.orders").sort("id").to_pydict() == {"id": [1, 2, 3]}
    assert session.sql("SELECT count(*) AS n FROM ice.sales.orders").to_pydict() == {"n": [3]}

    bt.from_pydict({"id": [9]}).write.table("ice.sales.orders", mode="overwrite", session=session)
    assert session.table("ice.sales.orders").to_pydict() == {"id": [9]}

    with pytest.raises(PlanError, match="already exists"):
        bt.from_pydict({"id": [0]}).write.table("ice.sales.orders", session=session)

    catalog.drop_table("sales.orders")
    assert not catalog.has_table("sales.orders")


def test_an_overwrite_with_a_new_schema_recreates_the_table(lake):
    session, catalog = lake
    catalog.create_table("t", bt.from_pydict({"id": [1]}))
    bt.from_pydict({"name": ["x"]}).write.table("ice.t", mode="overwrite", session=session)
    assert session.table("ice.t").to_pydict() == {"name": ["x"]}


def test_partitioning_on_create_is_refused_not_dropped(lake):
    _, catalog = lake
    with pytest.raises(PlanError, match="partition spec"):
        catalog.create_table("t", bt.from_pydict({"id": [1], "p": ["a"]}), partition_by=["p"])
    assert not catalog.has_table("t")


def test_a_namespace_with_tables_needs_cascade(lake):
    _, catalog = lake
    catalog.create_namespace("raw")
    catalog.create_table("raw.t", bt.from_pydict({"id": [1]}))
    with pytest.raises(PlanError, match="still holds"):
        catalog.drop_namespace("raw")
    catalog.drop_namespace("raw", cascade=True)
    assert catalog.list_namespaces() == ["default"]
