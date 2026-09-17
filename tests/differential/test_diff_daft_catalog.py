"""The catalog workflow a Daft user writes, run through Daft and through Batcher.

Daft's session API and Batcher's name the same operations (`attach`, `create_namespace`,
`create_table`, `list_tables`, `read_table` / `table`, `drop_table`), so a ported script
should see the same tables and the same rows. This runs one in-memory script both ways and
compares what each engine reports. It checks the *shape* of the API — which names exist and
what they list — and the rows, not Daft's identifier objects: Daft returns `Identifier`,
Batcher a dotted string, and the comparison is on the strings.

Daft is the oracle here, not DuckDB: DuckDB has no Python catalog object to hold these calls
against. The SQL spelling of the same operations is checked against DuckDB in
`test_diff_sql_catalog_tables.py`.

One naming difference is visible here by design. Daft's session lists tables fully
qualified (``mem.ns.t``); `SessionCatalog.list_tables` lists them relative to the current
catalog (``ns.t``), as `Catalog.list_tables` does in both engines.
"""

from __future__ import annotations

import pytest

import batcher as bt

daft = pytest.importorskip("daft")

pytestmark = pytest.mark.differential


def test_the_same_script_sees_the_same_tables_and_rows():
    d_session = daft.Session()
    d_catalog = daft.Catalog.from_pydict({"ns.t": {"x": [1, 2]}}, name="mem")
    d_session.attach_catalog(d_catalog)
    d_session.use("mem")
    d_session.create_table("ns.u", daft.from_pydict({"y": [3, 4]}))

    b_session = bt.Session()
    b_catalog = bt.Catalog.from_pydict({"ns.t": {"x": [1, 2]}}, name="mem")
    b_session.catalog.attach(b_catalog)
    b_session.catalog.use("mem")
    b_session.catalog.create_table("ns.u", bt.from_pydict({"y": [3, 4]}))

    assert [str(i) for i in d_catalog.list_tables()] == b_catalog.list_tables() == ["ns.t", "ns.u"]
    assert [str(i) for i in d_session.list_tables()] == [
        f"mem.{name}" for name in b_session.catalog.list_tables()
    ]
    assert [str(n) for n in d_catalog.list_namespaces()] == ["ns"]
    assert "ns" in b_catalog.list_namespaces()
    for name in ("mem.ns.t", "ns.u"):
        assert d_session.read_table(name).to_pydict() == b_session.table(name).to_pydict()
    assert d_session.has_table("ns.u") is b_session.catalog.has_table("ns.u") is True

    d_session.drop_table("ns.u")
    b_session.catalog.drop_table("ns.u")
    assert d_session.has_table("ns.u") is b_session.catalog.has_table("ns.u") is False
