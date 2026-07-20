"""A lakehouse split must carry everything that decides which rows it yields.

A worker never sees the source object — it receives a *pickled split* and rebuilds a
reader from its fields. Anything the source knew and the split does not is therefore
applied single-node and dropped distributed, and the failure mode is the worst kind:
the same query returns different rows depending on how it ran, with no error, because
the extra rows are real rows from a real file.

Two such fields were being dropped, both reachable from the public API:

* Iceberg's constructor ``row_filter`` — `plan_files` prunes at *file* granularity, so
  surviving files still hold non-matching rows, and the filter is not part of Kyber's
  pushed predicate, so no `Filter` re-checks them. Measured before the fix: single-node
  10 rows, distributed 100.
* Hudi's ``as_of_instant`` — `splits()` enumerated the *latest* slices, so a time-travel
  read returned the current table.
"""

from __future__ import annotations

import pickle

import pyarrow as pa
import pytest

pytestmark = pytest.mark.unit


# ---- Iceberg -----------------------------------------------------------------


@pytest.fixture
def iceberg_table(tmp_path):
    """A real 100-row Iceberg table over a local SQL catalog."""
    pytest.importorskip("pyiceberg")
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "wh"
    warehouse.mkdir()
    uri = f"sqlite:///{tmp_path}/cat.db"
    catalog = SqlCatalog("c", uri=uri, warehouse=f"file://{warehouse}")
    catalog.create_namespace("ns")
    data = pa.table({"id": pa.array(list(range(100)), pa.int64())})
    catalog.create_table("ns.t", schema=data.schema).append(data)
    return {"name": "c", "type": "sql", "uri": uri, "warehouse": f"file://{warehouse}"}


def _rows(batches) -> int:
    return sum(b.num_rows for b in batches)


def test_a_row_filter_is_applied_by_every_split(iceberg_table) -> None:
    from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

    source = IcebergSource("ns.t", catalog=iceberg_table, row_filter="id < 10")
    single_node = _rows(source.read())
    distributed = sum(_rows(s.read()) for s in source.splits())

    assert single_node == 10
    assert distributed == single_node, "the splits returned rows the row_filter excludes"


def test_a_row_filter_survives_being_shipped_to_a_worker(iceberg_table) -> None:
    """The split is pickled to the worker; the filter has to be inside it."""
    from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

    source = IcebergSource("ns.t", catalog=iceberg_table, row_filter="id < 10")
    shipped = [pickle.loads(pickle.dumps(s)) for s in source.splits()]

    assert sum(_rows(s.read()) for s in shipped) == 10


def test_no_row_filter_still_reads_everything(iceberg_table) -> None:
    """The fix must not turn an absent filter into a restrictive one."""
    from batcher.io.formats.lakehouse.iceberg.source import IcebergSource

    source = IcebergSource("ns.t", catalog=iceberg_table)

    assert _rows(source.read()) == 100
    assert sum(_rows(s.read()) for s in source.splits()) == 100


# ---- Hudi --------------------------------------------------------------------


class _FakeSlice:
    num_records = 5

    def base_file_relative_path(self) -> str:
        return "p/base.parquet"


class _FakeHudiTable:
    """Records which slice-enumeration call the source made.

    A real Hudi table with history needs a Spark writer to produce, so the behaviour
    under test — *which* API the source asks for its slices — is observed directly. That
    is the whole bug: `read()` used the as-of call and `splits()` did not.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def get_file_slices(self, filters=None):
        self.calls.append(("latest", None))
        return [_FakeSlice()]

    def get_file_slices_as_of(self, instant, filters=None):
        self.calls.append(("as_of", instant))
        return [_FakeSlice()]


def test_splits_enumerate_the_as_of_slices(monkeypatch) -> None:
    from batcher.io.formats.lakehouse import hudi as hudi_mod

    fake = _FakeHudiTable()
    # `HudiSource` uses `__slots__`, so the stub goes on the class.
    monkeypatch.setattr(hudi_mod.HudiSource, "_table", lambda self: fake)
    source = hudi_mod.HudiSource("file:///tmp/t", as_of_instant="20260718120000000")

    source._file_slices()

    assert fake.calls == [("as_of", "20260718120000000")], (
        "a time-travel read enumerated the latest slices"
    )


def test_splits_without_an_instant_use_the_latest_slices(monkeypatch) -> None:
    from batcher.io.formats.lakehouse import hudi as hudi_mod

    fake = _FakeHudiTable()
    monkeypatch.setattr(hudi_mod.HudiSource, "_table", lambda self: fake)
    source = hudi_mod.HudiSource("file:///tmp/t")

    source._file_slices()

    assert fake.calls == [("latest", None)]


def test_a_hudi_split_carries_the_instant() -> None:
    """The worker rebuilds its reader from the split alone."""
    from batcher.io.formats.lakehouse.hudi import HudiFileSliceSplit

    split = HudiFileSliceSplit("file:///tmp/t", "p/base.parquet", {}, 5, "20260718120000000")

    assert pickle.loads(pickle.dumps(split)).as_of_instant == "20260718120000000"
