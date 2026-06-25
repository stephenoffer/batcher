"""PyArrow adapter — single-node Acero/compute comparator.

PyArrow has no SQL surface, so it sits out the standard SQL suites (``n/a``) and
participates in the operator-mix, where cases build on ``pyarrow.compute`` and the
Acero-backed ``Table`` group-by/join/sort methods. The handle is the Arrow table
itself (PyArrow is also the cross-engine interchange format).
"""

from __future__ import annotations

import importlib.util

import pyarrow as pa

from .base import Engine


class PyArrowEngine(Engine):
    name = "pyarrow"
    tier = "single"
    supports_sql = False

    @classmethod
    def available(cls) -> bool:
        return importlib.util.find_spec("pyarrow") is not None

    def handle(self, table: pa.Table) -> pa.Table:
        return table

    def read_parquet(self, uri: str) -> pa.Table:
        import pyarrow.dataset as ds

        return ds.dataset(uri).to_table()
