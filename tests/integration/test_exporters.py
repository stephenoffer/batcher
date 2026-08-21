"""Symmetric exit adapters: `to_arrow`/`to_pandas`/`to_polars` mirror `from_*`.

These execute the plan and convert the result, so they need the native engine.
`from_pandas`/`from_polars`/… must return a `Dataset` (not a bare `Source`), and a
round-trip through a framework must preserve the data. Optional frameworks raise a
typed `BackendError` (not an opaque `ImportError`) when absent.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import BackendError


@pytest.mark.integration
def test_to_arrow_returns_table():
    ds = bt.from_pydict({"x": [1, 2, 3], "y": [10, 20, 30]}).filter(bt.col("x") > 1)
    table = ds.to_arrow()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 2
    assert sorted(table.column("x").to_pylist()) == [2, 3]


@pytest.mark.integration
def test_from_constructors_return_dataset():
    ds = bt.from_pydict({"a": [1, 2, 3]})
    assert isinstance(ds, bt.Dataset)


@pytest.mark.integration
def test_to_polars_roundtrip():
    pl = pytest.importorskip("polars")
    ds = bt.from_pydict({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df = ds.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 3
    # Round-trip back through the framework constructor.
    back = bt.from_polars(df).to_pydict()
    assert back["a"] == [1, 2, 3]


@pytest.mark.integration
def test_to_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    ds = bt.from_pydict({"a": [1, 2, 3]})
    pdf = ds.to_pandas()
    assert isinstance(pdf, pd.DataFrame)
    assert bt.from_pandas(pdf).to_pydict()["a"] == [1, 2, 3]


@pytest.mark.integration
def test_missing_framework_raises_backend_error(monkeypatch):
    """With pandas genuinely unimportable, `to_pandas` raises the typed error.

    The absence is simulated by poisoning the `sys.modules` entry, which is what makes
    *every* spelling of the import fail. Patching `builtins.__import__` does not: the guard
    goes through `importlib.import_module`, which returns a module already in `sys.modules`
    without consulting `__import__` at all. So once any earlier test in the process had
    imported pandas, that simulation left the guard seeing pandas as present, and the run
    failed deep inside pyarrow's pandas bridge instead — a pass or fail decided by test
    order rather than by the behavior under test.
    """
    import sys

    monkeypatch.setitem(sys.modules, "pandas", None)
    ds = bt.from_pydict({"a": [1]})
    with pytest.raises(BackendError, match="\\[pandas\\]"):
        ds.to_pandas()
