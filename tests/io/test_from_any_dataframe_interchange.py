"""`bt.from_any` accepts an object that exports only the DataFrame interchange protocol.

That is what Polars' ``from_dataframe`` consumes. The interchange objects pandas and Polars
hand out are the natural fixtures: neither exports ``__arrow_c_stream__``, and pandas' one
subclasses a protocol class that is *also* called ``DataFrame``, which the type dispatch used
to mistake for a pandas frame.
"""

from __future__ import annotations

import warnings

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.io


def test_a_pandas_interchange_object_converts_with_nulls():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"a": [1.5, None], "s": ["x", None]})
    exchange = frame.__dataframe__()
    assert not hasattr(exchange, "__arrow_c_stream__")
    ds = bt.from_any(exchange)
    assert ds.schema == pa.schema([("a", pa.float64()), ("s", pa.string())])
    assert ds.to_pydict() == {"a": [1.5, None], "s": ["x", None]}


def test_a_polars_interchange_object_converts_like_the_frame_itself():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"i": [1, None, 3], "s": ["a", "b", None]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # Polars deprecates the protocol
        exchange = frame.__dataframe__()
    via_protocol = bt.from_any(exchange)
    direct = bt.from_polars(frame)
    assert via_protocol.to_pydict() == direct.to_pydict() == frame.to_dict(as_series=False)
    assert via_protocol.schema.field("i").type == pa.int64()


def test_a_real_pandas_frame_still_takes_the_pandas_path():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"n": [1, 2]})
    assert bt.from_any(frame).to_pydict() == {"n": [1, 2]}


def test_an_empty_interchange_frame_keeps_its_columns():
    pd = pytest.importorskip("pandas")
    exchange = pd.DataFrame({"a": pd.Series([], dtype="int64")}).__dataframe__()
    ds = bt.from_any(exchange)
    assert ds.columns == ["a"]
    assert ds.count() == 0
