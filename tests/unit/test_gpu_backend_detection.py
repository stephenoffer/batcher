"""`DfBackend` tells cuDF from pandas by the library, not by a method the two now share.

It used to probe `hasattr(lib.DataFrame, "from_arrow")`. pandas 3.0 added that classmethod --
without cuDF's `to_arrow` -- so under pandas 3 the pandas backend classified itself as a device
and every result conversion raised `'DataFrame' object has no attribute 'to_arrow'`. Measured on
an A10G job with cuDF 26.08 (which requires pandas >= 3): 501 of the translator suite's cases
failed that way before any cuDF code ran. The pandas backend is not test-only: the router and the
distributed aggregate build `DfBackend(pandas)` in production.

CI runs pandas 2, where the old probe happened to be right, so these use stand-in modules to pin
the pandas-3 shape without needing pandas 3 installed.
"""

from __future__ import annotations

import types

import pandas as pd

from batcher.core.gpu_plan.backend import DfBackend


def _lib(name: str, *, from_arrow: bool) -> types.ModuleType:
    class DataFrame:
        pass

    if from_arrow:
        DataFrame.from_arrow = staticmethod(lambda table: table)  # type: ignore[attr-defined]
    lib = types.ModuleType(name)
    lib.DataFrame = DataFrame
    return lib


def test_pandas_with_from_arrow_is_still_the_host_backend():
    """The pandas-3 shape: `DataFrame.from_arrow` exists and the library is still pandas."""
    assert not DfBackend(_lib("pandas", from_arrow=True)).is_gpu


def test_cudf_is_the_device_backend():
    assert DfBackend(_lib("cudf", from_arrow=True)).is_gpu


def test_the_installed_pandas_is_the_host_backend():
    assert not DfBackend(pd).is_gpu


def test_pandas_3_declines_a_nan_bearing_input_instead_of_reading_it_as_missing():
    """Under pandas 3 a float NaN in an Arrow-backed column is *missing*: `sum` skips it and a
    NaN group key can fold into the null group, where the engine keeps NaN a value. The host
    backend therefore declines such input, and the CPU engine answers."""
    import pyarrow as pa
    import pytest

    from batcher.core.gpu_plan.backend import Unsupported

    lib = _lib("pandas", from_arrow=True)
    lib.__version__ = "3.0.3"
    be = DfBackend(lib)
    assert be.nan_is_missing
    with pytest.raises(Unsupported):
        be.from_arrow(pa.table({"v": pa.array([1.0, float("nan")])}))


def test_pandas_2_and_cudf_keep_nan_a_value():
    old = _lib("pandas", from_arrow=False)
    old.__version__ = "2.3.3"
    device = _lib("cudf", from_arrow=True)
    device.__version__ = "26.08.01"
    assert not DfBackend(old).nan_is_missing
    assert not DfBackend(device).nan_is_missing
