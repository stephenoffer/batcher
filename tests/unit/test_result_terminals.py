"""The result terminals' options: stats, to_numpy, to_pandas, __array__, show, equals, iterators.

AP-162 `stats(keep_result=True)`, AP-170 `to_numpy(nulls=)`, AP-171
`to_pandas(dtype_backend=)`, AP-172 ``np.asarray(ds, copy=False)``, AP-173/174 `show`'s
widths, `file` and footer plus the notebook repr's column cap, AP-179 `equals` tolerances, and
AP-167 closing an `iter_batches` iterator. Every default is checked to be unchanged.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_tables_equal

# --- AP-162: stats(keep_result=True) ------------------------------------------------------


def test_stats_keeps_the_result_from_the_same_run():
    calls: list[int] = []

    def count(batch):
        calls.append(batch.num_rows)
        return batch

    ds = bt.from_pydict({"k": ["a", "b", "a"], "v": [1, 2, 3]}).map_batches(count)
    run = ds.filter(bt.col("v") > 1).stats(keep_result=True)
    ran = len(calls)
    assert ran >= 1
    assert run.rows == run.result.num_rows == 2
    assert_tables_equal(run.result, ds.filter(bt.col("v") > 1).collect())
    assert len(calls) == ran * 2  # exactly one more run: the collect() above


def test_stats_discards_the_result_by_default():
    run = bt.from_pydict({"x": [1, 2]}).stats()
    assert run.result is None
    kept = bt.from_pydict({"x": [1, 2]}).stats(keep_result=True)
    assert kept.result is not None
    assert dataclasses.replace(kept, result=None) == kept  # the table takes no part in equality
    assert "result=" not in repr(kept)


# --- AP-170: to_numpy(nulls=...) ----------------------------------------------------------

_GAPPY = {
    "n": [1, None, 3],
    "f": [math.nan, None, 2.0],
    "b": [True, None, False],
    "s": ["a", None, "c"],
}


def test_to_numpy_default_is_nan_with_a_warning():
    """The default, ``"nan"``, is the conversion `to_numpy` always made."""
    with pytest.warns(UserWarning, match="float64 with NaN"):
        out = bt.from_pydict(_GAPPY).to_numpy()
    assert out["n"].dtype == np.float64 and math.isnan(out["n"][1])
    assert out["s"].tolist() == ["a", None, "c"]


def test_to_numpy_raise_names_the_column():
    with pytest.raises(bt.PlanError, match="column 'n' holds a null"):
        bt.from_pydict(_GAPPY).to_numpy(nulls="raise")
    clean = bt.from_pydict({"x": [1, 2]}).to_numpy(nulls="raise")
    assert clean["x"].tolist() == [1, 2]


def test_to_numpy_mask_keeps_dtypes_and_tells_nan_from_null():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the mask is the information the warning says is lost
        out = bt.from_pydict(_GAPPY).to_numpy(nulls="mask")
    assert out["n"].dtype == np.int64 and out["n"].mask.tolist() == [False, True, False]
    assert out["b"].dtype == np.bool_ and out["b"].mask.tolist() == [False, True, False]
    assert math.isnan(out["f"].data[0]) and out["f"].mask.tolist() == [False, True, False]
    assert out["s"].mask.tolist() == [False, True, False]


def test_to_numpy_mask_over_many_batches_and_empty():
    ds = bt.from_pydict({"x": [None if i % 3 == 0 else i for i in range(40_000)]})
    out = ds.to_numpy(nulls="mask")["x"]
    assert len(out) == 40_000 and int(out.mask.sum()) == len(range(0, 40_000, 3))
    empty = ds.filter(bt.col("x") > 10**9).to_numpy(nulls="mask")["x"]
    assert len(empty) == 0


def test_to_numpy_rejects_an_unknown_mode():
    with pytest.raises(bt.PlanError, match="nulls"):
        bt.from_pydict({"x": [1]}).to_numpy(nulls="zero")


# --- AP-171: to_pandas(dtype_backend=...) -------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy_nullable", "pyarrow"])
def test_to_pandas_backends_keep_large_integers_exact(backend):
    pytest.importorskip("pandas")
    frame = bt.from_pydict({"x": [2**62 + 1, None]}).to_pandas(dtype_backend=backend)
    assert frame["x"].iloc[0] == 2**62 + 1
    assert frame["x"].isna().tolist() == [False, True]


def test_to_pandas_default_is_unchanged():
    pytest.importorskip("pandas")
    frame = bt.from_pydict({"x": [1, None]}).to_pandas()
    assert str(frame["x"].dtype) == "float64"


def test_to_pandas_rejects_an_unknown_backend_before_running():
    with pytest.raises(bt.PlanError, match="dtype_backend"):
        bt.from_pydict({"x": [1]}).to_pandas(dtype_backend="arrow")


# --- AP-172: np.asarray(ds, copy=False) ---------------------------------------------------


def test_asarray_copy_false_raises_as_numpy_requires():
    ds = bt.from_pydict({"x": [1, 2]})
    with pytest.raises(ValueError, match="copy=False"):
        np.asarray(ds, copy=False)
    assert np.asarray(ds).tolist() == [[1], [2]]
    assert np.asarray(ds, copy=True).tolist() == [[1], [2]]


# --- AP-173/174: show and the notebook repr -----------------------------------------------


def test_show_writes_to_file_and_leaves_stdout_alone(capsys):
    buf = io.StringIO()
    bt.from_pydict({"x": [1, 2]}).show(file=buf)
    assert "[2 rows x 1 column]" in buf.getvalue()
    assert capsys.readouterr().out == ""


def test_show_footer_tells_an_exact_limit_from_a_longer_result():
    """A result of exactly `limit` rows is complete; only a longer one is "first N"."""
    exact, longer = io.StringIO(), io.StringIO()
    bt.from_pydict({"x": [1, 2, 3]}).show(3, file=exact)
    bt.from_pydict({"x": [1, 2, 3, 4]}).show(3, file=longer)
    assert exact.getvalue().splitlines()[-1] == "[3 rows x 1 column]"
    assert longer.getvalue().splitlines()[-1] == "[first 3 rows x 1 column]"
    assert longer.getvalue().count("| 4 ") == 0


def test_show_widths_are_configurable():
    ds = bt.from_pydict({f"c{i}": ["v" * 50] for i in range(6)})
    narrow, default = io.StringIO(), io.StringIO()
    ds.show(max_width=40, max_cell_width=8, file=narrow)
    ds.show(file=default)
    assert all(len(line) <= 40 for line in narrow.getvalue().splitlines()[:-1])
    assert "vvvvv..." in narrow.getvalue()
    assert "(3 not shown)" in narrow.getvalue()
    assert "v" * 29 + "..." in default.getvalue()  # the default cell width is unchanged
    with pytest.raises(bt.PlanError):
        ds.show(max_cell_width=2)


def test_notebook_repr_caps_its_columns():
    wide = bt.from_pydict({f"c{i}": [1] for i in range(120)})
    html = wide._repr_html_()
    assert html.count("<th>") == 51
    assert "70 more columns" in html
    assert "(lazy, 120 columns" in html
    narrow = bt.from_pydict({"a": [1]})._repr_html_()
    assert "more columns" not in narrow


# --- AP-179: equals(rtol=, atol=, check_dtypes=) ------------------------------------------


def test_equals_default_stays_exact():
    a, b = bt.from_pydict({"x": [0.1 + 0.2]}), bt.from_pydict({"x": [0.3]})
    assert not a.equals(b)
    assert a.equals(b, rtol=1e-12)
    assert a.equals(b, atol=1e-12)
    assert not a.equals(b, atol=1e-20)


def test_equals_tolerance_lines_up_unordered_rows_by_exact_columns():
    a = bt.from_pydict({"k": [1, 2], "x": [0.1 + 0.2, 5.0]})
    b = bt.from_pydict({"k": [2, 1], "x": [5.0, 0.3]})
    assert a.equals(b, rtol=1e-12)
    assert not a.equals(b, rtol=1e-12, ordered=True)


def test_equals_tolerance_requires_matching_nulls_and_treats_nan_as_equal():
    a = bt.from_pydict({"x": [math.nan, None, 1.0]})
    assert a.equals(a, rtol=1e-9)
    b = bt.from_pydict({"x": [math.nan, 2.0, 1.0]})
    assert not a.equals(b, atol=10.0)


def test_equals_check_dtypes():
    ints, floats = bt.from_pydict({"x": [1, 2]}), bt.from_pydict({"x": [1.0, 2.0]})
    assert not ints.equals(floats)
    assert ints.equals(floats, check_dtypes=False)
    assert not ints.equals(bt.from_pydict({"x": [1.5, 2.0]}), check_dtypes=False)
    with pytest.raises(bt.PlanError):
        ints.equals(floats, rtol=-1.0)


# --- AP-167: closing an iter_batches iterator ---------------------------------------------


@pytest.fixture
def twenty_files(tmp_path):
    for i in range(20):
        pq.write_table(
            pa.table({"x": list(range(i * 1000, i * 1000 + 1000))}), tmp_path / f"{i:02d}.parquet"
        )
    return str(tmp_path)


def _settled(values: list[int]) -> int:
    seen = -1
    deadline = time.monotonic() + 5
    while len(values) != seen and time.monotonic() < deadline:
        seen = len(values)
        time.sleep(0.3)
    return len(values)


@pytest.mark.parametrize(
    "options",
    [{}, {"batch_size": 100}, {"prefetch_batches": 2}],
    ids=["plain", "rebatched", "prefetch"],
)
def test_closing_the_iterator_stops_the_source_read(twenty_files, options):
    """Closing after one batch stops reading: far fewer than the 20 files are processed."""
    calls: list[int] = []

    def seen(batch):
        calls.append(batch.num_rows)
        return batch

    batches = bt.read(twenty_files).map_batches(seen).iter_batches(**options)
    next(batches)
    batches.close()
    assert _settled(calls) < 20
    with contextlib.closing(bt.read(twenty_files).iter_batches(**options)) as again:
        first = next(again)
    assert first.num_rows > 0
