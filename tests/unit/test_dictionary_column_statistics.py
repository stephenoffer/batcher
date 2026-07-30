"""A dictionary-encoded column must report the statistics of the relation the engine runs.

Dictionary encoding is the default for a low-cardinality column in Parquet and ORC, and it is
what a pandas categorical arrives as. Every exact statistic silently vanished on one: the
distinct count, the mean, the sum and the bounds, all four at once.

Two causes, both the shape of asking a type predicate about a *label* rather than about
values. `pa.types.is_integer(dictionary<values=int64, indices=int32>)` is `False`, so the
ordered-type gate rejected a column of perfectly ordered integers; and `count_distinct`,
`min_max`, `mean` and `sum` have no dictionary kernel, so they raised into the
"not derivable" path that exists for genuinely underivable columns.

The width was wrong in the other direction and for a different reason: the engine *decodes*
dictionary encoding at the FFI boundary, so reporting the encoded 4.0 bytes per row for a
column that runs at 9.2 under-sizes the memory envelope by 2.3x.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_ROWS = 20_000
_CARD = 12


def _source(column: pa.Array):
    return bt.from_arrow(pa.table({"c": column}))._sources[0]


def _both(values: np.ndarray):
    """The same values stored plainly and dictionary-encoded."""
    plain = pa.array(values)
    return _source(plain), _source(plain.dictionary_encode())


@pytest.fixture(params=["int64", "float64", "string"])
def pair(request):
    ints = np.arange(_ROWS) % _CARD
    values = {
        "int64": ints.astype("int64"),
        "float64": ints.astype("float64"),
        "string": np.array([f"cat_{i}" for i in ints]),
    }[request.param]
    return _both(values)


def test_the_distinct_count_survives_encoding(pair):
    plain, encoded = pair
    assert plain.column_ndv("c") == encoded.column_ndv("c") == _CARD


def test_the_mean_and_sum_survive_encoding(pair):
    plain, encoded = pair
    assert plain.column_mean("c") == encoded.column_mean("c")
    assert plain.column_sum("c") == encoded.column_sum("c")


def test_the_bounds_survive_encoding(pair):
    """The ordered-type gate read the dictionary label and rejected ordered values."""
    plain, encoded = pair
    lo, hi = plain.column_bounds("c"), encoded.column_bounds("c")
    assert (lo is None) == (hi is None)
    if lo is not None:
        assert (lo.min, lo.max) == (hi.min, hi.max)


def test_a_predicate_count_survives_encoding(pair):
    plain, encoded = pair
    value = plain.column_bounds("c").min if plain.column_bounds("c") else "cat_0"
    assert plain.column_predicate_count("eq", "c", value) == encoded.column_predicate_count(
        "eq", "c", value
    )


def test_the_width_is_the_decoded_width_not_the_encoded_one():
    """The engine decodes at the FFI boundary, so the encoded size is not what runs."""
    values = np.array([f"cat_{i % _CARD}" for i in range(_ROWS)])
    decoded_truth = pa.array(values).nbytes / _ROWS
    encoded = _source(pa.array(values).dictionary_encode())
    assert encoded.column_cheap_stat("c").avg_bytes == pytest.approx(decoded_truth, rel=0.05)
    # And it is genuinely different from the encoded size it used to report.
    assert pa.array(values).dictionary_encode().nbytes / _ROWS < decoded_truth / 2


def test_the_width_estimate_stays_cheap_in_rows():
    """It is measured off the dictionary, so growing the rows must not grow the work."""
    small = pa.array([f"cat_{i % _CARD}" for i in range(1_000)]).dictionary_encode()
    large = pa.array([f"cat_{i % _CARD}" for i in range(200_000)]).dictionary_encode()
    assert _source(small).column_cheap_stat("c").avg_bytes == pytest.approx(
        _source(large).column_cheap_stat("c").avg_bytes, rel=0.01
    )


def test_an_all_null_dictionary_has_no_bounds_but_an_exact_null_count():
    """The safety property: a column that really has no bounds must still report none.

    It still carries the exact null count, which is the behavior a string column already has
    -- the cheap fact is not discarded along with the inexact one.
    """
    from batcher.plan.stats import Provenance

    column = pa.array([None] * _ROWS, type=pa.string()).dictionary_encode()
    stat = _source(column).column_bounds("c")
    assert (stat.min, stat.max) == (None, None)
    assert stat.null_count == float(_ROWS)
    assert stat.null_count_provenance is Provenance.EXACT


def test_a_plain_column_is_unchanged():
    """Nothing about a non-dictionary column may move."""
    plain = _source(pa.array(np.arange(_ROWS) % _CARD))
    stat = plain.column_bounds("c")
    assert (stat.min, stat.max) == (0, _CARD - 1)
    assert plain.column_ndv("c") == _CARD


def test_a_query_over_a_dictionary_column_is_still_right():
    """Statistics are a sizing and planning input, so they may never change the relation."""
    values = np.array([f"cat_{i % 4}" for i in range(1_000)])
    ds = bt.from_arrow(
        pa.table({"c": pa.array(values).dictionary_encode(), "v": pa.array(np.arange(1_000))})
    )
    out = ds.group_by("c").agg(n=bt.col("v").count()).collect().to_pydict()
    assert sorted(out["n"]) == [250, 250, 250, 250]
    assert ds.filter(bt.col("c") == "cat_1").collect().num_rows == 250
