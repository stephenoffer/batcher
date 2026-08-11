"""`peak_max` / `peak_min` — local extrema, checked against Polars.

Turning points are how a series is summarized: the highs of a price trace, the spikes in a
sensor reading, the local optima of a scan. Polars spells them `peak_max`/`peak_min` and is
the oracle for every **interior** row, which is where the definition is unambiguous: a peak
is strictly beyond both neighbours, so a plateau has none.

The **edges** are asserted against Batcher's own rule instead, because the two libraries
decide them differently and deliberately. Batcher says an edge row is never a peak — it has
only one neighbour, and a peak is defined by both. Polars counts a row that beats its single
neighbour, which makes a lone row a `peak_max` but not a `peak_min`. Comparing the ends
against Polars would pin a convention Batcher does not hold, so this file compares the
interior against Polars and the ends against the documented rule.

The null handling gets its own tests because it is where the composition could go wrong
without changing a single interior value: a neighbouring null makes the comparison null, and
a null is not `False` until it is made so.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

pl = pytest.importorskip("polars")


def _batcher(values, method, dtype=None):
    table = pa.table(
        {
            "t": pa.array(range(len(values)), type=pa.int64()),
            "x": pa.array(values, type=dtype or pa.int64()),
        }
    )
    expr = getattr(bt.col("x"), method)(order_by=["t"])
    return bt.from_arrow(table).with_columns(p=expr).sort("t").to_pydict()["p"]


def _polars(values, method):
    # The dtype is explicit because an empty Python list gives Polars a `null` series,
    # which its peak kernels refuse — a fixture artefact, not a semantic difference.
    return getattr(pl.Series("x", values, dtype=pl.Int64), method)().to_list()


@pytest.mark.parametrize(
    "values",
    [
        [1, 5, 2, 8, 3],
        [1, 2, 3, 4, 5],  # monotone: no interior turning point
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1],  # a plateau is not a peak
        [1, 3, 3, 1],  # a plateau at the top is still not a peak
        [7],  # one row has no neighbours
        [],  # empty
        [2, 1, 2, 1, 2],  # alternating
        [-5, -1, -5],  # negatives
    ],
    ids=["mixed", "up", "down", "flat", "plateau_top", "single", "empty", "zigzag", "negative"],
)
@pytest.mark.parametrize("method", ["peak_max", "peak_min"])
def test_interior_local_extrema_match_polars(values, method):
    """Every row with a neighbour on both sides, where the definition is unambiguous."""
    got, want = _batcher(values, method), _polars(values, method)
    assert got[1:-1] == want[1:-1], f"{values} interior: {got} vs polars {want}"


@pytest.mark.parametrize("method", ["peak_max", "peak_min"])
def test_a_neighbouring_null_is_not_a_peak_rather_than_a_null(method):
    """The comparison against a null is null; a peak flag must be a boolean, not unknown."""
    got = _batcher([1, 5, None, 8, 3], method, dtype=pa.int64())
    assert all(isinstance(v, bool) for v in got), got
    assert got[2] is False, "a null value is not an extremum"
    assert got[1] is False and got[3] is False, "a row beside a null cannot be judged a peak"


@pytest.mark.parametrize("method", ["peak_max", "peak_min"])
def test_the_first_and_last_rows_are_never_peaks(method):
    """Batcher's edge rule, which is the one place it departs from Polars on purpose.

    An edge row has one neighbour, and "beyond both neighbours" is not a question that row
    can answer. Taking the symmetric rule keeps `peak_max` and `peak_min` mirror images and
    keeps the answer stable when a partition is split differently.
    """
    for values in ([9, 1, 1], [1, 1, 9], [9, 1, 9], [7], [1, 2]):
        got = _batcher(values, method)
        assert got[0] is False, (values, got)
        assert got[-1] is False, (values, got)
    # ...and this is where Polars says otherwise, recorded so the divergence is deliberate.
    assert _polars([7], "peak_max") == [True]
    assert _batcher([7], "peak_max") == [False]


def test_peaks_restart_at_every_partition():
    """A partition boundary is an edge, so the row beside it is not judged against the other
    partition's values."""
    table = pa.table(
        {
            "g": pa.array(["a", "a", "a", "b", "b", "b"]),
            "t": pa.array([0, 1, 2, 0, 1, 2], type=pa.int64()),
            "x": pa.array([1, 9, 1, 1, 9, 1], type=pa.int64()),
        }
    )
    got = (
        bt.from_arrow(table)
        .with_columns(p=bt.col("x").peak_max(partition_by=["g"], order_by=["t"]))
        .sort("g", "t")
        .to_pydict()["p"]
    )
    assert got == [False, True, False, False, True, False]


def test_peaks_survive_repartitioning():
    n = 300
    table = pa.table(
        {
            "g": pa.array([f"s{i % 5}" for i in range(n)]),
            "t": pa.array(list(range(n)), type=pa.int64()),
            "x": pa.array([(i * 37) % 11 for i in range(n)], type=pa.int64()),
        }
    )
    ds = bt.from_arrow(table)
    expr = bt.col("x").peak_max(partition_by=["g"], order_by=["t"])
    one = ds.with_columns(p=expr).sort("g", "t").to_pydict()["p"]
    many = ds.repartition(8).with_columns(p=expr).sort("g", "t").to_pydict()["p"]
    assert one == many
    assert any(one), "the fixture must actually contain peaks"
