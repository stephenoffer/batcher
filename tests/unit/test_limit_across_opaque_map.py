"""`limit` must never be pushed past a map whose row count it cannot predict.

This is a live wrong-answer bug upstream (ray-project/ray#36295, recorded in the field
guides): the optimizer pushes a `limit` *above* a map on the assumption that the map
preserves row count, and when the map filters or expands rows the pushed limit caps the
read at the wrong place — silently returning the wrong rows.

Batcher's `limit_extra` lists this as unsound by construction and does not implement it,
alongside pushing below filter/unnest/sample/distinct/aggregate. These tests pin that
reasoning against regression, using the guides' own check: compare against the result
computed in full and sliced afterwards.

The comparisons are deliberately **order-sensitive** (`==` on lists, not `assert_same`):
`limit` is a prefix operation, so an order-independent check cannot see the bug at all.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import batcher as bt

pytestmark = pytest.mark.unit

_N = 100


def _source() -> bt.Dataset:
    return bt.from_pydict({"x": list(range(_N))})


def _keep_even(batch: pa.RecordBatch) -> pa.RecordBatch:
    """A map that DROPS rows — the limit must not be counted before it."""
    return batch.filter(pc.equal(pc.bit_wise_and(batch.column("x"), pa.scalar(1)), 0))


def _triple(batch: pa.RecordBatch) -> dict:
    """A map that EXPANDS rows — one in, three out."""
    return {"x": [v for v in batch.column("x").to_pylist() for _ in range(3)]}


@pytest.mark.parametrize("n", [1, 5, 7, 40])
def test_limit_after_a_row_dropping_map_matches_the_full_result(n: int) -> None:
    piped = _source().map_batches(_keep_even, output_columns=["x"]).limit(n).to_pydict()["x"]
    full = _source().map_batches(_keep_even, output_columns=["x"]).to_pydict()["x"]
    assert piped == full[:n]


@pytest.mark.parametrize("n", [1, 7, 50])
def test_limit_after_a_row_expanding_map_matches_the_full_result(n: int) -> None:
    piped = _source().map_batches(_triple, output_columns=["x"]).limit(n).to_pydict()["x"]
    full = _source().map_batches(_triple, output_columns=["x"]).to_pydict()["x"]
    assert piped == full[:n]
    assert len(piped) == n


def test_a_dropping_map_yields_fewer_rows_than_the_limit_when_it_must() -> None:
    """The failure the pushdown causes is over-counting: capping the *read* at n leaves
    fewer than n rows after the filter. Asking for more than survive proves the count is
    taken after the map."""
    kept = _source().map_batches(_keep_even, output_columns=["x"]).limit(_N).to_pydict()["x"]
    assert kept == list(range(0, _N, 2))


def test_the_pushdown_is_still_sound_across_a_row_preserving_map() -> None:
    """The guard must not cost the case that *is* safe: a 1:1 map still limits correctly."""
    out = _source().map_batches(lambda b: b, output_columns=["x"]).limit(4).to_pydict()["x"]
    assert out == [0, 1, 2, 3]
