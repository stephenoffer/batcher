"""`iter_batches` shapes its stream: format, ragged tail, local shuffle, look-ahead.

Each option is off by default, so the plain call is pinned unchanged beside them. The shuffle
is held to the two properties that make it a shuffle rather than a reordering bug: it is a
permutation of the input (no row lost or repeated, across a morsel boundary), and it actually
moves rows, so a no-op implementation fails here.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

N = 40_000  # past two 16,384-row morsels


def _ds(n: int = N) -> bt.Dataset:
    return bt.from_pydict({"i": list(range(n))})


def test_the_default_stream_is_unchanged():
    batches = list(_ds(10).iter_batches(4))
    assert all(isinstance(b, pa.RecordBatch) for b in batches)
    assert [b.num_rows for b in batches] == [4, 4, 2]
    assert [v for b in batches for v in b.column("i").to_pylist()] == list(range(10))


@pytest.mark.parametrize(
    ("fmt", "kind"),
    [("numpy", dict), ("pandas", "DataFrame"), ("pyarrow", pa.RecordBatch)],
)
def test_batch_format_converts_each_batch(fmt, kind):
    first = next(iter(_ds(10).iter_batches(5, batch_format=fmt)))
    if isinstance(kind, str):
        assert type(first).__name__ == kind
    else:
        assert isinstance(first, kind)
    rows = first["i"] if fmt != "pyarrow" else first.column("i")
    assert list(np.asarray(rows)) == [0, 1, 2, 3, 4]


def test_drop_last_drops_only_the_ragged_tail():
    sizes = [b.num_rows for b in _ds(10).iter_batches(4, drop_last=True)]
    assert sizes == [4, 4]


def test_drop_last_needs_a_batch_size():
    with pytest.raises(PlanError, match="drop_last"):
        next(iter(_ds().iter_batches(drop_last=True)))


def test_the_local_shuffle_is_a_permutation_that_moves_rows():
    batches = list(_ds().iter_batches(1000, local_shuffle_buffer_size=5000, local_shuffle_seed=7))
    got = [v for b in batches for v in b.column("i").to_pylist()]
    assert sorted(got) == list(range(N))  # nothing lost, nothing repeated
    assert got != list(range(N))  # and the order genuinely changed
    assert all(b.num_rows == 1000 for b in batches)  # exact width survives the shuffle


def test_the_local_shuffle_is_reproducible_by_seed_and_differs_across_seeds():
    def order(seed):
        return [
            v
            for b in _ds().iter_batches(local_shuffle_buffer_size=5000, local_shuffle_seed=seed)
            for v in b.column("i").to_pylist()
        ]

    assert order(3) == order(3)
    assert order(3) != order(4)


def test_prefetch_yields_the_same_stream():
    plain = [b.column("i").to_pylist() for b in _ds().iter_batches(3000)]
    ahead = [b.column("i").to_pylist() for b in _ds().iter_batches(3000, prefetch_batches=4)]
    assert ahead == plain


@pytest.mark.parametrize(
    "kw",
    [
        {"batch_format": "arrow"},
        {"local_shuffle_buffer_size": 0},
        {"local_shuffle_seed": -1},
        {"prefetch_batches": -1},
        {"drop_last": "yes", "batch_size": 2},
    ],
)
def test_an_invalid_option_names_itself(kw):
    name = next(k for k in kw if k != "batch_size")
    with pytest.raises(PlanError, match=name):
        next(iter(_ds().iter_batches(**kw)))
