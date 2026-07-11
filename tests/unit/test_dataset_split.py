"""`ds.ml.train_test_split` / `ds.ml.random_split` produce reproducible disjoint parts.

The split is a pure desugaring: one reproducible uniform per row (`with_random`, keyed
on the stable row index) sliced at the cumulative fraction boundaries. The properties
that matter for an ML workflow are that the parts are **disjoint**, **cover every row**,
and are **stable** across runs and seeds — a leaky or non-reproducible split silently
corrupts every downstream metric, so they are pinned here rather than the exact sizes
(which are binomial around ``fraction * n``).
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError

pytestmark = pytest.mark.unit

_N = 2000


def _ids(ds) -> set[int]:
    return set(ds.to_pydict()["value"])


@pytest.fixture
def ds():
    return bt.range(0, _N)


def test_train_test_split_is_disjoint_and_covering(ds):
    train, test = ds.ml.train_test_split(0.2, seed=42)
    tr, te = _ids(train), _ids(test)
    assert not (tr & te), "train and test overlap — the split leaks"
    assert len(tr | te) == _N, "some rows landed in neither part"


def test_train_test_split_is_reproducible_for_a_seed(ds):
    first, _ = ds.ml.train_test_split(0.2, seed=42)
    second, _ = ds.ml.train_test_split(0.2, seed=42)
    assert _ids(first) == _ids(second)


def test_train_test_split_varies_with_the_seed(ds):
    a, _ = ds.ml.train_test_split(0.2, seed=42)
    b, _ = ds.ml.train_test_split(0.2, seed=43)
    assert _ids(a) != _ids(b)


def test_train_test_split_respects_the_requested_proportion(ds):
    _, test = ds.ml.train_test_split(0.2, seed=42)
    # Binomial(_N, 0.2): ~5 sigma is ±90 rows, so this is a stable bound, not a flake.
    assert 0.2 * _N - 90 < test.count() < 0.2 * _N + 90


def test_train_test_split_preserves_the_schema(ds):
    train, test = ds.ml.train_test_split(0.2, seed=1)
    assert train.columns == ds.columns
    assert test.columns == ds.columns


def test_split_parts_stay_lazy(ds):
    """Splitting builds plans; nothing executes until a terminal op."""
    train, _ = ds.ml.train_test_split(0.2, seed=1)
    assert isinstance(train, bt.Dataset)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.5])
def test_train_test_split_rejects_out_of_range_test_size(ds, test_size):
    with pytest.raises(PlanError, match=r"test_size must be in \(0, 1\)"):
        ds.ml.train_test_split(test_size)


def test_random_split_three_way_is_disjoint_and_covering(ds):
    train, val, test = ds.ml.random_split([0.7, 0.15, 0.15], seed=7)
    tr, va, te = _ids(train), _ids(val), _ids(test)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == _N


def test_random_split_is_reproducible(ds):
    a = [_ids(p) for p in ds.ml.random_split([0.5, 0.5], seed=3)]
    b = [_ids(p) for p in ds.ml.random_split([0.5, 0.5], seed=3)]
    assert a == b


def test_random_split_rejects_fractions_that_do_not_sum_to_one(ds):
    with pytest.raises(PlanError, match=r"must sum to 1\.0"):
        ds.ml.random_split([0.5, 0.6])


def test_random_split_rejects_non_positive_fractions(ds):
    with pytest.raises(PlanError, match="must be > 0"):
        ds.ml.random_split([1.5, -0.5])


def test_random_split_rejects_empty_fractions(ds):
    with pytest.raises(PlanError, match="non-empty"):
        ds.ml.random_split([])


def test_split_drops_its_internal_random_column(ds):
    """The synthetic uniform never reaches the output schema."""
    for part in ds.ml.random_split([0.5, 0.5], seed=1):
        assert part.columns == ["value"]
