"""Length-grouped ordering and the padding it saves.

The two claims worth pinning are opposite: the ordering must *reduce* padding, and it must not
sort the epoch. A plain sort would beat it on the first and destroy training on the second, so a
test that only checked padding would accept the wrong implementation.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import ColumnNotFoundError, PlanError
from batcher.ml import length_grouped_order, padding_waste

pytestmark = pytest.mark.unit

# Alternating short and long, which is the worst case for a naive batching order.
_LENGTHS = [1, 9, 2, 8, 3, 7, 4, 6]


def _corpus(lengths=None):
    return bt.from_pydict({"tokens": [[1] * n for n in (lengths or _LENGTHS)]})


def _lengths_of(ds):
    return [len(row) for row in ds.to_pydict()["tokens"]]


def test_padding_waste_measures_the_batch_rectangle():
    # Batches (1,9),(2,8),(3,7),(4,6): 60 padded positions, 40 real.
    assert padding_waste(_corpus(), "tokens", batch_size=2) == pytest.approx(20 / 60)


def test_a_uniform_corpus_wastes_nothing():
    assert padding_waste(_corpus([5] * 8), "tokens", batch_size=4) == 0.0


def test_a_batch_size_of_one_wastes_nothing():
    """One row per batch is its own maximum, whatever the lengths."""
    assert padding_waste(_corpus(), "tokens", batch_size=1) == 0.0


def test_the_ordering_reduces_padding():
    before = padding_waste(_corpus(), "tokens", batch_size=2)
    after = padding_waste(
        length_grouped_order(_corpus(), "tokens", batch_size=2), "tokens", batch_size=2
    )
    assert after < before


def test_the_ordering_keeps_every_row_and_no_extra_columns():
    ordered = length_grouped_order(_corpus(), "tokens", batch_size=2)
    assert ordered.columns == ["tokens"]
    assert sorted(_lengths_of(ordered)) == sorted(_LENGTHS)


def test_a_megabatch_of_one_batch_is_a_plain_shuffle():
    """The comparison point: with no window to sort in, only the shuffle survives."""
    ordered = length_grouped_order(_corpus(), "tokens", batch_size=8, megabatch_factor=1)
    assert sorted(_lengths_of(ordered)) == sorted(_LENGTHS)


def test_the_ordering_does_not_sort_the_whole_epoch():
    """A global sort would minimize padding and ruin training; the window is what prevents it."""
    lengths = list(range(1, 41))
    ordered = length_grouped_order(
        _corpus(lengths), "tokens", batch_size=2, megabatch_factor=2, seed=3
    )
    got = _lengths_of(ordered)
    assert got != sorted(got)


def test_a_larger_window_groups_lengths_more_tightly():
    lengths = [n % 20 + 1 for n in range(80)]
    narrow = padding_waste(
        length_grouped_order(_corpus(lengths), "tokens", batch_size=4, megabatch_factor=2),
        "tokens",
        batch_size=4,
    )
    wide = padding_waste(
        length_grouped_order(_corpus(lengths), "tokens", batch_size=4, megabatch_factor=20),
        "tokens",
        batch_size=4,
    )
    assert wide <= narrow


def test_the_same_seed_gives_the_same_epoch():
    a = _lengths_of(length_grouped_order(_corpus(), "tokens", batch_size=2, seed=7))
    b = _lengths_of(length_grouped_order(_corpus(), "tokens", batch_size=2, seed=7))
    assert a == b


def test_a_text_column_is_measured_in_characters():
    docs = bt.from_pydict({"body": ["a", "aaaaaaaaa", "aa", "aaaaaaaa"]})
    ordered = length_grouped_order(docs, "body", batch_size=2)
    assert sorted(len(v) for v in ordered.to_pydict()["body"]) == [1, 2, 8, 9]


def test_a_column_with_no_length_is_rejected():
    numbers = bt.from_pydict({"n": [1, 2, 3]})
    with pytest.raises(PlanError, match="no length"):
        length_grouped_order(numbers, "n", batch_size=2)


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": 2, "megabatch_factor": 0}])
def test_an_invalid_window_is_rejected(kwargs):
    with pytest.raises(PlanError):
        length_grouped_order(_corpus(), "tokens", **kwargs)


def test_an_absent_column_names_the_ones_that_exist():
    with pytest.raises(ColumnNotFoundError):
        padding_waste(_corpus(), "missing", batch_size=2)
