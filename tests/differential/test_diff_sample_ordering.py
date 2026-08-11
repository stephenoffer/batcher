"""A sample of a sorted relation comes back sorted.

`Sample` is order-preserving in the engine: a fractional sample is a per-morsel filter, and
the fixed-count pass gathers its winners by ascending row index within each batch and emits
batches in order (`bc-interp::ops::reshape::sample_n_batches`). Rows are only ever dropped,
and dropping rows from a sorted relation leaves it sorted.

An optimizer rule (`eliminate_sort_before_sample`) used to delete the `Sort` beneath a
`Sample` on the grounds that the sampled *multiset* does not depend on input order. That is
true and it is not sufficient — the multiset is not the only observable — so
``ds.sort("a").sample(fraction=0.3)`` returned its rows in scan order instead of `a` order.

Nothing caught it, and the reason is the point of this file: every check on that rule used
`assert_same`, which is order-independent **by design**, so the one property that broke was
the one nothing compared. These tests assert the observed order explicitly.
"""

from __future__ import annotations

import random

import pyarrow as pa

import batcher as bt


def _shuffled(n: int = 200) -> pa.Table:
    """`n` rows whose scan order is emphatically not their sorted order."""
    rng = random.Random(7)
    values = list(range(n))
    rng.shuffle(values)
    return pa.table({"a": values, "b": list(range(n))})


def test_fractional_sample_of_a_sorted_relation_is_sorted(duck):
    t = _shuffled()
    duck.register("t", t)
    out = bt.from_arrow(t).sort("a").sample(fraction=0.3, seed=42).collect().to_pydict()["a"]
    assert out, "the sample kept no rows, so it proves nothing about order"
    assert out == sorted(out)


def test_fixed_count_sample_of_a_sorted_relation_is_sorted(duck):
    t = _shuffled()
    duck.register("t", t)
    out = bt.from_arrow(t).sort("a").sample(n=40, seed=42).collect().to_pydict()["a"]
    assert len(out) == 40
    assert out == sorted(out)


def test_descending_sample_of_a_sorted_relation_is_descending(duck):
    t = _shuffled()
    duck.register("t", t)
    out = (
        bt.from_arrow(t)
        .sort("a", descending=True)
        .sample(fraction=0.3, seed=42)
        .collect()
        .to_pydict()["a"]
    )
    assert out
    assert out == sorted(out, reverse=True)


def test_sampling_keeps_the_same_rows_whether_or_not_the_input_was_sorted(duck):
    """The multiset really is order-independent — which is what makes the sound form of
    the rewrite (dropping the sort under an *aggregate*) legitimate."""
    t = _shuffled()
    duck.register("t", t)
    sorted_first = bt.from_arrow(t).sort("a").sample(fraction=0.3, seed=42).collect()
    unsorted = bt.from_arrow(t).sample(fraction=0.3, seed=42).collect()
    assert sorted(sorted_first.to_pydict()["a"]) == sorted(unsorted.to_pydict()["a"])


def test_aggregate_over_a_sampled_sort_matches_duckdb(duck):
    """The shape whose sort *is* removable — the result must be unchanged by that."""
    t = _shuffled()
    duck.register("t", t)
    got = bt.from_arrow(t).sort("a").sample(fraction=0.3, seed=42).agg(s=bt.col("b").sum())
    direct = bt.from_arrow(t).sample(fraction=0.3, seed=42).agg(s=bt.col("b").sum())
    assert got.collect().to_pydict() == direct.collect().to_pydict()
