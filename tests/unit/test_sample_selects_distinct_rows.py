"""`sample` selects distinct rows, not rows, and these pin what that costs.

Hashing a row's values is what makes the sampler partition-independent: the same rows are
chosen on one node or a hundred. The price is that identical rows hash identically, so
they are all kept or all dropped, and the selection unit is the distinct row.

On duplicate-heavy input the realized fraction is then far from the requested one. These
tests state the current behaviour rather than endorse it -- fixing it needs a per-row
disambiguator in the hash, which costs a shuffle and changes which rows every existing
query samples. Should that land, these are the tests to rewrite, and
`Dataset.sample`/`plan.logical.Sample` are the docs to correct with them.
"""

from __future__ import annotations

import batcher as bt


def _two_valued(rows: int = 10_000) -> bt.Dataset:
    """`rows` rows holding exactly two distinct values."""
    return bt.from_pydict({"flag": ["a", "b"] * (rows // 2)})


def test_fraction_on_two_distinct_values_is_not_the_fraction() -> None:
    """The realized fraction snaps to whole duplicate groups."""
    data = _two_valued()
    kept = {fraction: data.sample(frac=fraction, seed=1).count() for fraction in (0.1, 0.5, 0.9)}

    # Every result is a whole group: nothing, one value, or both.
    assert set(kept.values()) <= {0, 5_000, 10_000}
    # And the requested fraction is not what comes back.
    assert kept[0.1] != 1_000
    assert kept[0.9] != 9_000


def test_fixed_count_on_two_distinct_values_returns_one_value() -> None:
    """`n` fills from the smallest-hash group first, so it can return one value repeated."""
    data = _two_valued()
    sampled = data.sample(n=1_000, seed=1).to_pydict()["flag"]

    assert len(sampled) == 1_000
    assert len(set(sampled)) == 1


def test_distinct_rows_sample_properly() -> None:
    """With a distinguishing column present the sampler is row-level, as intended."""
    data = bt.from_pydict({"flag": ["a", "b"] * 5_000, "key": list(range(10_000))})

    tenth = data.sample(frac=0.1, seed=1).count()
    half = data.sample(frac=0.5, seed=1).count()

    assert 800 < tenth < 1_200
    assert 4_500 < half < 5_500

    # And both values are represented, which the low-cardinality projection loses.
    sampled = data.sample(n=1_000, seed=1).to_pydict()["flag"]
    assert set(sampled) == {"a", "b"}


def test_sampling_stays_reproducible_for_a_seed() -> None:
    """Whatever it selects, it selects the same thing twice."""
    data = bt.from_pydict({"flag": ["a", "b"] * 5_000, "key": list(range(10_000))})

    first = data.sample(n=500, seed=7).sort("key").to_pydict()
    second = data.sample(n=500, seed=7).sort("key").to_pydict()
    assert first == second

    other = data.sample(n=500, seed=8).sort("key").to_pydict()
    assert other != first


def test_sampling_is_partition_independent() -> None:
    """The property the value hash exists to buy."""
    data = bt.from_pydict({"flag": ["a", "b"] * 5_000, "key": list(range(10_000))})
    sampled = data.sample(n=500, seed=3).sort("key")

    single = sampled.collect(num_partitions=1).to_pydict()
    many = sampled.collect(num_partitions=8).to_pydict()
    assert single == many
