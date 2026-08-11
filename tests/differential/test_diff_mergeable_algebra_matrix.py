"""`combine_finalize(partition(partial(pₖ)))` == single-node, for every aggregate.

The mergeable-algebra invariant is what "single-node == distributed" rests on: an aggregate
that has no correct `partial → combine → finalize` form works perfectly on one machine, passes
every local test, and returns *wrong numbers* at cluster scale rather than raising.

CI installs no Ray, so the Ray-backed matrix never runs there — the repo contract says as much.
This file covers the same invariant without a cluster: `repartition(n)` drives the very same
partial/combine/finalize primitives the distributed executor composes, so a broken merge shows
up here, in the PR gate, instead of on someone's cluster.

The keys are chosen to break naive grouping rather than to be representative: a null key, `0.0`
and `-0.0` (which must fold into *one* group), `NaN` (likewise), `inf`, and a key column whose
nulls are interleaved. The values carry nulls at coprime strides so no group is all-null by
accident and none is null-free either.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt

pytestmark = pytest.mark.differential

_N = 600
#: One entry per residue class, so every adversarial key lands in the fixture.
_KEY_BY_RESIDUE = {
    0: None, 1: 0.0, 2: -0.0, 3: float("nan"), 4: 1.0, 5: 2.0, 6: float("inf"), 7: -1.0,
}  # fmt: skip


def _table() -> pa.Table:
    return pa.table(
        {
            "k": pa.array([_KEY_BY_RESIDUE[i % 8] for i in range(_N)], type=pa.float64()),
            "g": pa.array([f"s{i % 5}" if i % 11 else None for i in range(_N)]),
            "x": pa.array([None if i % 13 == 0 else float((i * 37) % 101) - 50 for i in range(_N)]),
            "i": pa.array(
                [None if i % 17 == 0 else (i * 7) % 1000 - 500 for i in range(_N)],
                type=pa.int64(),
            ),
            "b": pa.array([None if i % 19 == 0 else (i % 3 == 0) for i in range(_N)]),
            "s": pa.array([None if i % 23 == 0 else f"v{i % 7}" for i in range(_N)]),
        }
    )


def _aggregates():
    """Every aggregate with a mergeable form, keyed by name for a readable failure."""
    c = bt.col
    return {
        "sum": c("x").sum(), "mean": c("x").mean(), "min": c("x").min(), "max": c("x").max(),
        "count": c("x").count(), "n_unique": c("x").n_unique(),
        "std": c("x").std(), "var": c("x").var(),
        "median": c("x").median(), "quantile": c("x").quantile(0.9),
        "product": c("i").product(), "isum": c("i").sum(), "kahan_sum": c("x").kahan_sum(),
        "skewness": c("x").skewness(), "kurtosis": c("x").kurtosis(), "mad": c("x").mad(),
        "entropy": c("s").entropy(), "mode": c("s").mode(), "any_value": c("s").any_value(),
        "bool_and": c("b").bool_and(), "bool_or": c("b").bool_or(),
        "bit_and": c("i").bit_and(), "bit_or": c("i").bit_or(), "bit_xor": c("i").bit_xor(),
        "arg_min": c("s").arg_min(c("x")), "arg_max": c("s").arg_max(c("x")),
        "approx_n_unique": c("s").approx_n_unique(), "approx_median": c("x").approx_median(),
    }  # fmt: skip


def _same(a, b) -> bool:
    """Equal, tolerating float reassociation — the one difference partitioning may cause.

    `combine` is associative in exact arithmetic but IEEE addition is not, so the partition
    count moves the last bits. Neumaier compensation and Chan's parallel Welford bound that
    error near the last bits; they do not remove it, and nothing can while the partition
    count is free. A relative tolerance is the honest comparison — an exact one would be a
    test nobody could keep green, and asserting bit-identity would claim something the
    engine does not promise.
    """
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if x is None or y is None:
            if x is not None or y is not None:
                return False
        elif isinstance(x, float) and isinstance(y, float):
            if math.isnan(x) != math.isnan(y):
                return False
            if not math.isnan(x) and abs(x - y) > 1e-6 * max(1.0, abs(y)):
                return False
        elif x != y:
            return False
    return True


@pytest.mark.parametrize("key", ["k", "g"], ids=["float_key", "string_key"])
@pytest.mark.parametrize("name", sorted(_aggregates()))
def test_an_aggregate_merges_identically_across_partitions(key, name):
    ds = bt.from_arrow(_table())
    agg = _aggregates()[name]
    one = ds.group_by(key).agg(v=agg).sort(key).to_pydict()
    many = ds.repartition(16).group_by(key).agg(v=agg).sort(key).to_pydict()
    assert list(one) == list(many), f"{name}: column names differ"
    assert _same(one[key], many[key]), f"{name}: group keys differ\n{one[key]}\n{many[key]}"
    assert _same(one["v"], many["v"]), f"{name}: values differ\n{one['v']}\n{many['v']}"


@pytest.mark.parametrize("partitions", [2, 3, 7, 16, 64])
def test_the_partition_count_never_changes_the_groups(partitions):
    """The grouping itself, isolated from any aggregate: same keys, same counts, any split.

    This is where a key-identity bug shows: `0.0` and `-0.0` must be one group and every NaN
    must be one group, in every partitioning, or a shuffle would split one group in two.
    """
    ds = bt.from_arrow(_table())
    one = ds.group_by("k").agg(n=bt.count()).sort("k").to_pydict()
    many = ds.repartition(partitions).group_by("k").agg(n=bt.count()).sort("k").to_pydict()
    assert _same(one["k"], many["k"])
    assert one["n"] == many["n"]
    # The fixture must actually contain the collapsing keys, or this proves nothing.
    assert len(one["k"]) == 7, f"expected 0.0/-0.0 and the NaNs to fold: {one['k']}"


def test_a_global_aggregate_merges_too():
    """No group keys at all — the shape whose `combine` has nothing to key on."""
    ds = bt.from_arrow(_table())
    for name, agg in _aggregates().items():
        one = ds.agg(v=agg).to_pydict()["v"]
        many = ds.repartition(16).agg(v=agg).to_pydict()["v"]
        assert _same(one, many), f"{name}: {one} vs {many}"
