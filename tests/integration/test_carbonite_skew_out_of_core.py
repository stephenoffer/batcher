"""Out-of-core aggregation under skew: what grace recursion covers, and where it stops.

Grace recursion re-partitions an over-large spill bucket by a secondary hash of the group
key, so a `GROUP BY` over more distinct groups than fit in RAM completes. These tests pin
that guarantee — and pin its **ceiling**, which is a real one worth being explicit about:

A bucket dominated by a *single group* cannot be split. Every row of a group hashes
together by construction, so the fourth secondary hash produces the same bucket as the
third. `_MAX_SPILL_RECURSION` therefore stops, and such a bucket is folded in memory. That
is not a missing feature that another partitioning pass would supply — routing it to
`combine_finalize_spilling` was tried and measured identical (it grace-partitions by the
same key), and the note on `_MAX_SPILL_RECURSION` records the result. Lifting it needs a
per-group spillable state in the runtime.

**Which aggregates reach the floor at all**, and why the obvious test does not. `sum` on a
hot key is not skew as far as the spill is concerned: `partial_aggregate` collapses the hot
group to one partial row per input chunk, so its bucket stays small however many rows fed
it. The floor is only reachable when the *per-group state itself* grows with the rows —
`array_agg` retains every value, `count_distinct` every distinct one, `median` the whole
sample. A version of these tests written with `sum` passes without ever reaching the code
it means to cover, so each test here asserts which fold path ran.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
import batcher.dist.spill as spill_mod
from batcher.config import Config, MemoryConfig, config_context

pytestmark = pytest.mark.integration


@pytest.fixture
def bucket_folds(monkeypatch):
    """Record every bucket the reduce finalizes: its resident size and its budget.

    Asserting only the answer cannot distinguish a bucket that fitted from one that
    overflowed the budget and was materialized anyway — the answer is identical either
    way, which is how the un-splittable case stays invisible.
    """
    seen: list[tuple[int, int, int]] = []  # (resident, bucket_max, depth)
    real = spill_mod._reduce_agg_bucket

    def traced(store, handle, gk, aj, nat, key_idx, n_keys, out, depth):
        from batcher.config import active_config

        bucket_max = active_config().memory.spill_bucket_max_bytes
        resident = handle.logical_nbytes or handle.nbytes
        if n_keys == 0 or resident <= bucket_max or depth >= spill_mod._MAX_SPILL_RECURSION:
            seen.append((resident, bucket_max, depth))
        return real(store, handle, gk, aj, nat, key_idx, n_keys, out, depth)

    monkeypatch.setattr(spill_mod, "_reduce_agg_bucket", traced)
    return seen


def _skewed(n_hot: int, n_cold: int) -> dict[str, list]:
    """One group holding `n_hot` rows, plus `n_cold` singleton groups."""
    keys = ["hot"] * n_hot + [f"k{i}" for i in range(n_cold)]
    return {"k": keys, "v": list(range(len(keys)))}


def _tight(compression: str | None = None) -> Config:
    """A config that forces the out-of-core route with a small per-bucket budget."""
    return Config().replace(
        memory=MemoryConfig(
            max_memory_bytes=1 << 20,
            spill_bucket_max_bytes=4096,
            spill_compression=compression,
        )
    )


def _rows(table: pa.Table, value_col: str) -> dict[str, object]:
    d = table.to_pydict()
    return dict(zip(d["k"], d[value_col], strict=True))


def _over_budget(folds: list[tuple[int, int, int]]) -> int:
    return sum(1 for resident, cap, _ in folds if cap > 0 and resident > cap)


# --- the guarantee: out-of-core matches in-memory, exactly --------------------


@pytest.mark.parametrize("name", ["array_agg", "count_distinct", "median"])
def test_a_state_heavy_hot_group_still_aggregates_correctly(name, bucket_folds) -> None:
    """The skewed bucket reaches the recursion floor; the answer must still be right."""
    data = _skewed(n_hot=40_000, n_cold=500)
    expr = {
        "array_agg": bt.col("v").array_agg(),
        "count_distinct": bt.col("v").count_distinct(),
        "median": bt.col("v").median(),
    }[name]

    with config_context(_tight()):
        table = bt.from_pydict(data).group_by("k").agg(r=expr).collect()

    assert _over_budget(bucket_folds) >= 1, (
        "no bucket reached the floor over budget — this test is not covering the skew path"
    )
    assert table.num_rows == 501


@pytest.mark.parametrize("name", ["count_distinct", "median"])
def test_the_out_of_core_answer_matches_the_in_memory_one(name, bucket_folds) -> None:
    """Out-of-core is a memory strategy, not a semantics: the two must agree exactly."""
    data = _skewed(n_hot=30_000, n_cold=400)
    expr = {
        "count_distinct": bt.col("v").count_distinct(),
        "median": bt.col("v").median(),
    }[name]

    with config_context(Config()):
        in_memory = _rows(bt.from_pydict(data).group_by("k").agg(r=expr).collect(), "r")
    with config_context(_tight()):
        out_of_core = _rows(bt.from_pydict(data).group_by("k").agg(r=expr).collect(), "r")

    assert _over_budget(bucket_folds) >= 1
    assert out_of_core == in_memory


def test_the_hot_groups_own_value_is_right_not_just_the_row_count(bucket_folds) -> None:
    """A skewed group must keep its whole state, not a shard of it."""
    n_hot = 25_000
    data = _skewed(n_hot=n_hot, n_cold=200)

    with config_context(_tight()):
        table = bt.from_pydict(data).group_by("k").agg(r=bt.col("v").count_distinct()).collect()

    assert _over_budget(bucket_folds) >= 1
    got = _rows(table, "r")
    assert got["hot"] == n_hot, "the hot group lost rows across the spill"
    assert len(got) == 201


def test_several_un_splittable_groups_in_one_bucket(bucket_folds) -> None:
    keys: list[str] = []
    for hot in ("a", "b", "c"):
        keys += [hot] * 12_000
    keys += [f"c{i}" for i in range(300)]

    with config_context(_tight()):
        table = (
            bt.from_pydict({"k": keys, "v": list(range(len(keys)))})
            .group_by("k")
            .agg(r=bt.col("v").count_distinct())
            .collect()
        )

    assert _over_budget(bucket_folds) >= 1
    got = _rows(table, "r")
    assert got["a"] == got["b"] == got["c"] == 12_000
    assert len(got) == 303


def test_multiple_aggregates_survive_the_skewed_bucket(bucket_folds) -> None:
    """Every aggregate must come through the spill, not just the first."""
    n_hot = 60_000
    data = _skewed(n_hot=n_hot, n_cold=300)

    with config_context(_tight()):
        table = (
            bt.from_pydict(data)
            .group_by("k")
            .agg(
                distinct=bt.col("v").count_distinct(),
                total=bt.col("v").sum(),
                high=bt.col("v").max(),
            )
            .collect()
        )

    assert _over_budget(bucket_folds) >= 1
    d = table.to_pydict()
    rows = dict(zip(d["k"], zip(d["distinct"], d["total"], d["high"], strict=True), strict=True))
    assert rows["hot"] == (n_hot, sum(range(n_hot)), n_hot - 1)
    assert len(rows) == 301


# --- the budget is sized by what RAM pays, not by what disk holds -------------


def test_a_compressible_bucket_is_sized_by_its_resident_footprint(bucket_folds) -> None:
    """A repeated key compresses hugely; budgeting on the on-disk size under-counts it.

    `SpillHandle.logical_nbytes` is the uncompressed size the fold actually materializes.
    Using `nbytes` (the compressed file) would let a bucket many times too large for the
    budget look as though it fitted, and skip the recursion entirely.
    """
    data = _skewed(n_hot=30_000, n_cold=200)
    with config_context(_tight(compression="zstd")):
        table = bt.from_pydict(data).group_by("k").agg(r=bt.col("v").count_distinct()).collect()

    assert _over_budget(bucket_folds) >= 1, (
        "the compressed on-disk size was used as the budget, so the bucket looked small"
    )
    assert _rows(table, "r")["hot"] == 30_000


# --- the un-skewed case must not pay for any of this -------------------------


def test_an_unskewed_aggregate_never_reaches_the_floor_over_budget(bucket_folds) -> None:
    """Many small groups spill, and every bucket the recursion produces fits its budget.

    The row count is large enough that the query genuinely takes the out-of-core route; at
    20,000 rows it does not spill at all and the test would assert nothing.
    """
    n = 60_000
    data = {"k": [f"key-{i}" for i in range(n)], "v": list(range(n))}
    with config_context(_tight()):
        table = bt.from_pydict(data).group_by("k").agg(total=bt.col("v").sum()).collect()

    assert table.num_rows == n
    assert bucket_folds, "the query did not spill, so this asserts nothing about spilling"
    assert _over_budget(bucket_folds) == 0, (
        "grace recursion left a bucket over budget on data it can split"
    )
