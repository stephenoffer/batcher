"""A dominant sort key must not pin its rows to one reducer.

A range partition keeps equal keys together, because the result is the ordered
concatenation of the buckets. So a value holding share `f` of the rows puts `f·N` of them
on a single reducer *however wide the shuffle is* — and that is not a slow query, it is a
query that has stopped scaling. Measured over 600,000 rows with 40% on one key, the busiest
bucket held ~244,000 rows at 4, 8, 16, 32 and 64 buckets alike, while the even share fell
from 150,000 to 9,375: an overload of 1.6x rising to 26x purely by adding workers.

`plan_hot_split` gives that value a bucket of its own and spreads it across `subs` physical
buckets, one per contiguous run of mappers. It is sound because every row in those buckets
*ties* on the key, so their order is free — subject to the one constraint that makes a
limited sort's answer match single-node, which is that concatenating the sub-buckets in
order must reproduce mapper order.

These pin both halves: that the split levels the buckets (and keeps levelling them as the
cluster grows), and that it declines rather than risk a wrong answer.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from batcher.dist.executors.partition_io import (
    merge_boundaries,
    sample_key_grid,
    sample_probs,
)
from batcher.dist.executors.partition_io.ranges import (
    hot_key_share,
    hot_sub_bucket,
    isolate_hot_value,
    plan_hot_split,
    split_hot_bucket,
)

pytestmark = pytest.mark.unit


def _skewed(n_parts=24, per=25_000, share=0.40, hot=777, seed=3):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_parts):
        k = rng.integers(0, 5_000, per).astype("int64")
        k[rng.random(per) < share] = hot
        parts.append(k)
    return parts


def _grids(parts, probs):
    return [(list(np.quantile(p, probs)), len(p)) for p in parts]


# --- detection -----------------------------------------------------------------------


def test_the_dominant_value_and_its_share_are_read_off_the_sample():
    parts = _skewed()
    grids = _grids(parts, sample_probs(16, len(parts)))
    value, share = hot_key_share(grids)
    assert value == 777
    assert share == pytest.approx(0.40, abs=0.03)


def test_a_uniform_key_reports_no_meaningful_share():
    rng = np.random.default_rng(1)
    parts = [rng.integers(0, 100_000, 20_000) for _ in range(8)]
    got = hot_key_share(_grids(parts, sample_probs(8, 8)))
    assert got is None or got[1] < 0.01


@pytest.mark.parametrize(
    ("grid", "hot"),
    [(["a", "b", "b", "b"], "b"), ([b"\x01", b"\x02", b"\x02", b"\x02"], b"\x02")],
    ids=["text", "binary"],
)
def test_a_lexical_grid_reports_its_dominant_value(grid, hot):
    """A text or binary key is counted like any other, and this test used to assert the
    opposite.

    It read: "a string key has no cheap successor to isolate it with, so the split is not
    offered for one." That reason was wrong. A byte key's immediate successor is the value
    with a `\x00` appended — *exact*, where the float path's `nextafter` is only the nearest
    representable — so the split was being withheld from the key families most likely to be
    skewed, and a skewed string sort simply stopped scaling with the cluster.
    """
    value, share = hot_key_share([(grid, 100)])
    assert value == hot
    assert share == pytest.approx(0.75)


def test_an_empty_sample_is_declined():
    assert hot_key_share([]) is None
    assert hot_key_share([([], 0)]) is None


# --- isolation -----------------------------------------------------------------------


@pytest.mark.parametrize("hot", [0.0, 777.0, -5.0, 1e9])
def test_isolating_a_value_gives_it_a_bucket_holding_only_it(hot):
    """`bucketize` sends a key to `#{b : b <= key}`, so isolating `hot` needs `hot` itself
    and the next representable value above it."""
    boundaries, bucket = isolate_hot_value([-100.0, 500.0, 2000.0], hot)
    edges = np.asarray(boundaries)
    probe = np.array([hot, np.nextafter(hot, np.inf), np.nextafter(hot, -np.inf)])
    idx = np.searchsorted(edges, probe, side="right")
    assert idx[0] == bucket, "the hot value lands in its own bucket"
    assert idx[1] != bucket, "the next value above does not"
    assert idx[2] != bucket, "nor the one below"


@pytest.mark.parametrize(
    ("boundaries", "hot", "successor"),
    [
        ([b"\x00", b"mmm", b"\xff"], b"kkk", b"kkk\x00"),
        ([b"\x00", b"mmm", b"\xff"], b"mmm", b"mmm\x00"),
        (["a", "m", "z"], "k", "k\x00"),
        (["a", "m", "z"], "", "\x00"),
    ],
    ids=["binary", "binary-on-a-boundary", "text", "empty-text"],
)
def test_isolating_a_lexical_value_gives_it_a_bucket_holding_only_it(boundaries, hot, successor):
    """The byte successor claim, stated as the routing property it has to produce.

    Nothing sorts between `hot` and `hot + \x00`: a value above `hot` either has `hot` as a
    proper prefix, so its next byte is at least `\x00` and it is at or above the successor, or
    it differs inside `hot`'s own bytes and is above both. So `hot` gets a bucket to itself.
    """
    widened, bucket = isolate_hot_value(list(boundaries), hot)
    assert widened == sorted(set(widened)), "boundaries stay ascending and deduplicated"
    assert successor in widened, "the immediate successor is what closes the bucket"

    def route(value):
        return sum(1 for b in widened if b <= value)

    assert route(hot) == bucket, "the hot value lands in its own bucket"
    assert route(successor) != bucket, "the next value above does not"
    assert route(hot + b"\x00\x00" if isinstance(hot, bytes) else hot + "\x00\x00") != bucket
    # And the value immediately below, where there is one to name: dropping the last byte of
    # a value gives a strict prefix, which sorts before it.
    if hot:
        assert route(hot[:-1]) != bucket, "nor the one below"


def test_isolation_keeps_the_boundaries_ascending_and_deduplicated():
    boundaries, _ = isolate_hot_value([1.0, 5.0, 5.0, 9.0], 5.0)
    assert boundaries == sorted(set(boundaries))


def test_isolating_a_value_already_on_a_boundary_adds_only_its_successor():
    before = [1.0, 5.0, 9.0]
    after, _ = isolate_hot_value(before, 5.0)
    assert len(after) == len(before) + 1


# --- sub-bucket assignment -----------------------------------------------------------


@pytest.mark.parametrize("subs", [2, 3, 8])
def test_sub_buckets_are_contiguous_and_ascending_in_mapper_order(subs):
    """Concatenating the sub-buckets in order must reproduce mapper order — that is what
    makes a limited sort return single-node's rows. Round-robin would interleave them."""
    got = [hot_sub_bucket(m, 24, subs, False) for m in range(24)]
    assert got == sorted(got)
    assert set(got) == set(range(subs))


@pytest.mark.parametrize("subs", [2, 3, 8])
def test_a_descending_layout_would_be_reversed(subs):
    """Kept because the arithmetic is the part that is settled: the driver reads the buckets
    high to low while the engine's sort keeps ties in input order either way (see below), so
    mapper 0 belongs in the sub-bucket read first. `plan_hot_split` still refuses a
    descending sort — the Flight reduce disagreed with the unsplit shuffle even with this
    layout, and an unexplained reordering is not something to ship."""
    got = [hot_sub_bucket(m, 24, subs, True) for m in range(24)]
    assert got == sorted(got, reverse=True)
    assert set(got) == set(range(subs))


@pytest.mark.parametrize("nulls_first", [False, True])
def test_a_descending_sort_is_not_split(nulls_first):
    """The stated limitation, pinned so it cannot be quietly re-enabled. A skewed descending
    sort keeps the unsplit partition and pays the imbalance — time, never an answer."""
    parts = _skewed()
    grids = _grids(parts, sample_probs(16, len(parts)))
    boundaries = merge_boundaries(grids, 16)
    assert plan_hot_split(grids, boundaries, 16, nulls_first, True) is None
    assert plan_hot_split(grids, boundaries, 16, nulls_first, False) is not None


def test_the_engine_sorts_ties_stably_in_both_directions():
    """The fact the reversal above rests on, checked against the engine rather than assumed —
    getting it backwards is invisible to any assertion on keys or on the row multiset."""
    import pyarrow as pa

    import batcher as bt

    t = pa.table(
        {"k": pa.array([5] * 5, type=pa.int64()), "p": pa.array(range(5), type=pa.int64())}
    )
    ds = bt.from_arrow(t)
    assert ds.sort("k").collect().column("p").to_pylist() == [0, 1, 2, 3, 4]
    assert ds.sort("k", descending=True).collect().column("p").to_pylist() == [0, 1, 2, 3, 4]


def test_no_mapper_is_assigned_past_the_last_sub_bucket():
    for m in range(64):
        assert 0 <= hot_sub_bucket(m, 64, 5, False) < 5


def test_splitting_moves_only_the_hot_bucket_and_shifts_the_rest():
    parts = [["a"], ["b"], ["HOT"], ["c"], ["d"]]
    out = split_hot_bucket(parts, hot_bucket=2, subs=3, sub=1)
    assert out == [["a"], ["b"], [], ["HOT"], [], ["c"], ["d"]]


def test_every_mapper_contributes_its_hot_rows_exactly_once():
    """Across the fleet the hot bucket's rows must appear once and only once — a split that
    dropped or duplicated a mapper's share would change the relation."""
    subs, mappers = 4, 12
    seen = [0] * (3 + subs - 1)
    for m in range(mappers):
        sub = hot_sub_bucket(m, mappers, subs, False)
        out = split_hot_bucket([[], [f"hot{m}"], []], 1, subs, sub)
        for i, b in enumerate(out):
            seen[i] += len(b)
    assert sum(seen) == mappers


# --- the planner ---------------------------------------------------------------------


def test_a_skewed_key_is_planned_for_a_split():
    parts = _skewed()
    grids = _grids(parts, sample_probs(16, len(parts)))
    plan = plan_hot_split(grids, merge_boundaries(grids, 16), 16, False, False)
    assert plan is not None
    _boundaries, logical, hot_bucket, subs = plan
    assert logical >= 16
    assert 0 < hot_bucket < logical - 1
    assert 2 <= subs <= 64


def test_a_key_that_barely_overloads_is_left_alone():
    """Below 2x the mean the imbalance is cheaper than the extra buckets, so the split has
    to decline — a shuffle must not pay for a skew it does not have."""
    parts = _skewed(share=0.05)
    grids = _grids(parts, sample_probs(8, len(parts)))
    assert plan_hot_split(grids, merge_boundaries(grids, 8), 8, False, False) is None


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_first", [False, True])
def test_the_hot_bucket_is_never_the_one_the_nulls_ride(descending, nulls_first):
    """Nulls go to whichever end the caller's concatenation puts first, and splitting that
    bucket would scatter them through the result. Isolation is what makes this safe rather
    than lucky: inserting both `hot` and its successor as boundaries leaves a bucket on each
    side of the hot one, so it can be neither end — checked here for every combination,
    because the guard that would otherwise catch it is unreachable and so unprovable."""
    parts = _skewed()
    grids = _grids(parts, sample_probs(16, len(parts)))
    plan = plan_hot_split(grids, merge_boundaries(grids, 16), 16, nulls_first, descending)
    if descending:  # not split at all; the layout question below does not arise
        assert plan is None
        return
    assert plan is not None
    _boundaries, logical, hot_bucket, _subs = plan
    front = logical - 1 if descending else 0
    null_bucket = front if nulls_first else (logical - 1 - front)
    assert hot_bucket != null_bucket
    assert 0 < hot_bucket < logical - 1


# --- the scaling claim, measured -----------------------------------------------------


def _busiest(parts, keys, workers, split):
    probs = sample_probs(workers, workers)
    grids = _grids(np.array_split(keys, workers), probs)
    boundaries = merge_boundaries(grids, workers)
    plan = plan_hot_split(grids, boundaries, workers, False, False) if split else None
    if plan is None:
        edges = np.asarray(boundaries, dtype=float)
        return np.bincount(np.searchsorted(edges, keys, side="right"), minlength=workers).max()
    boundaries, logical, hot, subs = plan
    edges = np.asarray(boundaries, dtype=float)
    counts = np.zeros(logical + subs - 1, dtype=np.int64)
    for m, p in enumerate(parts):
        idx = np.searchsorted(edges, p, side="right")
        sub = hot_sub_bucket(m, len(parts), subs, False)
        phys = np.where(idx < hot, idx, np.where(idx == hot, hot + sub, idx + subs - 1))
        counts += np.bincount(phys, minlength=len(counts))
    return counts.max()


def test_the_busiest_bucket_shrinks_as_the_cluster_grows():
    """The whole point, stated as the thing that was false before: doubling the workers must
    roughly halve the busiest bucket. Without the split it does not move at all, so the sort
    reduce is pinned and speedup is capped at `1/share` however many nodes are added."""
    parts = _skewed()
    keys = np.concatenate(parts)

    unsplit = [_busiest(parts, keys, w, False) for w in (8, 16, 32, 64)]
    assert max(unsplit) / min(unsplit) < 1.05, "the unsplit bucket was supposed to be flat"

    split = [_busiest(parts, keys, w, True) for w in (8, 16, 32, 64)]
    for a, b in itertools.pairwise(split):
        assert b < a * 0.75, f"doubling the workers must shrink the busiest bucket: {split}"
    assert split[-1] < unsplit[-1] / 10


@pytest.mark.parametrize("kind", ["binary", "text"])
def test_a_skewed_lexical_key_keeps_scaling_as_the_cluster_grows(kind):
    """The scaling property, for the key families that were denied it.

    Without the split the busiest bucket does not move as the shuffle widens — 40% of the rows
    tie, and equal keys cannot be separated — so the reduce is pinned and the speedup from more
    nodes is capped at `1/share` however many are added. With it, the busiest bucket must track
    the even share down.
    """
    import pyarrow as pa

    from batcher.dist.executors.partition_io.ranges import bucketize

    rows, share = 60_000, 0.40
    hot = b"\x7f" * 8 if kind == "binary" else "z" * 8
    cold = [
        (i.to_bytes(8, "big") if kind == "binary" else f"{i:08d}")
        for i in range(rows - int(rows * share))
    ]
    values = [hot] * int(rows * share) + cold
    values = [values[(i * 7919) % rows] for i in range(rows)]
    dtype = pa.binary(8) if kind == "binary" else pa.string()
    batch = pa.table({"k": pa.array(values, type=dtype)}).to_batches()[0]

    busiest = []
    for buckets in (8, 16, 32):
        grids = [(sample_key_grid([batch], "k", sample_probs(buckets, 1)), rows)]
        bounds = merge_boundaries(grids, buckets)
        split = plan_hot_split(grids, bounds, buckets, False, False)
        assert split is not None, f"a {share:.0%} share over {buckets} buckets must be split"
        widened, logical, hot_bucket, subs = split
        sizes = [
            sum(b.num_rows for b in part)
            for part in bucketize([batch], "k", widened, logical, False, False)
        ]
        # The hot bucket is spread over `subs` physical sub-buckets, one per run of mappers.
        cold_max = max(sizes[:hot_bucket] + sizes[hot_bucket + 1 :], default=0)
        busiest.append(max(cold_max, sizes[hot_bucket] // subs))

    for a, b in itertools.pairwise(busiest):
        assert b < a * 0.75, f"doubling the buckets must shrink the busiest one: {busiest}"
    assert busiest[-1] < rows / 32 * 1.2, "and it must land near the even share"


def test_the_split_declines_on_a_key_that_carries_nan():
    """Isolation re-sorts the boundary list to place the hot value, and NaN has no total
    order — every comparison against it is false, so `sorted` returns an arrangement that
    depends on input order rather than on values. `bucketize` would then route by
    `searchsorted` against a sequence it does not assume, and rows land in the wrong bucket:
    right keys, right order, some ties carrying the wrong payload. Caught by the differential
    suite only when other tests ran first, because a learned grid is what put the NaN in the
    boundaries."""
    parts = _skewed()
    grids = _grids(parts, sample_probs(16, len(parts)))
    boundaries = merge_boundaries(grids, 16)
    assert plan_hot_split(grids, boundaries, 16, False, False) is not None

    with_nan = [*boundaries, float("nan")]
    assert plan_hot_split(grids, with_nan, 16, False, False) is None


def test_isolation_is_never_asked_to_order_a_nan():
    """The decline above is what keeps `isolate_hot_value` inside its precondition, so this
    records the precondition rather than the symptom: given only finite boundaries it returns
    a strictly increasing list, which is what `searchsorted` requires."""
    parts = _skewed()
    grids = _grids(parts, sample_probs(16, len(parts)))
    plan = plan_hot_split(grids, merge_boundaries(grids, 16), 16, False, False)
    assert plan is not None
    boundaries, _logical, _hot, _subs = plan
    assert all(b < c for b, c in itertools.pairwise(boundaries))
