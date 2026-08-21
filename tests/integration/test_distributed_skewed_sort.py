"""A distributed sort over a dominant key must return exactly what one node returns.

The range partitioner now spreads a hot value across several buckets instead of pinning it
to one (`partition_io.ranges.plan_hot_split`), which is the difference between a sort whose
reduce shrinks as workers are added and one whose busiest reducer sits at the same size
forever. That rearrangement is only sound because the rows it moves all *tie* on the sort
key — so what these check is the part the arithmetic cannot: that the relation is unchanged.

Every case is held to key order **and** the full row multiset, because a skew bug here does
not lose the keys. It returns the right keys carrying the wrong rows, which an
order-independent or key-only assertion reads as a pass.
"""

from __future__ import annotations

import collections
import dataclasses
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")

pytestmark = pytest.mark.integration

HOT = 777


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def skewed(tmp_path_factory):
    """Twelve files over a key where one value holds ~40% of the rows.

    On shared scratch when the cluster has it, because the workers read these files
    directly; a single-node box reaches `tmp_path` everywhere.
    """
    from batcher.dist.shuffle_io import shared_scratch_root

    root = shared_scratch_root()
    base = (
        os.path.join(root, f"skewed_sort_{os.getpid()}")
        if root
        else str(tmp_path_factory.mktemp("skewed"))
    )
    os.makedirs(base, exist_ok=True)
    rng = np.random.default_rng(3)
    for i in range(12):
        n = 8_000
        k = rng.integers(0, 2_000, n).astype("int64")
        k[rng.random(n) < 0.40] = HOT
        # `nk` is the same key with the hot value NULLED, so a sort on it puts the
        # dominant share in the null bucket — the one the split must refuse to touch.
        nk = pa.array([None if v == HOT else int(v) for v in k], type=pa.int64())
        # The same skew spelled as a text and a fixed-width binary key. Those families used to
        # be refused the split outright, so a skewed `ORDER BY <string>` stopped scaling with
        # the cluster; they carry the identical distribution so a difference between them and
        # `k` is the split's doing and not the data's.
        text = pa.array([f"{int(v):06d}" for v in k], type=pa.string())
        blob = pa.array([int(v).to_bytes(8, "big") for v in k], type=pa.binary(8))
        pq.write_table(
            pa.table(
                {
                    "k": k,
                    "nk": nk,
                    "tk": text,
                    "bk": blob,
                    "payload": np.arange(i * n, (i + 1) * n, dtype="int64"),
                }
            ),
            os.path.join(base, f"skew_{i}.parquet"),
        )
    return base


@pytest.fixture
def disk_shuffle():
    """Pin the disk transport, which is where the range partitioner runs."""
    from batcher.config import active_config, set_config

    cfg = active_config()
    set_config(
        dataclasses.replace(
            cfg,
            distributed=dataclasses.replace(
                cfg.distributed, transport="disk", shared_filesystem=True
            ),
        )
    )
    try:
        yield
    finally:
        set_config(cfg)


def _rows(table: pa.Table):
    return collections.Counter(tuple(sorted(r.items())) for r in table.to_pylist())


def test_the_hot_value_really_dominates(skewed):
    """The fixture has to actually be skewed, or every test below passes vacuously."""
    k = bt.read.parquet(skewed).collect().column("k").to_pylist()
    assert collections.Counter(k)[HOT] / len(k) > 0.3


@pytest.fixture
def no_split(monkeypatch):
    """Turn the hot-value split off, so a test can compare against the shuffle it replaced."""
    from batcher.dist.executors import sort as sort_mod

    monkeypatch.setattr(sort_mod, "plan_hot_split", lambda *a, **k: None)


@pytest.mark.parametrize("descending", [False, True])
def test_a_skewed_sort_returns_the_single_node_relation(skewed, disk_shuffle, descending):
    ds = bt.read.parquet(skewed).sort("k", descending=descending)
    single = ds.collect()
    dist = ds.collect(distributed=True, num_workers=8)

    keys = dist.column("k").to_pylist()
    assert keys == sorted(keys, reverse=descending)
    assert keys == single.column("k").to_pylist()
    assert _rows(single) == _rows(dist)


def test_a_descending_skewed_sort_is_not_split(skewed, disk_shuffle):
    """The stated limitation, checked end to end rather than only in the planner: a skewed
    descending sort takes the unsplit partition and pays the imbalance."""
    from batcher.dist.executors.partition_io.ranges import plan_hot_split

    seen: list = []
    ds = bt.read.parquet(skewed).sort("k", descending=True)
    single = ds.collect()
    dist = ds.collect(distributed=True, num_workers=8)
    assert dist.column("k").to_pylist() == single.column("k").to_pylist()
    assert _rows(single) == _rows(dist)
    # and the planner refuses it for every null placement
    for nulls_first in (False, True):
        seen.append(plan_hot_split([([777.0] * 60, 1000)] * 8, [0.0, 900.0], 8, nulls_first, True))
    assert seen == [None, None]


@pytest.mark.parametrize("descending", [False])
def test_splitting_the_hot_value_changes_nothing_about_the_result(
    skewed, disk_shuffle, descending, request
):
    """The claim, stated so that it can actually fail. Comparing the split run against
    *single-node* cannot: an unlimited sort over a load-balanced source already orders rows
    tied on the key differently from one node, so a full-sequence assertion is red before
    the split is even involved, and keys-plus-multiset is blind to exactly the thing the
    split can get wrong. Comparing against the **unsplit shuffle** isolates it — and it is
    what caught a descending assignment that reversed the sub-buckets one time too many,
    returning ascending runs of descending rows. Ascending only, because that is the only
    direction `plan_hot_split` splits.
    """
    ds = bt.read.parquet(skewed).sort("k", descending=descending)
    with_split = ds.collect(distributed=True, num_workers=8).to_pylist()
    request.getfixturevalue("no_split")
    without = ds.collect(distributed=True, num_workers=8).to_pylist()
    assert with_split == without


@pytest.mark.parametrize("lim", [1, 137, 5_000])
def test_a_skewed_limited_sort_selects_the_same_rows(skewed, disk_shuffle, lim):
    """The hardest case the split has to survive. A `LIMIT` inside the hot value's run cuts
    *between* the sub-buckets, so the row it stops at is decided by the order they are
    concatenated in — which is why they are laid out by mapper rather than round-robin."""
    ds = bt.read.parquet(skewed).sort("k").limit(lim)
    single = ds.collect()
    dist = ds.collect(distributed=True, num_workers=8)

    assert dist.to_pylist() == single.to_pylist()


def test_a_skewed_sort_with_nulls_keeps_them_at_the_right_end(skewed, disk_shuffle):
    """Nulls ride whichever end the concatenation puts first, and the split refuses to touch
    the bucket they land in. `nk` is null exactly where the key is hot, so the dominant share
    *is* the null bucket — the case the decline exists for."""
    ds = bt.read.parquet(skewed).sort("nk")
    single = ds.collect()
    dist = ds.collect(distributed=True, num_workers=8)
    assert dist.column("nk").to_pylist() == single.column("nk").to_pylist()
    assert _rows(single) == _rows(dist)


# --- text and binary keys: the families the split used to be withheld from ------------


@pytest.mark.parametrize("key", ["tk", "bk"], ids=["text", "binary"])
@pytest.mark.parametrize("descending", [False, True])
def test_a_skewed_lexical_sort_returns_the_single_node_relation(
    skewed, disk_shuffle, key, descending
):
    """The rearrangement is only sound because the moved rows tie, so the relation must be
    byte-identical to single-node — keys in order, and the same rows carrying them."""
    ds = bt.read.parquet(skewed).sort(key, descending=descending)
    single = ds.collect()
    dist = ds.collect(distributed=True, num_workers=8)

    keys = dist.column(key).to_pylist()
    assert keys == sorted(keys, reverse=descending)
    assert keys == single.column(key).to_pylist()
    assert _rows(dist) == _rows(single)


@pytest.mark.parametrize("key", ["tk", "bk"], ids=["text", "binary"])
def test_splitting_a_lexical_hot_value_changes_nothing_about_the_result(
    skewed, disk_shuffle, no_split, key
):
    """Split against unsplit, on the same data through the same executor.

    A skew bug here does not lose keys — it returns the right keys carrying the wrong rows, so
    this compares the full row multiset as well as the key order. `no_split` is what makes it a
    comparison rather than a restatement.
    """
    ds = bt.read.parquet(skewed).sort(key)
    unsplit = ds.collect(distributed=True, num_workers=8)
    del no_split  # the fixture is undone by monkeypatch after this test
    assert unsplit.column(key).to_pylist() == sorted(unsplit.column(key).to_pylist())
    single = ds.collect()
    assert _rows(unsplit) == _rows(single)


@pytest.mark.parametrize("key", ["tk", "bk"], ids=["text", "binary"])
def test_a_skewed_lexical_limited_sort_selects_the_same_rows(skewed, disk_shuffle, key):
    """`ORDER BY <lexical> LIMIT k` over a dominant value: the sub-buckets must concatenate in
    mapper order, or the limit keeps a different set of tied rows than one node would."""
    ds = bt.read.parquet(skewed).sort(key).limit(500)
    assert _rows(ds.collect(distributed=True, num_workers=8)) == _rows(ds.collect())
