"""A distributed sort that reuses learned range boundaries equals one that samples for them.

The optimization skips the SAMPLE barrier — a full execution of the mapped prefix over every
split, done solely to collect a quantile grid — on any sort shape that has been sampled
before. What it must not change is the sorted relation, and that is what this pins: the same
query is run once to learn the grid and again to reuse it, and both are compared against the
single-node result.

The comparison is deliberately asymmetric, because a sort promises different things about
different columns. The *key* column is compared positionally: its ordering is the sort's
entire contract and is fully determined. The relation as a whole is compared as a multiset.
The payload is not compared positionally, because duplicate keys may order their payloads
differently across partitions and asserting otherwise would assert something the sort never
promised — the duplicate keys here exist precisely so that distinction is exercised.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _data(n=60_000):
    rng = np.random.default_rng(23)
    # A key domain far smaller than the row count, so duplicates are dense and the
    # tie-ordering distinction above is genuinely exercised.
    return pa.table(
        {
            "k": rng.integers(0, n // 4, n).astype("int64"),
            "v": np.arange(n, dtype="int64"),
        }
    )


def _rows(table):
    d = table.to_pydict()
    return d["k"], sorted(zip(d["k"], d["v"], strict=True))


def test_learned_boundaries_produce_the_same_sorted_relation():
    ds = bt.from_arrow(_data())
    query = ds.sort("k")

    ref_keys, ref_multi = _rows(query.collect())

    # First run samples and persists the grid; second reuses it.
    first_keys, first_multi = _rows(query.collect(distributed=True))
    second_keys, second_multi = _rows(query.collect(distributed=True))

    assert first_keys == ref_keys
    assert second_keys == ref_keys
    assert first_multi == ref_multi
    assert second_multi == ref_multi


def test_the_second_run_reuses_a_grid_rather_than_sampling_again(monkeypatch):
    """The equivalence test above would pass just as well if nothing were ever learned and
    both runs sampled, so observe the reuse directly.

    Counts what the driver got back from the store: `None` means it fell through to the
    SAMPLE barrier, a grid means it skipped it. The first distributed run must sample (there
    is nothing to reuse yet) and the second must not.
    """
    import batcher.dist.flight_sort as flight_sort

    verdicts: list[bool] = []
    real_load = flight_sort.load_learned_grids

    def watched(shape_key):
        grids = real_load(shape_key)
        verdicts.append(grids is not None)
        return grids

    monkeypatch.setattr(flight_sort, "load_learned_grids", watched)

    query = bt.from_arrow(_data(20_000)).sort("k")
    query.collect(distributed=True)
    query.collect(distributed=True)

    assert verdicts == [False, True], (
        f"expected sample-then-reuse, got {verdicts} — the grid was not persisted, "
        "or the shape key is not stable across two runs of the same query"
    )
