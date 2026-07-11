"""Distributed ORDER BY a temporal key equals single-node.

A `Date`/`Timestamp` leading sort key has an order-preserving integer backing (days / ticks),
so the range-partition sample and scatter route it on that backing — a distributed
`ORDER BY <date>` now balances and sorts exactly like single-node instead of failing with
"range-partition key must be a numeric column". This is the common TPC-H shape (sort by
`l_shipdate`). String keys stay single-node (lexical order is not numeric-backed).
"""

from __future__ import annotations

import datetime

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("ray", reason="ray not installed")
pytest.importorskip("batcher._native", reason="native engine not built")


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    from conftest import init_test_ray, shutdown_test_ray

    started = init_test_ray(4)
    yield
    shutdown_test_ray(started)


def _data(n=50_000):
    rng = np.random.default_rng(11)
    base = datetime.date(2019, 1, 1)
    days = rng.integers(0, 4000, n)
    dates = pa.array([base + datetime.timedelta(days=int(d)) for d in days], pa.date32())
    secs = rng.integers(0, 10**8, n)
    epoch = datetime.datetime(2019, 1, 1)
    ts = pa.array([epoch + datetime.timedelta(seconds=int(s)) for s in secs], pa.timestamp("us"))
    return pa.table({"d": dates, "t": ts, "v": np.arange(n, dtype="int64")})


def _source():
    tbl = _data()
    return bt.from_arrow(tbl) if hasattr(bt, "from_arrow") else bt.from_pydict(tbl.to_pydict())


@pytest.mark.integration
@pytest.mark.parametrize("key", ["d", "t"])
@pytest.mark.parametrize("descending", [False, True])
def test_distributed_temporal_sort_equals_single_node(key, descending):
    ds = _source().select(key, "v").sort(key, descending=descending)
    sn = ds.collect(distributed=False).column(key).to_pylist()
    di = ds.collect(distributed=True).column(key).to_pylist()
    assert di == sn  # bit-identical order to single-node
    # and actually ordered
    if descending:
        assert all(di[i] >= di[i + 1] for i in range(len(di) - 1))
    else:
        assert all(di[i] <= di[i + 1] for i in range(len(di) - 1))
