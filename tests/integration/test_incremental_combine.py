"""Bounded-memory reducer combine: incremental == flat.

A wide-shuffle reducer combines each mapper's partial into a running merged state
as it arrives, instead of collecting all W partials and combining once. Because
`combine` is associative, the incremental merge equals the flat one — so the
reducer's memory (one merged partial + one in-flight fetch) is independent of the
mapper count, which is what lets the shuffle scale to tens of thousands of workers.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count

pytest.importorskip("batcher._native", reason="native engine not built")


def _gk_aj(ds):
    agg = ds._plan
    gk = json.dumps([{"expr": k.expr.to_ir(), "alias": k.alias} for k in agg.group_keys])
    aj = json.dumps([s.agg.to_ir(s.alias) for s in agg.aggregates])
    return gk, aj


def _rows(t):
    return sorted(
        tuple(round(v, 6) if isinstance(v, float) else v for v in r.values()) for r in t.to_pylist()
    )


def test_incremental_combine_matches_flat_over_many_partials():
    import batcher._native as nat

    rng = np.random.default_rng(5)
    n = 80_000
    t = pa.table(
        {"k": rng.integers(0, 50, n).astype("int64"), "v": rng.integers(0, 100, n).astype("int64")}
    )
    ds = bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=count(), a=col("v").mean())
    gk, aj = _gk_aj(ds)

    # One partial per chunk = many mapper partials feeding one reducer.
    partials = [nat.partial_aggregate(gk, aj, [b]) for b in t.to_batches(max_chunksize=2000)]
    assert len(partials) >= 20  # genuinely wide fan-in

    flat = nat.combine_finalize(gk, aj, partials)

    # Incremental: hold only the running merged partial (never the whole list).
    running = None
    for p in partials:
        merged = [p] if running is None else [running, p]
        running = nat.combine(gk, aj, merged)
    incremental = nat.combine_finalize(gk, aj, [running])

    assert _rows(pa.Table.from_batches([flat])) == _rows(pa.Table.from_batches([incremental]))
    # And both equal the single-node aggregate.
    assert _rows(pa.Table.from_batches([incremental])) == _rows(ds.collect())
