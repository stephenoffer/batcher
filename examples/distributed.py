# examples: skip  # noqa: ERA001  (harness marker: needs the [ray] extra + a cluster)
"""Distributed execution: the same code, single-node or on a cluster.

Scaling out is a deployment change, not a rewrite. The identical pipeline runs on one
core or across Ray workers, and the result is bit-identical because every stateful
operator is built from mergeable ``partial → combine → finalize`` primitives. Ray is
used for scheduling only; bulk data moves over Arrow Flight, never the object store.

This runs against a *local* Ray cluster (Ray spins one up in-process), so it works on a
laptop. It no-ops cleanly when Ray is not installed (``pip install 'batcher[ray]'``).

    python examples/distributed.py
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pyarrow as pa

import batcher as bt
from batcher import col, count


def _rowset(table: pa.Table) -> set:
    return {tuple(r.values()) for r in table.to_pylist()}


def main() -> None:
    if importlib.util.find_spec("ray") is None:
        print("ray not installed — skipping (pip install 'batcher[ray]')")
        return

    rng = np.random.default_rng(0)
    n = 50_000
    data = pa.table(
        {
            "k": rng.integers(0, 100, n).astype("int64"),
            "v": rng.integers(0, 50, n).astype("int64"),
        }
    )

    def grouped(ds: bt.Dataset):
        return ds.group_by("k").agg(total=col("v").sum(), n=count())

    single_node = grouped(bt.from_arrow(data)).collect()
    # Same query, fanned out across workers — Ray schedules; data moves over Flight.
    distributed = grouped(bt.from_arrow(data)).collect(distributed=True, num_workers=2)

    print(f"single-node {single_node.num_rows} groups == distributed {distributed.num_rows}")
    assert _rowset(single_node) == _rowset(distributed), "distributed must equal single-node"


if __name__ == "__main__":
    main()
