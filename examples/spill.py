"""Out-of-core execution: bounded memory via spill-to-disk.

Stateful operators (aggregation, join, sort) spill to disk when they would exceed
the memory envelope, so a query stays alive under bounded memory and the result is
identical to the in-memory run. Carbonite *decides* to spill from the configured
budget — the caller does not ask for it. Here a deliberately tiny budget forces the
out-of-core path on a small dataset so the example runs anywhere.

    python examples/spill.py
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

import batcher as bt
from batcher import col, count
from batcher.config import Config, MemoryConfig, config_context


def _rowset(table: pa.Table) -> set:
    return {tuple(r.values()) for r in table.to_pylist()}


def main() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    data = pa.table(
        {
            "k": rng.integers(0, 200, n).astype("int64"),
            "v": rng.integers(0, 100, n).astype("int64"),
        }
    )

    def grouped(ds: bt.Dataset):
        return ds.group_by("k").agg(total=col("v").sum(), n=count())

    in_memory = grouped(bt.from_arrow(data)).collect()

    # A 1-byte cap makes every sized breaker exceed the budget, so the engine forces
    # the out-of-core (partition-and-spill) path. The result must be identical.
    tiny_budget = Config().replace(memory=MemoryConfig(max_memory_bytes=1))
    with config_context(tiny_budget):
        spilled = grouped(bt.from_arrow(data)).collect()

    print(f"groups: {in_memory.num_rows} (in-memory) == {spilled.num_rows} (spilled)")
    assert _rowset(in_memory) == _rowset(spilled), "spilled result must equal in-memory"


if __name__ == "__main__":
    main()
