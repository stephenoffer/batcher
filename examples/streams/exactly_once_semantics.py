"""What makes a restart safe: idempotent writes and a durable position.

Exactly-once is not a property of the sink alone. It is the pair of an idempotent write and a
position recorded with it, so a replay overwrites rather than appends. Demonstrating the
difference is a matter of running the same batch twice.

    python examples/streams/exactly_once_semantics.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch
from batcher import col


def main() -> None:
    source = tpch("orders").select("o_orderkey", "o_totalprice").sort("o_orderkey")

    with tempfile.TemporaryDirectory() as directory:
        table = str(Path(directory) / "sink")

        batch = source.head(5_000)

        # At-least-once: an append replayed twice duplicates its rows.
        batch.write.delta(table)
        batch.write.delta(table, mode="append")
        duplicated = bt.read.delta(table)
        keys = duplicated.to_pydict()["o_orderkey"]
        print(f"append twice: {duplicated.count()} rows, {len(set(keys))} distinct keys")
        assert duplicated.count() == 10_000
        assert len(set(keys)) == 5_000

        # Exactly-once: a keyed merge is idempotent, so a replay is a no-op.
        idempotent = str(Path(directory) / "sink_merge")
        batch.write.delta(idempotent)
        batch.write.delta(idempotent, merge_on="o_orderkey")
        batch.write.delta(idempotent, merge_on="o_orderkey")

        merged = bt.read.delta(idempotent)
        merged_keys = merged.to_pydict()["o_orderkey"]
        print(f"merge three times: {merged.count()} rows, {len(set(merged_keys))} distinct")
        assert merged.count() == 5_000
        assert len(set(merged_keys)) == 5_000

        # And the values are the batch's, not doubled.
        expected = batch.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        actual = merged.agg(t=col("o_totalprice").sum()).to_pydict()["t"][0]
        assert abs(actual - expected) < 1e-3

        # The position that makes a restart resumable, recorded after the write.
        position = merged.agg(m=col("o_orderkey").max()).to_pydict()["m"][0]
        print("resume position:", position)
        assert source.filter(col("o_orderkey") > position).count() > 0


if __name__ == "__main__":
    main()
