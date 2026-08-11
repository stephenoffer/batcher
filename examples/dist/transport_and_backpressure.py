"""Moving bulk data between workers, and why it needs flow control.

Bulk batches move over Arrow Flight rather than through the scheduler's object store, with
credit-based backpressure: one credit is one batch slot, and a producer with no credits
blocks. Without that a fast producer simply fills a slow consumer's memory.

    python examples/dist/transport_and_backpressure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_distributed, tpch
from batcher import col
from batcher.config import get_option, option_names


def main() -> None:
    distributed = resolve_distributed()
    print("distributed:", distributed)

    # The flow-control knobs, as configuration rather than as magic constants.
    flow_options = [name for name in option_names() if "flow" in name or "credit" in name]
    print("flow-control options:", flow_options)
    for name in flow_options[:4]:
        print(f"  {name} = {get_option(name)}")

    lineitem = tpch("lineitem")

    # A shuffle-heavy query: the group key is high cardinality, so most rows move.
    shuffled = (
        lineitem.group_by("l_orderkey")
        .agg(revenue=col("l_extendedprice").sum(), lines=bt.count())
        .sort("l_orderkey")
    )

    # A broadcast-friendly one: the group key is tiny, so almost nothing moves.
    local = (
        lineitem.group_by("l_returnflag")
        .agg(revenue=col("l_extendedprice").sum())
        .sort("l_returnflag")
    )

    for name, query in (("high-cardinality", shuffled), ("low-cardinality", local)):
        single = query.collect(distributed=False, num_partitions=1)
        many = query.collect(distributed=distributed, num_partitions=8)
        print(f"{name:<20} {single.num_rows:>7} rows")

        assert single.schema == many.schema
        assert single.num_rows == many.num_rows
        left, right = single.to_pydict(), many.to_pydict()
        for column, values in left.items():
            if values and isinstance(values[0], float):
                assert all(
                    abs(a - b) <= abs(a) * 1e-12
                    for a, b in zip(values, right[column], strict=True)
                ), column
            else:
                assert values == right[column], column

    # The transport is a scheduling concern: it cannot change the answer, and the
    # cardinality of the group key is what decides how much it has to move.
    assert shuffled.count() > local.count()


if __name__ == "__main__":
    main()
