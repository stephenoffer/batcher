"""Two ways to join across a cluster, and the size that decides between them.

Broadcasting sends the small side to every worker; shuffling moves both sides by key.
Broadcast is cheaper until the small side stops being small, and the crossover is about the
memory each worker can spare — not a fixed row count.

    python examples/dist/broadcast_versus_shuffle.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_distributed, tpch
from batcher import col


def main() -> None:
    distributed = resolve_distributed()

    lineitem = tpch("lineitem")
    orders = tpch("orders")
    nation = tpch("nation")
    customer = tpch("customer")

    # Tiny right side: a broadcast candidate.
    tiny = customer.join(nation, left_on="c_nationkey", right_on="n_nationkey").group_by(
        "n_name"
    ).agg(customers=bt.count()).sort("n_name")

    # Large right side: a shuffle.
    large = (
        lineitem.join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .group_by("o_orderpriority")
        .agg(lines=bt.count())
        .sort("o_orderpriority")
    )

    for name, query in (("broadcast-shaped", tiny), ("shuffle-shaped", large)):
        started = time.perf_counter()
        single = query.collect(distributed=False, num_partitions=1)
        single_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        many = query.collect(distributed=distributed, num_partitions=8)
        many_ms = (time.perf_counter() - started) * 1000

        print(f"{name:<18} 1 partition {single_ms:7.1f} ms   8 partitions {many_ms:7.1f} ms")

        # Whichever strategy runs, the answer is the contract.
        assert single.schema == many.schema
        assert single.num_rows == many.num_rows
        assert single.to_pydict() == many.to_pydict()

    # The right side's size is the input to the decision, and it is knowable up front.
    print("nation rows:", nation.count(), "orders rows:", orders.count())
    assert nation.count() < orders.count()
    assert col is not None


if __name__ == "__main__":
    main()
