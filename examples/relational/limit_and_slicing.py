"""Taking a piece: head, tail, limit, slice, and every-nth.

`head` and `limit` are the same operator, and both are order-dependent only if you sorted
first. Without a sort, "the first ten rows" is whichever ten the scan produced, which is
allowed to change between runs and between partition counts.

    python examples/relational/limit_and_slicing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import tpch
from batcher import col


def main() -> None:
    orders = tpch("orders").select("o_orderkey", "o_totalprice").sort("o_orderkey")

    assert orders.head(5).count() == 5
    assert orders.limit(5).count() == 5
    assert orders.head(5).to_pydict() == orders.limit(5).to_pydict()

    # `tail` is the other end of the same sorted order.
    tail = orders.tail(3).to_pydict()
    print("last keys:", tail["o_orderkey"])
    assert tail["o_orderkey"] == sorted(tail["o_orderkey"])

    # `slice` takes an offset as well as a length — this is LIMIT/OFFSET.
    page_two = orders.slice(5, 5).to_pydict()
    first_ten = orders.head(10).to_pydict()
    assert page_two["o_orderkey"] == first_ten["o_orderkey"][5:]

    # Every nth row, for a cheap deterministic thinning of a large scan.
    thinned = orders.head(20).gather_every(4).to_pydict()
    print("thinned:", thinned["o_orderkey"])
    assert (
        thinned["o_orderkey"]
        == first_ten["o_orderkey"][:1] + orders.head(20).to_pydict()["o_orderkey"][4::4]
    )

    # `first`/`last` return one row as a tuple — they leave the Dataset world, so
    # they execute immediately rather than extending the plan.
    print("first row:", orders.first(), "named:", orders.first(named=True))
    assert len(orders.first()) == len(orders.columns)
    assert orders.first(named=True)["o_orderkey"] == orders.head(1).to_pydict()["o_orderkey"][0]

    # `item` is the scalar form, and only valid on a single cell.
    biggest = orders.sort("o_totalprice", descending=True).select("o_totalprice").head(1)
    print("largest order:", biggest.item())
    assert biggest.item() == max(orders.to_pydict()["o_totalprice"])
    assert col is not None


if __name__ == "__main__":
    main()
