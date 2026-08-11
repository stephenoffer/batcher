"""What happens when the device tier cannot run part of a plan.

The device tier is a *translator*: it converts the plan onto cuDF operator by operator, and
anything outside the translated subset is declined and runs on the CPU engine instead. A
decline costs a fallback; approximating would cost a wrong answer, so it never approximates.

That contract is what makes `backend="gpu"` documented as always safe, and it is the thing
to verify rather than assume.

    python examples/gpu/backend_fallback.py
    python examples/gpu/backend_fallback.py --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import resolve_device, tpch
from batcher import col


def main() -> None:
    device = resolve_device()
    print("device:", device)

    lineitem = tpch("lineitem").select(
        "l_orderkey", "l_shipmode", "l_comment", "l_quantity", "l_extendedprice"
    )

    # A plan the device tier handles: numeric filters and a grouped sum.
    simple = (
        lineitem.filter(col("l_quantity") > 5)
        .group_by("l_shipmode")
        .agg(revenue=col("l_extendedprice").sum())
        .sort("l_shipmode")
    )

    # A plan with a string function that the tier may decline. Either way the answer is
    # the same; only the tier that produced it differs.
    complex_plan = (
        lineitem.filter(col("l_comment").str.contains("final"))
        .with_columns(token=col("l_comment").str.extract(r"(\w+)"))
        .group_by("token")
        .agg(lines=bt.count())
        # A total order before the limit. Sorting on `lines` alone leaves ties free to
        # fall either way, so two runs of the *same* plan could return different rows —
        # and the comparison below would fail on data that is in fact identical.
        .sort("lines", "token", descending=[True, False])
        .limit(5)
    )

    for name, plan in (("simple", simple), ("with string work", complex_plan)):
        on_device = plan.collect(backend=device)
        on_cpu = plan.collect(backend="cpu")
        print(f"{name}: {on_device.num_rows} rows")

        # The contract: same schema, same rows, whichever tier ran it.
        assert on_device.schema == on_cpu.schema
        assert on_device.num_rows == on_cpu.num_rows
        assert on_device.to_pydict() == on_cpu.to_pydict()


if __name__ == "__main__":
    main()
