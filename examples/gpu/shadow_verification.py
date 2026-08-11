"""Verifying a device result against the CPU engine.

The device tier is a translator, not the same code on different hardware, so it is the one
tier whose agreement with the oracle has to be checked rather than assumed. Shadow
verification is that check: run both, compare schema first, then values.

Schema first is deliberate. The two defects this tier has actually shipped were both *type*
bugs with correct values — a DATE coming back as a timestamp, and an integer `abs` widening
to double. A value-only comparison would have passed both.

    python examples/gpu/shadow_verification.py
    python examples/gpu/shadow_verification.py --device cpu
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import has_gpu, resolve_device, tpch
from batcher import col


def main() -> None:
    device = resolve_device()
    print("device:", device, "| accelerator visible:", has_gpu())

    lineitem = tpch("lineitem")

    queries = {
        "grouped sum": lineitem.group_by("l_shipmode")
        .agg(revenue=col("l_extendedprice").sum())
        .sort("l_shipmode"),
        "date projection": lineitem.select("l_orderkey", "l_shipdate").head(1_000),
        "integer abs": lineitem.select(
            "l_orderkey", magnitude=(col("l_linenumber") - 4).abs()
        ).head(1_000),
        "filtered count": lineitem.filter(col("l_quantity") > 30).agg(n=bt.count()),
    }

    for name, query in queries.items():
        on_device = query.collect(backend=device)
        on_cpu = query.collect(backend="cpu")

        # Schema first: names *and* types.
        assert on_device.schema == on_cpu.schema, f"{name}: schema differs"
        assert on_device.column_names == on_cpu.column_names, name

        # Then values.
        assert on_device.num_rows == on_cpu.num_rows, name
        assert on_device.to_pydict() == on_cpu.to_pydict(), f"{name}: values differ"
        print(f"  {name:<18} {on_device.num_rows:>6} rows, schema and values agree")

    # The integer case keeps its integer type — the widening bug in one line.
    magnitude = queries["integer abs"].collect(backend=device)
    assert "int" in str(magnitude.schema.field("magnitude").type)

    # And the date case stays a date.
    dated = queries["date projection"].collect(backend=device)
    assert "date" in str(dated.schema.field("l_shipdate").type)


if __name__ == "__main__":
    main()
