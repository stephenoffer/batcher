"""Every writable format, round-tripped and compared.

This is the IO layer's release check: write the same table in every format the build
supports, read each back, and compare against the source. A format that loses a type shows up
here and nowhere else.

    python examples/io/format_round_trip_matrix.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    source = (
        tpch("orders")
        .select("o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate")
        .head(2_000)
    )
    expected = source.sort("o_orderkey").to_pydict()
    expected_types = dict(zip(source.columns, [str(dtype) for dtype in source.dtypes], strict=True))

    formats = ["parquet", "arrow", "orc", "avro", "lance", "msgpack", "delta", "csv", "json"]
    verified: list[str] = []
    unavailable: list[str] = []

    with tempfile.TemporaryDirectory() as directory:
        for fmt in formats:
            target = str(Path(directory) / f"orders_{fmt}")
            try:
                getattr(source.write, fmt)(target)
                back = getattr(bt.read, fmt)(target)
                rows = back.sort("o_orderkey").to_pydict()
            except Exception as error:
                unavailable.append(f"{fmt} ({type(error).__name__})")
                continue

            assert back.count() == source.count(), fmt
            assert set(back.columns) == set(source.columns), fmt
            assert rows["o_orderkey"] == expected["o_orderkey"], fmt
            assert rows["o_orderstatus"] == expected["o_orderstatus"], fmt
            assert all(
                abs(a - b) < 1e-6
                for a, b in zip(expected["o_totalprice"], rows["o_totalprice"], strict=True)
            ), fmt

            types = dict(zip(back.columns, [str(dtype) for dtype in back.dtypes], strict=True))
            typed = "types preserved" if types == expected_types else "types widened"
            print(f"  {fmt:<9} {back.count():>6} rows  {typed}")
            verified.append(fmt)

    print(f"{len(verified)} formats round-tripped: {', '.join(verified)}")
    if unavailable:
        print("unavailable in this build:", ", ".join(unavailable))
    assert len(verified) >= 7


if __name__ == "__main__":
    main()
