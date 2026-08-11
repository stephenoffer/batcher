"""Compression codecs measured on real data.

The trade is bytes against CPU, and the right point depends on where the bottleneck is.
Over object storage the bytes usually win; on a local NVMe with spare cores they often do
not. Measure on your data rather than reusing someone else's table.

    python examples/io/compression_tradeoffs.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem")
    expected = lineitem.count()

    with tempfile.TemporaryDirectory() as directory:
        measurements: dict[str, tuple[int, float, float]] = {}
        for codec in ("snappy", "zstd", "gzip"):
            target = Path(directory) / f"{codec}.parquet"

            started = time.perf_counter()
            lineitem.write.parquet(str(target), compression=codec)
            write_ms = (time.perf_counter() - started) * 1000

            started = time.perf_counter()
            rows = bt.read.parquet(str(target)).count()
            read_ms = (time.perf_counter() - started) * 1000

            assert rows == expected
            measurements[codec] = (target.stat().st_size, write_ms, read_ms)

        for codec, (size, write_ms, read_ms) in measurements.items():
            print(
                f"{codec:<8} {size / 1024:>8.0f} KiB  "
                f"write {write_ms:6.0f} ms  read {read_ms:6.0f} ms"
            )

        # zstd is smaller than snappy on this data, which is the usual ordering.
        assert measurements["zstd"][0] < measurements["snappy"][0]

        # Every codec round-trips identically — the trade is never fidelity.
        reference = lineitem.sort("l_orderkey", "l_linenumber").head(200).to_pydict()
        for codec in measurements:
            back = bt.read.parquet(str(Path(directory) / f"{codec}.parquet"))
            assert back.sort("l_orderkey", "l_linenumber").head(200).to_pydict() == reference


if __name__ == "__main__":
    main()
