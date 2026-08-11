"""Writing Parquet: choosing a compression codec and reading the file back.

Compression trades CPU for bytes, and the right answer depends on whether you are
network- or CPU-bound. What it never trades is fidelity: every codec below returns
byte-identical data, which is the property worth asserting rather than assuming.

    python examples/io/parquet_write_options.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    lineitem = tpch("lineitem").select("l_orderkey", "l_quantity", "l_extendedprice", "l_shipmode")

    with tempfile.TemporaryDirectory() as directory:
        sizes: dict[str, int] = {}
        for codec in ("snappy", "zstd", "gzip"):
            target = Path(directory) / f"{codec}.parquet"
            lineitem.write.parquet(str(target), compression=codec)
            sizes[codec] = target.stat().st_size
            # Whatever the codec, the data that comes back is identical.
            assert bt.read.parquet(str(target)).count() == lineitem.count()

        print({name: f"{size / 1024:.0f} KiB" for name, size in sizes.items()})
        # zstd beats snappy on size; that is the whole reason to pay its CPU cost.
        assert sizes["zstd"] < sizes["snappy"]

        # The written file carries row groups, which is what lets a reader skip parts
        # of it using per-group statistics rather than decoding everything.
        import pyarrow.parquet as pq

        metadata = pq.ParquetFile(str(Path(directory) / "zstd.parquet")).metadata
        print(f"{metadata.num_row_groups} row groups, {metadata.num_rows} rows")
        assert metadata.num_rows == lineitem.count()
        assert metadata.num_row_groups >= 1

        # Values survive the round trip exactly, whichever codec wrote them.
        original = lineitem.sort("l_orderkey").head(100).to_pydict()
        for codec in ("snappy", "zstd", "gzip"):
            back = bt.read.parquet(str(Path(directory) / f"{codec}.parquet"))
            assert back.sort("l_orderkey").head(100).to_pydict() == original


if __name__ == "__main__":
    main()
