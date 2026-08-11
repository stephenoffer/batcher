"""Two less common formats: Lance for vectors, MessagePack for interchange.

Lance is a columnar format built for random access and vector search, so it is the one to
reach for under an embedding index. MessagePack is a compact binary JSON, useful when
something on the other side speaks it and nothing else.

    python examples/io/lance_and_msgpack.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    part = tpch("part").select("p_partkey", "p_name", "p_retailprice").head(5_000)
    expected = part.sort("p_partkey").to_pydict()

    with tempfile.TemporaryDirectory() as directory:
        lance_path = str(Path(directory) / "part.lance")
        part.write.lance(lance_path)
        from_lance = bt.read.lance(lance_path)
        print("lance:", from_lance.count(), from_lance.columns)
        assert from_lance.count() == part.count()
        assert from_lance.sort("p_partkey").to_pydict()["p_name"] == expected["p_name"]

        pack_path = str(Path(directory) / "part.msgpack")
        part.write.msgpack(pack_path)
        from_pack = bt.read.msgpack(pack_path)
        print("msgpack:", from_pack.count(), from_pack.columns)
        assert from_pack.count() == part.count()
        assert from_pack.sort("p_partkey").to_pydict()["p_name"] == expected["p_name"]


if __name__ == "__main__":
    main()
