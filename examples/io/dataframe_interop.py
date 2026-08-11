"""Handing data to and from pandas, Polars, NumPy and Arrow.

Every one of these goes through Arrow, so a conversion is a reinterpretation of the same
buffers rather than a copy where the layouts agree. That is why `to_arrow` is free and
`to_pandas` is not: pandas needs its own representation for strings and nulls.

    python examples/io/dataframe_interop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from _common import tpch


def main() -> None:
    nation = tpch("nation").select("n_nationkey", "n_name", "n_regionkey")

    # Arrow: the native form.
    table = nation.to_arrow()
    print(table.schema)
    assert table.num_rows == 25
    assert bt.from_arrow(table).count() == 25

    # pandas, and back.
    frame = nation.to_pandas()
    print(frame.head(3))
    assert len(frame) == 25
    assert bt.from_pandas(frame).count() == 25

    # NumPy: a dict of one array per column, not a single matrix, because columns
    # legitimately differ in dtype.
    arrays = nation.to_numpy()
    print({name: array.dtype for name, array in arrays.items()})
    assert set(arrays) == set(nation.columns)
    assert arrays["n_nationkey"].shape == (25,)

    # Polars, if it is installed. It shares Arrow buffers, so this is close to free.
    try:
        import polars  # noqa: F401
    except ImportError:
        print("polars not installed; skipping that leg")
    else:
        frame_pl = nation.to_polars()
        assert frame_pl.height == 25
        assert bt.from_polars(frame_pl).count() == 25

    # A round trip through every hop preserves the values.
    original = nation.sort("n_nationkey").to_pydict()
    round_tripped = bt.from_pandas(nation.to_pandas()).sort("n_nationkey").to_pydict()
    assert round_tripped["n_name"] == original["n_name"]
    assert round_tripped["n_nationkey"] == original["n_nationkey"]


if __name__ == "__main__":
    main()
