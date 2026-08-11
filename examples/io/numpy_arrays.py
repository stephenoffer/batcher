"""Reading a NumPy array file as a Dataset.

A `.npy` file is a dense array with no column names, so it arrives as a single column of
lists rather than as one column per dimension. Splitting it into named columns is a
projection you write, which is the honest interface: the file does not carry the names.

    python examples/io/numpy_arrays.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import batcher as bt
from batcher import col


def main() -> None:
    matrix = np.arange(30, dtype=np.float64).reshape(10, 3)

    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "features.npy")
        np.save(path, matrix)

        loaded = bt.read.numpy(path)
        print(loaded.schema)
        assert loaded.count() == 10

        column = loaded.columns[0]
        # One row per array row, each holding the row's values as a list.
        widths = loaded.select(n=col(column).list.len()).to_pydict()["n"]
        assert set(widths) == {3}

        # Name the dimensions with a projection.
        named = loaded.select(
            x=col(column).list.get(0),
            y=col(column).list.get(1),
            z=col(column).list.get(2),
        )
        result = named.to_pydict()
        print(result["x"][:3], result["z"][:3])
        assert result["x"] == list(matrix[:, 0])
        assert result["z"] == list(matrix[:, 2])

        # And back out to NumPy.
        arrays = named.to_numpy()
        assert set(arrays) == {"x", "y", "z"}
        assert arrays["x"].shape == (10,)
        assert np.allclose(arrays["y"], matrix[:, 1])

        # `bt.from_numpy` is the direct route in, when the array is already in memory.
        direct = bt.from_numpy(matrix)
        assert direct.count() == 10


if __name__ == "__main__":
    main()
