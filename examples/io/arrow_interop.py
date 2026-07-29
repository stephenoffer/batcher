"""Moving data in and out of other frameworks, zero-copy where possible.

Arrow is the shared contract, so ``from_arrow``/``to_arrow`` are the cheapest boundary
there is. The pandas and Polars bridges go through Arrow too, which is why they are much
cheaper than a row-by-row conversion.

    python examples/io/arrow_interop.py
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import col


def main() -> None:
    # In from an Arrow table -- no copy of the buffers.
    table = pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})
    ds = bt.from_arrow(table)
    assert ds.count() == 3
    assert ds.schema.names == ["id", "v"]

    # Out to Arrow, after a transformation.
    out = ds.filter(col("id") > 1).to_arrow()
    print(out)
    assert isinstance(out, pa.Table)
    assert out.num_rows == 2

    # Python-native structures, for small results only: these *do* build Python objects.
    assert ds.to_pydict()["id"] == [1, 2, 3]
    assert ds.to_pylist()[0] == {"id": 1, "v": "a"}
    assert ds.to_dicts()[0]["v"] == "a"

    # A single scalar, when the query returns exactly one.
    total = ds.select(t=col("id").sum())
    assert total.item() == 6

    # NumPy, for a numeric block.
    arr = ds.select("id").to_numpy()
    print("numpy shape:", getattr(arr, "shape", None))
    assert arr is not None

    # pandas and Polars, if installed. Both cross through Arrow.
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("pandas not installed, skipping")
    else:
        pdf = ds.to_pandas()
        assert len(pdf) == 3
        assert bt.from_pandas(pdf).count() == 3

    try:
        import polars  # noqa: F401
    except ImportError:
        print("polars not installed, skipping")
    else:
        pl_df = ds.to_polars()
        assert pl_df.height == 3
        assert bt.from_polars(pl_df).count() == 3

    # Batch-wise iteration hands out Arrow batches, which is the streaming boundary.
    batches = list(ds.iter_batches(batch_size=2))
    print("batches:", [b.num_rows for b in batches])
    assert sum(b.num_rows for b in batches) == 3
    assert all(isinstance(b, pa.RecordBatch) for b in batches)

    # `equals` compares results rather than plans, and ignores row order by default.
    reordered = ds.sort("id", descending=True)
    assert reordered.equals(ds)


if __name__ == "__main__":
    main()
