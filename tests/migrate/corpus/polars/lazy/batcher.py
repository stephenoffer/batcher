import batcher as bt

lf = bt.from_pydict({"k": ["x", "y", "x"], "n": [3, 1, 2]})
out = lf.with_row_index("row_nr", offset=0).filter(bt.col("n") < 3).sort("n", descending=False, nulls_first=True).cache()
result = out.to_pylist()
