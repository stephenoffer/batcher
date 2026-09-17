import batcher as bt

ds = bt.from_items([{"s": "Hello", "n": 1}])
out = ds.with_columns(upper=bt.col("s").str.upper()).with_columns(plus=bt.col("n") + bt.lit(10))
print(out.to_pylist())
