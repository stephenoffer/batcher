import batcher as bt

df = bt.from_pydict({"x": [1, 5, 8], "s": ["apple", "banana", "cherry"]})
out = (
    df.with_columns(upper=bt.col("s").str.upper())
    .with_columns(head=bt.col("s").str.slice(0, 3))
    .with_columns(next=bt.col("x") + bt.lit(1))
    .with_columns(missing=bt.col("s").is_null())
)
result = out.sort("x", descending=False, nulls_first=False).to_pylist()
