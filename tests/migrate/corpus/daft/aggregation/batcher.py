import batcher as bt

df = bt.from_pydict({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
out = df.group_by("g").agg(
    bt.col("v").sum().alias("total"),
    bt.col("v").mean().alias("average"),
    bt.col("v").min().alias("lowest"),
)
result = out.sort("g", descending=False, nulls_first=False).to_pylist()
