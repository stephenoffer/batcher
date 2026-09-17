import batcher as bt

df = bt.from_pydict({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
out = df.group_by("g", maintain_order=False).agg(
    bt.col("v").min().alias("lowest"),
    bt.col("v").max(nan_policy="ignore").alias("highest"),
    bt.col("v").mean().alias("average"),
    bt.count().alias("rows"),
)
result = out.sort("g", descending=False, nulls_first=True).to_pylist()
