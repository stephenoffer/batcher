import batcher as bt

df = bt.from_pylist([{"g": "a", "v": 1}, {"g": "b", "v": 2}, {"g": "a", "v": 3}])
out = df.group_by("g").agg(
    bt.sum(bt.col("v")).alias("total"),
    bt.col("v").mean().alias("average"),
    bt.count().alias("rows"),
    bt.max(bt.col("v")).alias("highest"),
)
result = out.sort("g", descending=False, nulls_first=True).limit(10).to_pylist()
