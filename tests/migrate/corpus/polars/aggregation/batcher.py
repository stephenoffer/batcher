import batcher as bt

df = bt.from_pydict({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
# batcher-migrate: Polars `Expr.max` differs in Batcher (`Expr.max`): Polars max() ignores NaN (nan_max propagates it); Batcher orders NaN above every number and returns NaN. Param: nan_policy='ignore'
out = df.group_by("g").agg(
    bt.col("v").min().alias("lowest"),
    bt.col("v").max().alias("highest"),
    bt.col("v").mean().alias("average"),
    bt.count().alias("rows"),
)
result = out.sort("g", descending=False, nulls_first=True).to_pylist()
