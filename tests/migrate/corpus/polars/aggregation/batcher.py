import batcher as bt

df = bt.from_pydict({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
# batcher-migrate: Polars `Expr.max` differs in Batcher (`Expr.max`): Polars max() ignores NaN where Batcher orders NaN above every number. Port as col.max(nan_policy=ignore); the codemod keeps a marker until over() windows the composed form this parameter builds
out = df.group_by("g").agg(
    bt.col("v").min().alias("lowest"),
    bt.col("v").max().alias("highest"),
    bt.col("v").mean().alias("average"),
    bt.count().alias("rows"),
)
result = out.sort("g", descending=False, nulls_first=True).to_pylist()
