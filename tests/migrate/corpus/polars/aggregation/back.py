import polars as pl

df = pl.from_dict({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
# batcher-migrate: Polars `Expr.max` differs in Batcher (`Expr.max`): Polars max() ignores NaN (nan_max propagates it); Batcher orders NaN above every number and returns NaN. Param: nan_policy='ignore'
# batcher-migrate: Polars `Dataset.group_by` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.max` has no exact Polars spelling; left as written
out = df.group_by("g").agg(
    pl.col("v").min().alias("lowest"),
    pl.col("v").max().alias("highest"),
    pl.col("v").mean().alias("average"),
    pl.len().alias("rows"),
)
# batcher-migrate: Polars `Dataset.sort` has no exact Polars spelling; left as written
result = out.sort("g", descending=False, nulls_first=True).to_dicts()
