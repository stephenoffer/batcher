import polars as pl

df = pl.from_dict({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
# batcher-migrate: Polars `Expr.max` differs in Batcher (`Expr.max`): Polars max() ignores NaN where Batcher orders NaN above every number. Port as col.max(nan_policy=ignore); the codemod keeps a marker until over() windows the composed form this parameter builds
# batcher-migrate: Polars `Dataset.group_by` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.max` has no exact Polars spelling; left as written
out = df.group_by("g", maintain_order=False).agg(
    pl.col("v").min().alias("lowest"),
    pl.col("v").max().alias("highest"),
    pl.col("v").mean().alias("average"),
    pl.len().alias("rows"),
)
# batcher-migrate: Polars `Dataset.sort` has no exact Polars spelling; left as written
result = out.sort("g", descending=False, nulls_first=True).to_dicts()
