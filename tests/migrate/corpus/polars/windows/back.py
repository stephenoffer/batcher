import polars as pl

df = pl.from_dict({"g": ["a", "a", "b"], "t": [1, 2, 3], "v": [10, 20, 30]})
# batcher-migrate: Polars `Expr.max` differs in Batcher (`Expr.max`): Polars max() ignores NaN (nan_max propagates it); Batcher orders NaN above every number and returns NaN. Param: nan_policy='ignore'
# batcher-migrate: Polars `Expr.sum` differs in Batcher (`Expr.sum`): Polars returns 0 for an empty or all-null input; Batcher (SQL) returns null. Param: empty_value=0
# batcher-migrate: Polars `Expr.max` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.over` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.sum` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.with_columns` has no exact Polars spelling; left as written
out = df.with_columns(pl.col("v").max().over(partition_by=["g"]).alias("group_max"), pl.col("v").sum().over(partition_by=["g"], order_by=["t"], frame=(None, None)).alias("partition_total"))
# batcher-migrate: Polars `Dataset.sort` has no exact Polars spelling; left as written
result = out.sort("t", descending=False, nulls_first=True).to_dicts()
