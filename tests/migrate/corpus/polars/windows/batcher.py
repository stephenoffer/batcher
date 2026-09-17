import batcher as bt

df = bt.from_pydict({"g": ["a", "a", "b"], "t": [1, 2, 3], "v": [10, 20, 30]})
# batcher-migrate: Polars `Expr.max` differs in Batcher (`Expr.max`): Polars max() ignores NaN (nan_max propagates it); Batcher orders NaN above every number and returns NaN. Param: nan_policy='ignore'
# batcher-migrate: Polars `Expr.sum` differs in Batcher (`Expr.sum`): Polars returns 0 for an empty or all-null input; Batcher (SQL) returns null. Param: empty_value=0
out = df.with_columns(bt.col("v").max().over(partition_by=["g"]).alias("group_max"), bt.col("v").sum().over(partition_by=["g"], order_by=["t"], frame=(None, None)).alias("partition_total"))
result = out.sort("t", descending=False, nulls_first=True).to_pylist()
