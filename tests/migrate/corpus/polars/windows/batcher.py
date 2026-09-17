import batcher as bt

df = bt.from_pydict({"g": ["a", "a", "b"], "t": [1, 2, 3], "v": [10, 20, 30]})
# batcher-migrate: Polars `Expr.max` differs in Batcher (`Expr.max`): Polars max() ignores NaN where Batcher orders NaN above every number. Port as col.max(nan_policy=ignore); the codemod keeps a marker until over() windows the composed form this parameter builds
# batcher-migrate: Polars `Expr.sum` differs in Batcher (`Expr.sum`): Polars returns 0 for an empty or all-null input where Batcher returns null. Port as col.sum(empty_value=0); the codemod keeps a marker until over() windows the composed form this parameter builds
out = df.with_columns(bt.col("v").max().over(partition_by=["g"]).alias("group_max"), bt.col("v").sum().over(partition_by=["g"], order_by=["t"], frame=(None, None)).alias("partition_total"))
result = out.sort("t", descending=False, nulls_first=True).to_pylist()
