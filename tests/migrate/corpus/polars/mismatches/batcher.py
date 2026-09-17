import batcher as bt

df = bt.from_pydict({"g": ["a", "a", "b"], "v": [1, None, 3]})
distinct = df.select(bt.col("v").count_distinct(count_nulls=True))
running = df.with_columns(bt.col("v").cum_sum(reverse=False, propagate_nulls=True).alias("running"))
# batcher-migrate: Polars `Expr.top_k` differs in Batcher (`Expr.top_k`): Polars pads a group with fewer than k non-null values with its nulls; Batcher returns the values it has (DuckDB max(x, k))
largest = df.select(bt.col("v").top_k(2))
# batcher-migrate: Polars `Expr.hash` differs in Batcher (`Expr.hash`): Polars documents its hash as unstable across versions, so no Batcher algorithm reproduces it; recompute hashes on both sides
hashed = df.select(bt.col("v").hash())
# batcher-migrate: Polars `Expr.rolling_mean` differs in Batcher (`Expr.rolling_mean`): Polars defaults min_samples to the window size and returns null until the window fills; Batcher emits partial windows. Param: min_samples=
# batcher-migrate: Polars `Expr.rolling_sum` differs in Batcher (`Expr.rolling_sum`): Polars defaults min_samples to the window size and returns null until the window fills; Batcher emits partial windows. Param: min_samples=
smoothed = df.with_columns(bt.col("v").rolling_mean(2).alias("mean2"), bt.col("v").rolling_sum(2).alias("sum2"))
shape = df.shape
print(distinct, running, largest, hashed, smoothed, shape)
