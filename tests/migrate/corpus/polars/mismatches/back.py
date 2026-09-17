import polars as pl

df = pl.from_dict({"g": ["a", "a", "b"], "v": [1, None, 3]})
# batcher-migrate: Polars `Expr.count_distinct` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.select` has no exact Polars spelling; left as written
distinct = df.select(pl.col("v").count_distinct(count_nulls=True))
# batcher-migrate: Polars `Expr.cum_sum` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.with_columns` has no exact Polars spelling; left as written
running = df.with_columns(pl.col("v").cum_sum(reverse=False, propagate_nulls=True).alias("running"))
# batcher-migrate: Polars `Expr.top_k` differs in Batcher (`Expr.top_k`): Polars pads a group with fewer than k non-null values with its nulls; Batcher returns the values it has (DuckDB max(x, k))
# batcher-migrate: Polars `Expr.top_k` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.select` has no exact Polars spelling; left as written
largest = df.select(pl.col("v").top_k(2))
# batcher-migrate: Polars `Expr.hash` differs in Batcher (`Expr.hash`): Polars documents its hash as unstable across versions, so no Batcher algorithm reproduces it; recompute hashes on both sides
# batcher-migrate: Polars `Expr.hash` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.select` has no exact Polars spelling; left as written
hashed = df.select(pl.col("v").hash())
# batcher-migrate: Polars `Expr.rolling_mean` differs in Batcher (`Expr.rolling_mean`): Polars defaults min_samples to the window size and returns null until the window fills; Batcher emits partial windows. Param: min_samples=
# batcher-migrate: Polars `Expr.rolling_sum` differs in Batcher (`Expr.rolling_sum`): Polars defaults min_samples to the window size and returns null until the window fills; Batcher emits partial windows. Param: min_samples=
# batcher-migrate: Polars `Expr.rolling_mean` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.rolling_sum` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.with_columns` has no exact Polars spelling; left as written
smoothed = df.with_columns(pl.col("v").rolling_mean(2).alias("mean2"), pl.col("v").rolling_sum(2).alias("sum2"))
shape = df.shape
print(distinct, running, largest, hashed, smoothed, shape)
