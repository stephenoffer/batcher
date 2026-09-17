import polars as pl

df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, None, 3]})
distinct = df.select(pl.col("v").n_unique())
running = df.with_columns(pl.col("v").cum_sum().alias("running"))
largest = df.select(pl.col("v").top_k(2))
hashed = df.select(pl.col("v").hash())
smoothed = df.with_columns(pl.col("v").rolling_mean(2).alias("mean2"), pl.col("v").rolling_sum(2).alias("sum2"))
shape = df.shape
print(distinct, running, largest, hashed, smoothed, shape)
