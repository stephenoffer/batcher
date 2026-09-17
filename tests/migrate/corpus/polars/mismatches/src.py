import polars as pl

df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, None, 3]})
distinct = df.select(pl.col("v").n_unique())
running = df.with_columns(pl.col("v").cum_sum().alias("running"))
largest = df.select(pl.col("v").top_k(2))
shape = df.shape
print(distinct, running, largest, shape)
