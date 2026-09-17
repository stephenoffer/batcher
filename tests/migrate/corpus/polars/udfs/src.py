import polars as pl

df = pl.DataFrame({"x": [1, 2, 3]})
squared = df.with_columns(pl.col("x").map_elements(lambda v: v * v, return_dtype=pl.Int64).alias("sq"))
batched = df.lazy().map_batches(lambda frame: frame.with_columns(pl.col("x") + 1))
print(squared, batched)
