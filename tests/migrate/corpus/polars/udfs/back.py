import polars as pl

df = pl.from_dict({"x": [1, 2, 3]})
# batcher-migrate: Polars `Expr.map_elements` has no Batcher equivalent yet: Expr.map_elements, only as a batch-vectorized wrapper (per-row Python stays refused)
# batcher-migrate: Polars `Expr.map_elements` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.with_columns` has no exact Polars spelling; left as written
squared = df.with_columns(pl.col("x").map_elements(lambda v: v * v, return_dtype=pl.Int64).alias("sq"))
# batcher-migrate: Polars `LazyFrame.map_batches` differs in Batcher (`Dataset.map_batches`): Polars passes the whole DataFrame to fn once; Batcher calls fn per Arrow batch
# batcher-migrate: Polars `Dataset.map_batches` has no exact Polars spelling; left as written
batched = df.map_batches(lambda frame: frame.with_columns(pl.col("x") + 1))
print(squared, batched)
