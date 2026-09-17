import polars as pl
import batcher as bt

df = bt.from_pydict({"x": [1, 2, 3]})
# batcher-migrate: Polars `Expr.map_elements` has no Batcher equivalent yet: Expr.map_elements, only as a batch-vectorized wrapper (per-row Python stays refused)
squared = df.with_columns(bt.col("x").map_elements(lambda v: v * v, return_dtype=pl.Int64).alias("sq"))
# batcher-migrate: Polars `LazyFrame.map_batches` differs in Batcher (`Dataset.map_batches`): Polars passes the whole DataFrame to fn once; Batcher calls fn per Arrow batch
batched = df.map_batches(lambda frame: frame.with_columns(pl.col("x") + 1))
print(squared, batched)
