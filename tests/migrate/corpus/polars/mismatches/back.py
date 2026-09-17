import polars as pl

df = pl.from_dict({"g": ["a", "a", "b"], "v": [1, None, 3]})
# batcher-migrate: Polars `Expr.n_unique` differs in Batcher (`Expr.count_distinct`): Polars counts null as a distinct value; Batcher skips nulls. Param: count_nulls=True
# batcher-migrate: Polars `DataFrame.select` needs a manual rewrite: output-name inference for positional derived expressions (select(col('a') + 1) is refused today)
# batcher-migrate: Polars `Dataset.select` has no exact Polars spelling; left as written
distinct = df.select(pl.col("v").n_unique())
# batcher-migrate: Polars `Expr.cum_sum` differs in Batcher (`Expr.cum_sum`): Polars leaves a null row null; Batcher carries the running value through it. Param: skip_nulls=False; also reverse=
# batcher-migrate: Polars `Expr.cum_sum` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.with_columns` has no exact Polars spelling; left as written
running = df.with_columns(pl.col("v").cum_sum().alias("running"))
# batcher-migrate: Polars `Expr.top_k` differs in Batcher (`Expr.top_k`): Polars top_k(k) returns the k largest values; Batcher's Expr.top_k returns the k most frequent values (to be renamed so top_k means largest)
# batcher-migrate: Polars `DataFrame.select` needs a manual rewrite: output-name inference for positional derived expressions (select(col('a') + 1) is refused today)
# batcher-migrate: Polars `Dataset.select` has no exact Polars spelling; left as written
largest = df.select(pl.col("v").top_k(2))
shape = df.shape
print(distinct, running, largest, shape)
