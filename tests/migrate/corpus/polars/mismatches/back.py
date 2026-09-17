import polars as pl

df = pl.from_dict({"g": ["a", "a", "b"], "v": [1, None, 3]})
# batcher-migrate: Polars `Expr.n_unique` differs in Batcher (`Expr.count_distinct`): Polars counts null as a distinct value where Batcher skips it. Port as col.count_distinct(count_nulls=True); the codemod keeps a marker until over() windows the composed form this parameter builds
# batcher-migrate: Polars `DataFrame.select` needs a manual rewrite: output-name inference for positional derived expressions (select(col('a') + 1) is refused today)
# batcher-migrate: Polars `Dataset.select` has no exact Polars spelling; left as written
distinct = df.select(pl.col("v").n_unique())
# batcher-migrate: Polars `Expr.cum_sum` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.with_columns` has no exact Polars spelling; left as written
running = df.with_columns(pl.col("v").cum_sum(reverse=False, propagate_nulls=True).alias("running"))
# batcher-migrate: Polars `DataFrame.select` needs a manual rewrite: output-name inference for positional derived expressions (select(col('a') + 1) is refused today)
# batcher-migrate: Polars `Dataset.select` has no exact Polars spelling; left as written
largest = df.select(pl.col("v").top_k(2))
shape = df.shape
print(distinct, running, largest, shape)
