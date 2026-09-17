import batcher as bt
import polars as pl

df = pl.from_dict({"x": [1, 5, 8], "s": ["apple", "banana", "cherry"]})
# batcher-migrate: Polars `bt.when` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.then` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.otherwise` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.str.substr` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.str.regexp_matches` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Expr.str.contains` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.with_columns` has no exact Polars spelling; left as written
out = df.with_columns(bt.when(pl.col("x") > 2).then(pl.lit("big")).otherwise(pl.lit("small")).alias("size"), pl.col("s").str.to_uppercase().alias("upper"), pl.col("s").str.substr(2, 3).alias("middle"), pl.col("s").str.regexp_matches("an+").alias("has_an"), pl.col("s").str.contains("rr").alias("has_rr"), (pl.col("x") + 1).alias("next"))
# batcher-migrate: Polars `Dataset.sort` has no exact Polars spelling; left as written
result = out.sort("x", descending=False, nulls_first=True).to_dicts()
