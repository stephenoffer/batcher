import polars as pl

lf = pl.LazyFrame({"k": ["x", "y", "x"], "n": [3, 1, 2]}).lazy()
out = lf.with_row_count().filter(pl.col("n") < 3).sort("n").collect()
result = out.to_dicts()
