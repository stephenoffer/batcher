import polars as pl

lf = pl.from_dict({"k": ["x", "y", "x"], "n": [3, 1, 2]})
# batcher-migrate: Polars `Dataset.sort` has no exact Polars spelling; left as written
out = lf.with_row_index("row_nr", offset=0).filter(pl.col("n") < 3).sort("n", descending=False, nulls_first=True).cache()
result = out.to_dicts()
