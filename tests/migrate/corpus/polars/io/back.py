import polars as pl

events = pl.read_parquet("events/*.parquet")
recent = events.filter(pl.col("day") >= 20260101).cache()
# batcher-migrate: Polars `DataFrame.write_parquet` was rewritten to `Dataset.write.parquet`, which lacks: compression_level=, statistics=, row_group_size=, data_page_size=, metadata=
# batcher-migrate: Polars `Dataset.write.parquet` has no exact Polars spelling; left as written
recent.write.parquet("recent.parquet", compression="zstd")
raw = pl.read_csv("raw.csv")
print(raw.columns)
