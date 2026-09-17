import batcher as bt

events = bt.read.parquet("events/*.parquet")
recent = events.filter(bt.col("day") >= 20260101).cache()
# batcher-migrate: Polars `DataFrame.write_parquet` was rewritten to `Dataset.write.parquet`, which lacks: compression_level=, statistics=, row_group_size=, data_page_size=, metadata=
recent.write.parquet("recent.parquet", compression="zstd")
raw = bt.read.csv("raw.csv")
print(raw.columns)
