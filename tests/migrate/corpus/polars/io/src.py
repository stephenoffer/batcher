import polars as pl

events = pl.scan_parquet("events/*.parquet")
recent = events.filter(pl.col("day") >= 20260101).collect()
recent.write_parquet("recent.parquet", compression="zstd")
raw = pl.read_csv("raw.csv")
print(raw.columns)
