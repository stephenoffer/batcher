import batcher as bt

events = bt.read.parquet("s3://bucket/events/*.parquet")
recent = events.filter(bt.col("day") >= 20260101)
# batcher-migrate: Daft `DataFrame.write_parquet` differs in Batcher (`Dataset.write.parquet`): Daft write_parquet defaults to write_mode='append' and returns a DataFrame of written paths; Batcher ds.write.parquet defaults to mode='overwrite', a file sink rejects 'append', and it returns a WriteManifest. Param: mode='append' on file sinks; Daft compresses with snappy by default, Batcher with zstd
recent.write_parquet("s3://bucket/recent/")
raw = bt.read.csv("raw.csv")
print(raw.columns)
