import daft

events = daft.read_parquet("s3://bucket/events/*.parquet")
# batcher-migrate: Daft `Dataset.filter` has no exact Daft spelling; left as written
recent = events.filter(daft.col("day") >= 20260101)
# batcher-migrate: Daft `DataFrame.write_parquet` differs in Batcher (`Dataset.write.parquet`): Daft write_parquet defaults to write_mode='append' and returns a DataFrame of written paths; Batcher ds.write.parquet defaults to mode='overwrite', a file sink rejects 'append', and it returns a WriteManifest. Param: mode='append' on file sinks; Daft compresses with snappy by default, Batcher with zstd
# batcher-migrate: Daft `Dataset.write_parquet` has no exact Daft spelling; left as written
recent.write_parquet("s3://bucket/recent/")
raw = daft.read_csv("raw.csv")
print(raw.columns)
