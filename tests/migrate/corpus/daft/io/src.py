import daft

events = daft.read_parquet("s3://bucket/events/*.parquet")
recent = events.where(daft.col("day") >= 20260101)
recent.write_parquet("s3://bucket/recent/")
raw = daft.read_csv("raw.csv")
print(raw.column_names)
