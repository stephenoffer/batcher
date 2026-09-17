import ray.data

events = ray.data.read_parquet("s3://bucket/events/")
ids = ray.data.range(100)
events.write_parquet("s3://bucket/copy/")
print(ids.count())
