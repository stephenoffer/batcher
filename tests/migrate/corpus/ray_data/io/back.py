import batcher as bt
import ray.data
import ray.data.expressions

events = ray.data.read_parquet("s3://bucket/events/")
# batcher-migrate: Ray Data `bt.range` has no exact Ray Data spelling; left as written
ids = bt.range(100, name="id")
# batcher-migrate: Ray Data `Dataset.write_parquet` differs in Batcher (`Dataset.write.parquet`): Ray write_parquet defaults mode=SaveMode.APPEND into a directory; Batcher defaults mode='overwrite' (replacing existing output) and supports append only for delta/iceberg/hudi/snowflake. Pass mode explicitly; file-sink append is missing.
# batcher-migrate: Ray Data `Dataset.write_parquet` has no exact Ray Data spelling; left as written
events.write_parquet("s3://bucket/copy/")
print(ids.count())
