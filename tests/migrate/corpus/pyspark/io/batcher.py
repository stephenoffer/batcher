import batcher as bt

# batcher-migrate: PySpark `SparkSession.Builder.getOrCreate`: Spark session config not carried over: "spark.sql.shuffle.partitions"
events = bt.read.parquet("s3://bucket/events/")
raw = bt.read.csv("s3://bucket/raw.csv")
# batcher-migrate: PySpark `DataFrameReader.csv` needs a manual rewrite: Spark defaults to header=false and inferSchema=false (string columns named _c0, _c1, ...); Batcher reads the header and infers types. Codemod passes the header and inference options explicitly
strings = bt.read.csv("s3://bucket/raw.csv", header=True)
events.write.parquet("s3://bucket/by_day/", mode="overwrite", partition_by=["day"])
events.write.parquet("s3://bucket/copy/", mode="error")
events.write.json("s3://bucket/json/", mode="append")
print(raw.columns, strings.columns)
