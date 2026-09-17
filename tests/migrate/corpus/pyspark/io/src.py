from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("io").config("spark.sql.shuffle.partitions", "8").getOrCreate()
events = spark.read.parquet("s3://bucket/events/")
raw = spark.read.option("header", True).option("inferSchema", "true").csv("s3://bucket/raw.csv")
strings = spark.read.csv("s3://bucket/raw.csv", header=True)
events.write.mode("overwrite").partitionBy("day").parquet("s3://bucket/by_day/")
events.write.parquet("s3://bucket/copy/")
events.write.json("s3://bucket/json/", mode="append")
print(raw.columns, strings.columns)
