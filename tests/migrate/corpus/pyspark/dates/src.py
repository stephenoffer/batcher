from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("s3://bucket/events/")
out = (
    df.withColumn("day", F.date_format(F.col("ts"), "yyyy-MM-dd"))
    .withColumn("parsed", F.to_date("raw", "dd/MM/yyyy"))
    .withColumn("stamp", F.date_format("ts", "yyyy-MM-dd HH:mm:ss.SSS"))
    .withColumn("year", F.year("ts"))
)
out.printSchema()
