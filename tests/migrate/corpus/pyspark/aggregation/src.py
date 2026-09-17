from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([("a", 1), ("b", 2), ("a", 3)], ["g", "v"])
out = df.groupBy("g").agg(
    F.sum("v").alias("total"),
    F.avg("v").alias("average"),
    F.count("*").alias("rows"),
    F.max(F.col("v")).alias("highest"),
)
result = out.orderBy("g").take(10)
