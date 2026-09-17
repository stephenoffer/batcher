from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("verbs").getOrCreate()
df = spark.createDataFrame([(1, "a", 10), (2, "b", 20), (3, "c", 30)], ["id", "name", "score"])
out = (
    df.filter(F.col("score") > 10)
    .withColumn("points", F.col("score") * 2)
    .withColumnRenamed("name", "label")
    .select("id", "label", "points")
    .distinct()
    .orderBy("id")
)
result = out.take(10)
