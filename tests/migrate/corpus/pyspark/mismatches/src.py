from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([("a b", [1, 2]), ("c", [])], ["s", "xs"])
joined = df.select(F.concat(F.col("s"), F.lit("!")).alias("loud"))
parts = df.select(F.split("s", " ").alias("parts"))
exploded = df.select(F.explode("xs").alias("x"))
computed = df.selectExpr("size(xs) as n")
small = F.broadcast(df)
rdd = df.rdd
other = spark.createDataFrame([("a b", 1)], ["s", "n"])
paired = df.join(other, df.s == other.s, "inner")
print(joined, parts, exploded, computed, small, rdd, paired)
