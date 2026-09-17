from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([(1, "apple"), (5, "banana")], ["x", "s"])
out = (
    df.withColumn("size", F.when(F.col("x") > 2, "big").otherwise("small"))
    .withColumn("upper", F.upper("s"))
    .withColumn("head", F.substring("s", 0, 3))
    .withColumn("next", F.col("x") + F.lit(1))
    .withColumn("same", df.x)
)
out.show()
