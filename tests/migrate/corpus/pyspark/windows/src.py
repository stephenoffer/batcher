from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([("a", 1, 10), ("a", 2, 20), ("b", 3, 30)], ["g", "t", "v"])
by_group = Window.partitionBy("g").orderBy(F.desc("t"))
trailing = Window.partitionBy("g").orderBy("t").rowsBetween(Window.unboundedPreceding, Window.currentRow)
out = df.withColumn("rn", F.row_number().over(by_group)).withColumn("running", F.sum("v").over(trailing))
result = out.orderBy("t").take(10)
