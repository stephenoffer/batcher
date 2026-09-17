from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
orders = spark.createDataFrame([(1, 10), (2, 20), (3, 10), (4, None)], ["id", "customer"])
customers = spark.createDataFrame([(10, "Oslo"), (20, "Lima")], ["customer", "city"])
joined = orders.join(customers, on="customer", how="left_outer")
ranked = joined.orderBy(F.desc("customer"), F.col("id").asc())
both = orders.unionByName(orders)
result = ranked.take(10)
