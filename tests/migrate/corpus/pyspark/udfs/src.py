import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([(1,), (2,)], ["x"])
plus_one = F.udf(lambda v: v + 1, IntegerType())


@F.pandas_udf("long")
def times_two(s: pd.Series) -> pd.Series:
    return s * 2


out = df.withColumn("y", plus_one(F.col("x"))).withColumn("z", times_two(F.col("x")))
out.show()
