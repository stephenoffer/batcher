import batcher as bt
from pyspark.sql import functions as F

# batcher-migrate: PySpark `bt.from_pylist` has no exact PySpark spelling; left as written
df = bt.from_pylist([{"g": "a", "v": 1}, {"g": "b", "v": 2}, {"g": "a", "v": 3}])
# batcher-migrate: PySpark `Dataset.group_by` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `bt.count` has no exact PySpark spelling; left as written
out = df.group_by("g").agg(
    F.sum(F.col("v")).alias("total"),
    F.avg(F.col("v")).alias("average"),
    bt.count().alias("rows"),
    F.max(F.col("v")).alias("highest"),
)
# batcher-migrate: PySpark `Dataset.sort` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.to_pylist` has no exact PySpark spelling; left as written
result = out.sort("g", descending=False, nulls_first=True).limit(10).to_pylist()
