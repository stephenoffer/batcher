import batcher as bt
from pyspark.sql import functions as F

# batcher-migrate: PySpark `bt.from_pylist` has no exact PySpark spelling; left as written
df = bt.from_pylist([{"id": 1, "name": "a", "score": 10}, {"id": 2, "name": "b", "score": 20}, {"id": 3, "name": "c", "score": 30}])
# batcher-migrate: PySpark `Dataset.filter` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.with_columns` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.rename` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.sort` has no exact PySpark spelling; left as written
out = (
    df.filter(F.col("score") > 10)
    .with_columns(points=F.col("score") * 2)
    .rename({"name": "label"})
    .select("id", "label", "points")
    .distinct()
    .sort("id", descending=False, nulls_first=True)
)
# batcher-migrate: PySpark `Dataset.to_pylist` has no exact PySpark spelling; left as written
result = out.limit(10).to_pylist()
