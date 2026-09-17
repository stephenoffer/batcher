import batcher as bt
from pyspark.sql import functions as F

# batcher-migrate: PySpark `bt.from_pylist` has no exact PySpark spelling; left as written
df = bt.from_pylist([{"g": "a", "t": 1, "v": 10}, {"g": "a", "t": 2, "v": 20}, {"g": "b", "t": 3, "v": 30}])
# batcher-migrate: PySpark `Column.over` was rewritten; check: Spark orders nulls first in an ascending window key; Batcher last
# batcher-migrate: PySpark `Expr.over` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.with_columns` has no exact PySpark spelling; left as written
out = df.with_columns(rn=F.row_number().over(partition_by=["g"], order_by=[("t", True)])).with_columns(running=F.sum(F.col("v")).over(partition_by=["g"], order_by=["t"], frame=(None, 0)))
# batcher-migrate: PySpark `Dataset.sort` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.to_pylist` has no exact PySpark spelling; left as written
result = out.sort("t", descending=False, nulls_first=True).limit(10).to_pylist()
