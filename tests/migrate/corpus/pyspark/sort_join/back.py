import batcher as bt
from pyspark.sql import functions as F

# batcher-migrate: PySpark `bt.from_pylist` has no exact PySpark spelling; left as written
orders = bt.from_pylist([{"id": 1, "customer": 10}, {"id": 2, "customer": 20}, {"id": 3, "customer": 10}, {"id": 4, "customer": None}])
# batcher-migrate: PySpark `bt.from_pylist` has no exact PySpark spelling; left as written
customers = bt.from_pylist([{"customer": 10, "city": "Oslo"}, {"customer": 20, "city": "Lima"}])
# batcher-migrate: PySpark `Dataset.join` has no exact PySpark spelling; left as written
joined = orders.join(customers, "customer", how="left")
# batcher-migrate: PySpark `Dataset.sort` has no exact PySpark spelling; left as written
ranked = joined.sort("customer", F.col("id"), descending=[True, False], nulls_first=[False, True])
# batcher-migrate: PySpark `Dataset.select` has no exact PySpark spelling; left as written
# batcher-migrate: PySpark `Dataset.union` has no exact PySpark spelling; left as written
both = orders.union(orders.select(*orders.columns))
# batcher-migrate: PySpark `Dataset.to_pylist` has no exact PySpark spelling; left as written
result = ranked.limit(10).to_pylist()
