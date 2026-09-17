import batcher as bt

df = bt.from_pylist([{"g": "a", "t": 1, "v": 10}, {"g": "a", "t": 2, "v": 20}, {"g": "b", "t": 3, "v": 30}])
# batcher-migrate: PySpark `Column.over` was rewritten; check: Spark orders nulls first in an ascending window key; Batcher last
out = df.with_columns(rn=bt.row_number().over(partition_by=["g"], order_by=[("t", True)])).with_columns(running=bt.sum(bt.col("v")).over(partition_by=["g"], order_by=["t"], frame=(None, 0)))
result = out.sort("t", descending=False, nulls_first=True).limit(10).to_pylist()
