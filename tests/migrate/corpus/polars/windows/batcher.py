import batcher as bt

df = bt.from_pydict({"g": ["a", "a", "b"], "t": [1, 2, 3], "v": [10, 20, 30]})
out = df.with_columns(bt.col("v").max(nan_policy="ignore").over(partition_by=["g"]).alias("group_max"), bt.col("v").sum(empty_value=0).over(partition_by=["g"], order_by=["t"], frame=(None, None)).alias("partition_total"))
result = out.sort("t", descending=False, nulls_first=True).to_pylist()
