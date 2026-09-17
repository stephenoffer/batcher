import batcher as bt

df = bt.from_pydict({"a": [3, 1, 2]})
materialized = df.sort("a", descending=False, nulls_first=False).cache()
print(materialized.to_pydict(), df.count(), df.to_arrow())
