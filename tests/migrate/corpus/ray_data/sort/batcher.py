import batcher as bt

ds = bt.from_items([{"k": "b", "n": 2}, {"k": "a", "n": 3}, {"k": "c", "n": 1}])
result = ds.sort("n", descending=True).limit(2).to_pylist()
