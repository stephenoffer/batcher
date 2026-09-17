import batcher as bt

orders = bt.from_pydict({"id": [1, 2, 3, 4], "customer": [10, 20, 10, None]})
customers = bt.from_pydict({"customer": [10, 20], "city": ["Oslo", "Lima"]})
joined = orders.join(customers, "customer", how="left", suffix="_right")
result = joined.sort("customer", "id", descending=[True, False], nulls_first=True).to_pylist()
