import batcher as bt

orders = bt.from_pylist([{"id": 1, "customer": 10}, {"id": 2, "customer": 20}, {"id": 3, "customer": 10}, {"id": 4, "customer": None}])
customers = bt.from_pylist([{"customer": 10, "city": "Oslo"}, {"customer": 20, "city": "Lima"}])
joined = orders.join(customers, "customer", how="left")
ranked = joined.sort("customer", bt.col("id"), descending=[True, False], nulls_first=[False, True])
both = orders.union(orders.select(*orders.columns))
result = ranked.limit(10).to_pylist()
