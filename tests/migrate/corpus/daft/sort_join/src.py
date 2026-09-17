import daft

orders = daft.from_pydict({"id": [1, 2, 3, 4], "customer": [10, 20, 10, None]})
customers = daft.from_pydict({"customer": [10, 20], "city": ["Oslo", "Lima"]})
joined = orders.join(customers, on="customer", how="left")
result = joined.sort(["customer", "id"], desc=[True, False]).to_pylist()
