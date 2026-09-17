import polars as pl

orders = pl.DataFrame({"id": [1, 2, 3, 4], "customer": [10, 20, 10, None]})
customers = pl.DataFrame({"customer": [10, 20], "city": ["Oslo", "Lima"]})
joined = orders.join(customers, on="customer", how="left")
result = joined.sort("customer", "id", descending=[True, False]).to_dicts()
