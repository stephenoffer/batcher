import polars as pl

orders = pl.from_dict({"id": [1, 2, 3, 4], "customer": [10, 20, 10, None]})
customers = pl.from_dict({"customer": [10, 20], "city": ["Oslo", "Lima"]})
# batcher-migrate: Polars `Dataset.join` has no exact Polars spelling; left as written
joined = orders.join(customers, "customer", how="left", suffix="_right")
# batcher-migrate: Polars `Dataset.sort` has no exact Polars spelling; left as written
result = joined.sort("customer", "id", descending=[True, False], nulls_first=True).to_dicts()
