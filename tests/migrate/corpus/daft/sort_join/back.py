import daft

orders = daft.from_pydict({"id": [1, 2, 3, 4], "customer": [10, 20, 10, None]})
customers = daft.from_pydict({"customer": [10, 20], "city": ["Oslo", "Lima"]})
# batcher-migrate: Daft `DataFrame.join` was rewritten; check: Daft names a clashing right column 'right.<col>' (prefix= / suffix=) and accepts strategy= hints; Batcher appends suffix='_right'. Param: prefix='right.', suffix=''
# batcher-migrate: Daft `Dataset.join` has no exact Daft spelling; left as written
joined = orders.join(customers, "customer", how="left")
# batcher-migrate: Daft `Dataset.sort` has no exact Daft spelling; left as written
result = joined.sort("customer", "id", descending=[True, False], nulls_first=[True, False]).to_pylist()
