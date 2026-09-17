import batcher as bt

orders = bt.from_pydict({"id": [1, 2, 3, 4], "customer": [10, 20, 10, None]})
customers = bt.from_pydict({"customer": [10, 20], "city": ["Oslo", "Lima"]})
# batcher-migrate: Daft `DataFrame.join` was rewritten; check: Daft names a clashing right column 'right.<col>' (prefix= / suffix=) and accepts strategy= hints; Batcher appends suffix='_right'. Param: prefix='right.', suffix=''
joined = orders.join(customers, "customer", how="left")
result = joined.sort("customer", "id", descending=[True, False], nulls_first=[True, False]).to_pylist()
