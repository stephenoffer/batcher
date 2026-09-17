import batcher as bt

ds = bt.from_items([{"x": 1}, {"x": 2}])
for row in ds.iter_rows(named=True):
    print(row["x"])
# batcher-migrate: Ray Data `Dataset.to_pandas` was rewritten to `Dataset.to_pandas`, which lacks: limit= (raise if the dataset has more rows than limit)
print(ds.count(), ds.limit(1).to_pylist(), ds.to_pandas())
