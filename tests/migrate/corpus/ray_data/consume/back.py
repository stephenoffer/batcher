import ray.data
import ray.data.expressions

ds = ray.data.from_items([{"x": 1}, {"x": 2}])
# batcher-migrate: Ray Data `Dataset.iter_rows` has no exact Ray Data spelling; left as written
for row in ds.iter_rows(named=True):
    print(row["x"])
# batcher-migrate: Ray Data `Dataset.to_pandas` was rewritten to `Dataset.to_pandas`, which lacks: limit= (raise if the dataset has more rows than limit)
# batcher-migrate: Ray Data `Dataset.to_pylist` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Dataset.to_pandas` has no exact Ray Data spelling; left as written
print(ds.count(), ds.limit(1).to_pylist(), ds.to_pandas())
