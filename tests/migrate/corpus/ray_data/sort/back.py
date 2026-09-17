import ray.data
import ray.data.expressions

ds = ray.data.from_items([{"k": "b", "n": 2}, {"k": "a", "n": 3}, {"k": "c", "n": 1}])
# batcher-migrate: Ray Data `Dataset.sort` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Dataset.to_pylist` has no exact Ray Data spelling; left as written
result = ds.sort("n", descending=True).limit(2).to_pylist()
