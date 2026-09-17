import ray.data
import ray.data.expressions

ds = ray.data.from_items([{"s": "Hello", "n": 1}])
# batcher-migrate: Ray Data `Dataset.with_columns` has no exact Ray Data spelling; left as written
out = ds.with_columns(upper=ray.data.expressions.col("s").str.upper()).with_columns(plus=ray.data.expressions.col("n") + ray.data.expressions.lit(10))
# batcher-migrate: Ray Data `Dataset.to_pylist` has no exact Ray Data spelling; left as written
print(out.to_pylist())
