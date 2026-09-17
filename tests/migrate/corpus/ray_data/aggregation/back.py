import ray.data
import ray.data.expressions

ds = ray.data.from_items([{"g": "a", "v": 1}, {"g": "b", "v": 2}, {"g": "a", "v": 3}])
# batcher-migrate: Ray Data `Dataset.group_by` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `GroupBy.len` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Dataset.to_pylist` has no exact Ray Data spelling; left as written
counts = ds.group_by("g").len(name="count()").to_pylist()
# batcher-migrate: Ray Data `Dataset.group_by` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Expr.sum` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `GroupBy.agg` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Dataset.to_pylist` has no exact Ray Data spelling; left as written
sums = ds.group_by("g").agg(**{"sum(v)": ray.data.expressions.col("v").sum()}).to_pylist()
result = sorted(counts, key=lambda r: r["g"]) + sorted(sums, key=lambda r: r["g"])
