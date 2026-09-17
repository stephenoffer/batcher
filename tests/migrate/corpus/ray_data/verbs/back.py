import ray.data
import ray.data.expressions

ds = ray.data.from_items(
    [{"id": 1, "name": "a", "score": 10}, {"id": 2, "name": "b", "score": 20}, {"id": 3, "name": "c", "score": 30}]
)
# batcher-migrate: Ray Data `Dataset.filter` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Dataset.with_columns` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Dataset.rename` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Dataset.select` has no exact Ray Data spelling; left as written
# batcher-migrate: Ray Data `Dataset.drop` has no exact Ray Data spelling; left as written
out = (
    ds.filter(ray.data.expressions.col("score") > 10)
    .with_columns(points=ray.data.expressions.col("score") * 2)
    .rename({"name": "label"})
    .select("id", "label", "points")
    .drop("points")
    .limit(5)
)
# batcher-migrate: Ray Data `Dataset.to_pylist` has no exact Ray Data spelling; left as written
result = out.to_pylist()
