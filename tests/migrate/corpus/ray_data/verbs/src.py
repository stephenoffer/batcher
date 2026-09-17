import ray.data
from ray.data.expressions import col

ds = ray.data.from_items(
    [{"id": 1, "name": "a", "score": 10}, {"id": 2, "name": "b", "score": 20}, {"id": 3, "name": "c", "score": 30}]
)
out = (
    ds.filter(expr=col("score") > 10)
    .with_column("points", col("score") * 2)
    .rename_columns({"name": "label"})
    .select_columns(["id", "label", "points"])
    .drop_columns(["points"])
    .limit(5)
)
result = out.take_all()
