import batcher as bt

ds = bt.from_items(
    [{"id": 1, "name": "a", "score": 10}, {"id": 2, "name": "b", "score": 20}, {"id": 3, "name": "c", "score": 30}]
)
out = (
    ds.filter(bt.col("score") > 10)
    .with_columns(points=bt.col("score") * 2)
    .rename({"name": "label"})
    .select("id", "label", "points")
    .drop("points")
    .limit(5)
)
result = out.to_pylist()
