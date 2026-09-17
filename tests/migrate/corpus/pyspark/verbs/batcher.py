import batcher as bt

df = bt.from_pylist([{"id": 1, "name": "a", "score": 10}, {"id": 2, "name": "b", "score": 20}, {"id": 3, "name": "c", "score": 30}])
out = (
    df.filter(bt.col("score") > 10)
    .with_columns(points=bt.col("score") * 2)
    .rename({"name": "label"})
    .select("id", "label", "points")
    .distinct()
    .sort("id", descending=False, nulls_first=True)
)
result = out.limit(10).to_pylist()
