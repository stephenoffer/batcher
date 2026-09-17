import batcher as bt

df = bt.from_pydict({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "score": [10, 20, 30, 40]})
out = (
    df.filter(bt.col("score") > 10)
    .with_columns(points=bt.col("score") * 2)
    .select("id", "name", "points")
    .distinct()
    .sort("id", descending=False, nulls_first=False)
    .limit(2)
)
result = out.to_pylist()
