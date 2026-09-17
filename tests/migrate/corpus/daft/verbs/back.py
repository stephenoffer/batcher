import daft

df = daft.from_pydict({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "score": [10, 20, 30, 40]})
# batcher-migrate: Daft `Dataset.filter` has no exact Daft spelling; left as written
# batcher-migrate: Daft `Dataset.with_columns` has no exact Daft spelling; left as written
# batcher-migrate: Daft `Dataset.sort` has no exact Daft spelling; left as written
out = (
    df.filter(daft.col("score") > 10)
    .with_columns(points=daft.col("score") * 2)
    .select("id", "name", "points")
    .distinct()
    .sort("id", descending=False, nulls_first=False)
    .limit(2)
)
result = out.to_pylist()
