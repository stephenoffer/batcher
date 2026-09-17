import daft

df = daft.from_pydict({"a": [3, 1, 2]})
# batcher-migrate: Daft `Dataset.sort` has no exact Daft spelling; left as written
# batcher-migrate: Daft `Dataset.cache` has no exact Daft spelling; left as written
materialized = df.sort("a", descending=False, nulls_first=False).cache()
print(materialized.to_pydict(), df.count_rows(), df.to_arrow())
