import daft

df = daft.from_pydict({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
# batcher-migrate: Daft `Dataset.group_by` has no exact Daft spelling; left as written
out = df.group_by("g").agg(
    daft.col("v").sum().alias("total"),
    daft.col("v").mean().alias("average"),
    daft.col("v").min().alias("lowest"),
)
# batcher-migrate: Daft `Dataset.sort` has no exact Daft spelling; left as written
result = out.sort("g", descending=False, nulls_first=False).to_pylist()
