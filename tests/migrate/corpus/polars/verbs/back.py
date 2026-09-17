import polars as pl

df = pl.from_dict({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "score": [10, 20, 30, 40]})
# batcher-migrate: Polars `Dataset.with_columns` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.select` has no exact Polars spelling; left as written
# batcher-migrate: Polars `Dataset.sort` has no exact Polars spelling; left as written
out = (
    df.filter(pl.col("score") > 10)
    .with_columns((pl.col("score") * 2).alias("double"))
    .select("id", "name", pl.col("double").alias("points"))
    .unique()
    .sort("id", descending=False, nulls_first=True)
    .limit(2)
)
result = out.to_dicts()
