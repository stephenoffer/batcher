import polars as pl

df = pl.DataFrame({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "score": [10, 20, 30, 40]})
out = (
    df.filter(pl.col("score") > 10)
    .with_columns((pl.col("score") * 2).alias("double"))
    .select("id", "name", pl.col("double").alias("points"))
    .unique()
    .sort("id")
    .head(2)
)
result = out.to_dicts()
