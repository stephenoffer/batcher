import polars as pl

df = pl.DataFrame({"g": ["a", "a", "b"], "t": [1, 2, 3], "v": [10, 20, 30]})
out = df.with_columns(
    pl.col("v").max().over("g").alias("group_max"),
    pl.col("v").sum().over("g", order_by="t").alias("partition_total"),
)
result = out.sort("t").to_dicts()
