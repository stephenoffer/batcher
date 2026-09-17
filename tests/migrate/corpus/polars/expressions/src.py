import polars as pl

df = pl.DataFrame({"x": [1, 5, 8], "s": ["apple", "banana", "cherry"]})
out = df.with_columns(
    pl.when(pl.col("x") > 2).then(pl.lit("big")).otherwise(pl.lit("small")).alias("size"),
    pl.col("s").str.to_uppercase().alias("upper"),
    pl.col("s").str.slice(1, 3).alias("middle"),
    pl.col("s").str.contains("an+").alias("has_an"),
    pl.col("s").str.contains("rr", literal=True).alias("has_rr"),
    (pl.col("x") + 1).alias("next"),
)
result = out.sort("x").to_dicts()
