import polars as pl

df = pl.DataFrame({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
out = df.group_by("g").agg(
    pl.col("v").min().alias("lowest"),
    pl.col("v").max().alias("highest"),
    pl.col("v").mean().alias("average"),
    pl.len().alias("rows"),
)
result = out.sort("g").to_dicts()
