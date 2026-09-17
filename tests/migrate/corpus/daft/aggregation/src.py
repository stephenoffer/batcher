import daft
from daft import col

df = daft.from_pydict({"g": ["a", "b", "a", "b", "c"], "v": [1, 2, 3, 4, 5]})
out = df.groupby("g").agg(
    col("v").sum().alias("total"),
    col("v").mean().alias("average"),
    col("v").min().alias("lowest"),
)
result = out.sort("g").to_pylist()
