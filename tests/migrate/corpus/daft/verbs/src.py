import daft
from daft import col

df = daft.from_pydict({"id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "score": [10, 20, 30, 40]})
out = (
    df.where(col("score") > 10)
    .with_column("points", col("score") * 2)
    .select("id", "name", "points")
    .distinct()
    .sort("id")
    .limit(2)
)
result = out.to_pylist()
