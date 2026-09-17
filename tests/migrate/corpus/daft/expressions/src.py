import daft

df = daft.from_pydict({"x": [1, 5, 8], "s": ["apple", "banana", "cherry"]})
out = (
    df.with_column("upper", daft.col("s").upper())
    .with_column("head", daft.col("s").substr(0, 3))
    .with_column("next", daft.col("x") + daft.lit(1))
    .with_column("missing", daft.col("s").is_null())
)
result = out.sort("x").to_pylist()
