import daft

df = daft.from_pydict({"x": [1, 5, 8], "s": ["apple", "banana", "cherry"]})
# batcher-migrate: Daft `Dataset.with_columns` has no exact Daft spelling; left as written
# batcher-migrate: Daft `Expr.str.slice` has no exact Daft spelling; left as written
out = (
    df.with_columns(upper=daft.col("s").upper())
    .with_columns(head=daft.col("s").str.slice(0, 3))
    .with_columns(next=daft.col("x") + daft.lit(1))
    .with_columns(missing=daft.col("s").is_null())
)
# batcher-migrate: Daft `Dataset.sort` has no exact Daft spelling; left as written
result = out.sort("x", descending=False, nulls_first=False).to_pylist()
