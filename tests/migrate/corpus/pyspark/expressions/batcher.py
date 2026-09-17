import batcher as bt

df = bt.from_pylist([{"x": 1, "s": "apple"}, {"x": 5, "s": "banana"}])
out = (
    df.with_columns(size=bt.when(bt.col("x") > 2).then("big").otherwise("small"))
    .with_columns(upper=bt.col("s").str.upper())
    .with_columns(head=bt.col("s").str.substr(1, 3))
    .with_columns(next=bt.col("x") + bt.lit(1))
    .with_columns(same=df["x"])
)
# batcher-migrate: PySpark `DataFrame.show` was rewritten to `Dataset.show`, which lacks: truncate= and vertical= display options (Spark shows 20 rows by default)
out.show()
