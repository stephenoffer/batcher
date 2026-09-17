import batcher as bt

df = bt.from_pydict({"x": [1, 5, 8], "s": ["apple", "banana", "cherry"]})
out = df.with_columns(bt.when(bt.col("x") > 2).then(bt.lit("big")).otherwise(bt.lit("small")).alias("size"), bt.col("s").str.upper().alias("upper"), bt.col("s").str.substr(2, 3).alias("middle"), bt.col("s").str.regexp_matches("an+").alias("has_an"), bt.col("s").str.contains("rr").alias("has_rr"), (bt.col("x") + 1).alias("next"))
result = out.sort("x", descending=False, nulls_first=True).to_pylist()
