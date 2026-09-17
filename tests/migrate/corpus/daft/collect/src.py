import daft

df = daft.from_pydict({"a": [3, 1, 2]})
materialized = df.sort("a").collect()
print(materialized.to_pydict(), df.count_rows(), df.to_arrow())
