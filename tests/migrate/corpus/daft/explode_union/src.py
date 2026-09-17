import daft

lists = daft.from_pydict({"k": [1, 2, 3], "v": [[1, 2], [], None]})
flat = lists.explode(daft.col("v"))
both = lists.union(lists)
print(flat.to_pydict(), both.count_rows())
