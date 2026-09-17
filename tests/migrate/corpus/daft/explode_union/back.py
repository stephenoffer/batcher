import daft

lists = daft.from_pydict({"k": [1, 2, 3], "v": [[1, 2], [], None]})
# batcher-migrate: Daft `Dataset.explode` has no exact Daft spelling; left as written
flat = lists.explode("v", outer=True)
# batcher-migrate: Daft `Dataset.union` has no exact Daft spelling; left as written
both = lists.union(lists, distinct=True)
print(flat.to_pydict(), both.count_rows())
