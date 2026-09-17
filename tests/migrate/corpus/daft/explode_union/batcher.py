import batcher as bt

lists = bt.from_pydict({"k": [1, 2, 3], "v": [[1, 2], [], None]})
flat = lists.explode("v", outer=True)
both = lists.union(lists, distinct=True)
print(flat.to_pydict(), both.count())
