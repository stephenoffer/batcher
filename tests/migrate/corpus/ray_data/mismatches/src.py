import ray.data

ds = ray.data.from_items([{"k": 1}, {"k": 1}, {"k": 2}])
shuffled = ds.random_shuffle()
pinned = ds.materialize()
values = ds.unique("k")
widened = ds.add_column("k2", lambda frame: frame["k"] * 2)
print(shuffled, pinned, values, widened)
