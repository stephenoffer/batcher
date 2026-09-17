import ray.data

ds = ray.data.from_items([{"g": "a", "v": 1}, {"g": "b", "v": 2}, {"g": "a", "v": 3}])
counts = ds.groupby("g").count().take_all()
sums = ds.groupby("g").sum("v").take_all()
result = sorted(counts, key=lambda r: r["g"]) + sorted(sums, key=lambda r: r["g"])
