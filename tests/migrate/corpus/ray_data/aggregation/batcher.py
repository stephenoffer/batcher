import batcher as bt

ds = bt.from_items([{"g": "a", "v": 1}, {"g": "b", "v": 2}, {"g": "a", "v": 3}])
counts = ds.group_by("g").len(name="count()").to_pylist()
sums = ds.group_by("g").agg(**{"sum(v)": bt.col("v").sum()}).to_pylist()
result = sorted(counts, key=lambda r: r["g"]) + sorted(sums, key=lambda r: r["g"])
