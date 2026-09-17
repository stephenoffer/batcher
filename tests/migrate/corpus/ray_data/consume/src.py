import ray.data

ds = ray.data.from_items([{"x": 1}, {"x": 2}])
for row in ds.iter_rows():
    print(row["x"])
print(ds.count(), ds.take(1), ds.to_pandas())
