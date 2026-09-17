import ray.data
from ray.data.expressions import col, lit

ds = ray.data.from_items([{"s": "Hello", "n": 1}])
out = ds.with_column("upper", col("s").str.upper()).with_column("plus", col("n") + lit(10))
print(out.take_all())
