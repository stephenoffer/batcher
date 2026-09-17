import ray.data
import ray.data.expressions


def add_total(batch):
    batch["total"] = batch["a"] + batch["b"]
    return batch


def tag(frame, suffix):
    frame["tag"] = frame["name"] + suffix
    return frame


ds = ray.data.from_items([{"a": 1, "b": 2, "name": "x"}, {"a": 3, "b": 4, "name": "y"}])
# batcher-migrate: Ray Data `Dataset.map_batches` has no exact Ray Data spelling; left as written
out = ds.map_batches(add_total, batch_format="numpy").map_batches(tag, batch_format="pandas", fn_kwargs={"suffix": "!"})
# batcher-migrate: Ray Data `Dataset.to_pylist` has no exact Ray Data spelling; left as written
result = out.to_pylist()
