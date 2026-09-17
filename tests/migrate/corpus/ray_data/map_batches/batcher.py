import batcher as bt


def add_total(batch):
    batch["total"] = batch["a"] + batch["b"]
    return batch


def tag(frame, suffix):
    frame["tag"] = frame["name"] + suffix
    return frame


ds = bt.from_items([{"a": 1, "b": 2, "name": "x"}, {"a": 3, "b": 4, "name": "y"}])
out = ds.map_batches(add_total, batch_format="numpy").map_batches(tag, batch_format="pandas", fn_kwargs={"suffix": "!"})
result = out.to_pylist()
