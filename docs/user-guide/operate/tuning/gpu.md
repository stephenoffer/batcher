# Running a query on the GPU

This page covers the GPU backend: how to ask for it, what it does with more than one device,
and what happens to a shape it cannot run.

A supported relational query can run on the GPU (cuDF) instead of the CPU engine by
passing `backend=` to `collect()`. It is the same query and the same result. Only
*where* it runs changes.

```python
# docs: skip
ds = bt.read.parquet("s3://warehouse/events/")
q = ds.group_by("country").agg(revenue=bt.col("amount").sum())

q.collect(backend="cpu")  # the native engine (default)
q.collect(backend="gpu")  # force the cuDF GPU backend for any supported shape
q.collect(backend="auto")  # let Kyber decide GPU vs CPU by estimated size
```

`backend="auto"` is the adaptive choice. Kyber sends a query to the GPU only when the
estimated input is large enough to amortize the device overhead of host-to-device
transfer, cuDF import, and task dispatch. Below that crossover a small query stays on
the CPU engine, and anything unsupported or a GPU-less cluster falls back
transparently.

Above it, how far the query scales depends on the plan rather than on the cluster. A
chain that reduces (a group-by aggregate, a distinct, or a sort with a limit) splits
across every device, each reading its own shard from storage and reducing it, so the
memory that has to fit a device is one shard's rather than the whole working set's. A
chain with nothing to reduce splits too, with each shard's rows reassembled in order.
A join splits its large side and gives every device the small one, when the planner
sized the small side to fit.

That is why the shape of the query, not just its size, decides whether the GPU helps:

```python
# docs: skip
# reduces: shards across every device, whatever the input size
ds.group_by("country").agg(revenue=bt.col("amount").sum()).collect(backend="auto")

# reduces, and the sort and limit run once on the folded result
(
    ds.group_by("country")
    .agg(r=bt.col("amount").sum())
    .sort("r", descending=True)
    .limit(10)
    .collect(backend="auto")
)

# does not reduce: still splits, and the shards' rows reassemble in order
ds.filter(bt.col("amount") > 100).collect(backend="auto")

# a star schema: the fact side splits, every device reads the dimension itself
(
    facts.join(dims, on="sku")
    .group_by("category")
    .agg(r=bt.col("amount").sum())
    .collect(backend="auto")
)
```

A shard that a device cannot hold is subdivided and rerun on the device rather than
abandoned, and a shard whose device is lost is recomputed by the CPU engine, which
produces the identical partial. Losing a device costs that shard, not the query.

The crossover itself is learned rather than fixed. Each GPU or CPU group-by run
records its estimated rows and wall time to the metadata hub, Kyber fits a cost line
per backend and solves for their intersection, so the threshold self-corrects to the
hardware you have. Until enough runs are seen it uses the measured default,
`distributed.gpu_min_rows`.

## See also

- {doc}`performance`: the CPU levers, and the memory envelope both backends run inside.
- {doc}`/user-guide/ml/index`: the inference pipelines that use a device for the model rather
  than for the relational operators.
