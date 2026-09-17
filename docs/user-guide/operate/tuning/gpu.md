# Running a query on the GPU

This page covers the GPU backend for relational queries: how to ask for it, how it spreads one query across several devices, and what happens to a shape it can't run.

A supported relational query can run on the GPU through cuDF instead of the CPU engine. You choose with the `backend=` argument to `collect()`. The query and the result stay the same, and only *where* it runs changes.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://warehouse/events/")
q = ds.group_by("country").agg(revenue=bt.col("amount").sum())

q.collect(backend="cpu")  # the native engine (default)
q.collect(backend="gpu")  # force the cuDF GPU backend for any supported shape
q.collect(backend="auto")  # let Kyber decide GPU vs CPU by estimated size
```

## Let Kyber choose with `backend="auto"`

`backend="auto"` sends a query to the GPU only when the estimated input is large enough to repay the device's fixed overhead: host-to-device transfer, the cuDF import, and task dispatch. Below that crossover a query stays on the CPU engine.

The crossover is learned, not fixed. Every GPU or CPU group-by run records its actual input rows and wall time to the metadata hub, Kyber fits a cost line per backend, and the threshold moves to where the lines cross on your hardware. A faster GPU or a slower CPU shifts it without any configuration. Until enough runs exist for both backends, Kyber uses `distributed.gpu_min_rows`, 10,000,000 rows by default.

## How a query spreads across devices

Above the crossover, the plan's shape decides how far the query scales, more than the cluster does. A chain that reduces, such as a group-by aggregate, a distinct, or a sort with a limit, splits across every device. Each device reads its own shard from storage and reduces it, so what has to fit a device is one shard, not the whole working set. A chain with nothing to reduce splits too, and the shards' rows reassemble in order. A join splits its large side and gives every device the small side, when the planner sized the small side to fit.

```python
# docs: skip
# reduces: shards across every device
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

Losing a device costs a shard, not the query. A shard too large for its device is divided into `distributed.gpu_shard_subdivide` pieces, 4 by default, and rerun on the device. A shard whose device is lost, or that fails for any other reason, is recomputed by the CPU engine, which produces the identical partial result, as long as `distributed.gpu_shard_cpu_fallback` stays on.

## How the result stays correct

The GPU backend is always safe to request. Any shape outside the translated subset runs on the CPU engine instead, and so does every query on a cluster with no visible GPU.

That subset is a deliberate split of the engine's vocabulary rather than a list of features, as the figure shows:

![The device tier as a translator that must classify every tag. Every other execution tier consumes the same Rust bc_expr::Expr, so it has one definition of what a shape means and cannot drift. The device tier cannot, because cuDF has no Rust binding, so it restates the semantics in another language and every IR tag must be classified. Each tag is in exactly one of two sets. Translated tags, SUPPORTED_OPS and the expression handlers, cover filter, project, aggregate, sort, distinct, limit, window, unnest, unpivot and row_id. Declined tags, DECLINED_OPS and DECLINED_EXPRS, each carry a reason: asof_join, range_join and sample are not translated, and image, audio and geo expressions are Rust kernels with no dataframe equivalent. A tag in neither set fails test_gpu_vocabulary_contract. If every node of a plan translates, the plan runs on the device with cuDF, one shard per device, but only as a chain over a scan, a join of two chains, or a union of chains. If any node declines, the whole plan runs on the CPU engine and returns the same rows more slowly, which is why backend="gpu" is always safe to ask for.](/_static/diagrams/gpu_tier_decision.svg)

Every device result is also checked against the column types the engine declares for the plan before any backend runs. A result whose schema disagrees is refused and the CPU engine answers instead. That check is always on and touches no rows. For benchmark and staging runs, `distributed.gpu_shadow_verify=True` goes further and re-runs each result on the CPU engine to compare values, at the cost of doing the work twice.

## Requirements and limitations

- The backend needs a Ray cluster with visible GPUs. Without one, every `backend` value runs on the CPU engine.
- `backend` defaults to `"cpu"`, so the GPU is never used unless you ask for it.
- The learned crossover is fitted from group-by runs.
- This backend covers relational operators. Model inference reaches the GPU through the ML pipelines instead.

## See also

- {doc}`performance`: the CPU levers, and the memory envelope both backends run inside.
- {doc}`/user-guide/operate/running/gpu-diagnosis`: finding out why a GPU stage was slower than it should have been.
- {doc}`/user-guide/operate/running/gpu-fleets`: running Batcher inside a GPU datacenter.
- {doc}`/architecture/deep-dives/distribution/gpu-execution`: how the translator and the sharded fan-out work.
- {doc}`/ml/inference/index`: the inference pipelines that use a device for the model rather than for the relational operators.
- {doc}`/configuration/distributed-options`: every `distributed.gpu_*` setting.
