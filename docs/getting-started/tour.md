# A tour of the engine

This page is the breadth claim, in runnable form. Batcher says it covers SQL and DataFrames, batch and streaming, tables and media, analytics and models, one core and a cluster. Below is one small working example of each, on a single page, so you can judge the claim by running it rather than by reading about it.

Every block here executes on every commit, as the whole documentation does. Where an example genuinely needs a broker, a GPU, or a cloud account, it is marked and the runnable stand-in is next to it.

```python
import batcher as bt
```

## Relational work, and ETL

The core is the part that looks like every other engine: filter, group, aggregate, write. What is different is that nothing has run yet when you finish typing it.

```python
orders = bt.from_pydict(
    {
        "region": ["eu", "us", "eu", "us"],
        "amount": [120.0, 80.0, 45.0, 300.0],
        "day": ["01", "01", "02", "02"],
    }
)
by_region = (
    orders.filter(bt.col("amount") > 50).group_by("region").agg(total=bt.col("amount").sum())
)
print(by_region.sort("region").to_pydict())
# {'region': ['eu', 'us'], 'total': [120.0, 380.0]}
```

A partitioned write with a completion marker is the other half of an ETL step:

```python
manifest = orders.write.parquet("warehouse/orders", partition_by=["region"], mode="overwrite")
print({"rows": manifest.total_rows, "files": manifest.num_files})
# {'rows': 4, 'files': 2}
```

{doc}`/user-guide/transform/index` · {doc}`/user-guide/moving-data/writing-data` · {doc}`/cookbook/data-engineering/index`

## SQL over the same plan

SQL is not a second engine or a compatibility layer. It builds the same `LogicalPlan` the verbs build, so you can start in one and finish in the other:

```python
print(
    bt.sql(
        "SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY region",
        orders=orders,
    ).to_pydict()
)
# {'region': ['eu', 'us'], 'total': [165.0, 380.0]}
```

{doc}`/user-guide/analyze/sql` · {doc}`/api/relational/sql`

## Joins and window functions

```python
users = bt.from_pydict({"uid": [1, 2, 3], "region": ["eu", "us", "eu"]})
spend = bt.from_pydict({"uid": [1, 1, 2, 3], "amt": [5.0, 7.0, 9.0, 2.0]})

totals = users.join(spend, on="uid").group_by("uid", "region").agg(total=bt.col("amt").sum())
ranked = totals.with_columns(
    rank=bt.rank().over(partition_by="region", order_by=[("total", True)])
)
print(ranked.sort("uid").to_pydict())
# {'uid': [1, 2, 3], 'region': ['eu', 'us', 'eu'], 'total': [12.0, 9.0, 2.0], 'rank': [1, 1, 2]}
```

{doc}`/user-guide/analyze/joins` · {doc}`/user-guide/analyze/window-functions`

## Semi-structured data

JSON and nested columns are expressions, not a parsing step you bolt on the front:

```python
logs = bt.from_pydict(
    {"line": ['{"lvl":"ERR","msg":"disk full"}', '{"lvl":"INFO","msg":"ok"}']}
)
print(
    logs.select(
        lvl=bt.col("line").json.extract_string("$.lvl"),
        words=bt.col("line").json.extract_string("$.msg").str.split(" "),
    ).to_pydict()
)
# {'lvl': ['ERR', 'INFO'], 'words': [['disk', 'full'], ['ok']]}
```

{doc}`/api/accessors/nested` · {doc}`/cookbook/expressions/nested/index`

## Streaming

Batch is the bounded case of streaming, so the operators are the same ones. The `rate` source generates rows without any external service, which is what makes this runnable:

```python
print([batch.num_rows for batch in bt.read.rate_micro_batch(4, num_rows=8).iter_batches()])
# [4, 4]
```

Against Kafka the query is the same shape. The source line changes, and a JSON payload declares its fields, because the plan is typed before the first message arrives:

```python
# docs: skip
clicks = bt.read.kafka(
    "clicks",
    bootstrap_servers="broker:9092",
    value_format="json",
    value_schema={"page": "string", "ms": "int64"},
)
pages = clicks.select(page=bt.col("value").struct.field("page"), ms=bt.col("value").struct.field("ms"))
pages.write.delta("lake/live", trigger=bt.Trigger.processing_time("10s"), checkpoint="lake/_ck")
```

A file or Delta sink takes appended rows only. A running aggregate goes to a memory sink with `output_mode="complete"`, or to `write.for_each_batch` for a custom upsert.

{doc}`/user-guide/moving-data/streaming/index` · {doc}`/integrations/streams/kafka`

## Data quality as part of the plan

A contract is a plan rewrite, so the check costs one pass and travels with the query:

```python
raw = bt.from_pydict({"id": [1, 2, 3], "amount": [10.0, -1.0, 30.0]})
print(raw.dq.positive("amount").drop().to_pydict())
# {'id': [1, 3], 'amount': [10.0, 30.0]}
```

`.fail()` raises instead, and `.quarantine()` routes the bad rows aside.

{doc}`/user-guide/trust/data-quality`

## Images, audio, and video

Media columns are binary columns, and the accessors read them. A header question never decodes a pixel:

```python
import base64

png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAACCAIAAADq9gq6AAAAEUlEQVR4nGO4o6GBFTHgkgAA4EESwW1PLREAAAAASUVORK5CYII="
)
shot = bt.from_pydict({"img": [png]})
print(
    shot.select(
        fmt=bt.col("img").image.format(), aspect=bt.col("img").image.aspect_ratio()
    ).to_pydict()
)
# {'fmt': ['png'], 'aspect': [4.0]}
```

{doc}`/user-guide/transform/columns/media-accessor` · {doc}`/ml/preparing/multimodal/index`

## Running a model

A fitted model meets the data where the data already is. Anything with the scikit-learn contract works, and the call takes the same scaling arguments whether it runs in one process or across a cluster:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(
    np.array([[0.0, 0.0], [0.2, 0.1], [4.0, 4.0], [3.9, 4.1]]), np.array([0, 0, 1, 1])
)
rows = bt.from_pydict({"f0": [0.1, 4.0], "f1": [0.0, 3.9]})
print(rows.ml.predict(model, features=["f0", "f1"], output_column="label").to_pydict()["label"])
# [0, 1]
```

{doc}`/integrations/compute/scikit-learn` · {doc}`/ml/inference/index`

## Embeddings and vector search

Similarity is an expression over a list column, so retrieval is a query rather than a separate index service:

```python
docs = bt.from_pydict(
    {"doc": ["a", "b", "c"], "vec": [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]}
)
near = docs.ml.similarity_to([1.0, 0.0], column="vec").sort("score", descending=True).limit(2)
print(near.select("doc").to_pydict())
# {'doc': ['a', 'c']}
```

{doc}`/ml/retrieval/vector-search` · {doc}`/ml/retrieval/rag`

## Lakehouse tables

A Delta write is a transaction, and `merge_on` makes it a keyed upsert, which is what makes a retried job safe:

```python
bt.from_pydict({"id": [1, 2], "v": ["a", "b"]}).write.delta("lake/t", mode="append")
bt.from_pydict({"id": [2, 3], "v": ["B", "c"]}).write.delta("lake/t", mode="append", merge_on="id")
print(bt.read.delta("lake/t").sort("id").to_pydict())
# {'id': [1, 2, 3], 'v': ['a', 'B', 'c']}
```

{doc}`/user-guide/moving-data/lakehouse` · {doc}`/integrations/lakehouse/index`

## Geospatial and graphs

Both are function libraries over ordinary columns, so a spatial predicate is a predicate and a graph algorithm is a sequence of joins:

```python
places = bt.from_pydict(
    {"name": ["oslo", "lima"], "lon": [10.75, -77.03], "lat": [59.91, -12.04]}
)
print(
    places.select(
        name=bt.col("name"), cell=bt.geohash_encode(bt.col("lon"), bt.col("lat"), 5)
    ).to_pydict()
)
# {'name': ['oslo', 'lima'], 'cell': ['u4xsu', '6mc5x']}
```

```python
import batcher.graph as bg

star = bt.from_pydict({"src": [1, 2, 3, 4], "dst": [0, 0, 0, 0]})
ranked = bg.pagerank(bg.Graph.from_edges(star)).sort("pagerank", descending=True)
print([(node, round(score, 3)) for node, score in zip(*ranked.to_pydict().values())][0])
# (0, 0.524)
```

{doc}`/user-guide/analyze/domains/index`

## Interop, in and out

Nothing above requires committing to Batcher for a whole pipeline. It shares Arrow with the libraries already in your process, so a single step can move here and the result can go straight back:

```python
import polars as pl

frame = pl.DataFrame({"city": ["Oslo", "Lima", "Oslo"], "temp": [3.5, 19.0, 5.5]})
print(
    bt.from_polars(frame)
    .group_by("city")
    .agg(avg=bt.col("temp").mean())
    .sort("city")
    .to_pydict()
)
# {'city': ['Lima', 'Oslo'], 'avg': [19.0, 4.5]}
```

{doc}`/integrations/dataframes/index`

## The same code, on a cluster

Distribution is an argument on the terminal call, not a rewrite. The plan, the operators, and the result are the same; only the scheduling changes:

```python
# docs: skip
by_region.collect(distributed=True, num_workers=8)
```

That works because every stateful operator is written once as a mergeable `partial → combine → finalize` triple, so one core, every core, and a cluster differ only in how that triple is scheduled.

{doc}`/user-guide/operate/running/index` · {doc}`/architecture/deep-dives/operators/mergeable-algebra`

## Where to go next

| You want | Read |
| --- | --- |
| The five-minute version, with a single pipeline | {doc}`quickstart` |
| The ideas behind the model | {doc}`concepts/index` |
| A guide per capability | {doc}`/user-guide/index` |
| A runnable recipe for your problem | {doc}`/cookbook/index` |
| The equivalent of a call you already know | {doc}`migration/index` |
| Whether it is actually fast | {doc}`/benchmarks/index` |
