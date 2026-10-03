# A tour of the engine

This page runs one small example of each thing Batcher does: SQL, DataFrames, streams, media, models, lakehouse tables and graphs, from one core up to a cluster. Every block runs on every commit.

```python
import batcher as bt
```

## Relational work and SQL

Filter, group, aggregate, write. Nothing runs until you ask for a result.

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

A partitioned write returns a manifest:

```python
manifest = orders.write.parquet("warehouse/orders", partition_by=["region"], mode="overwrite")
print({"rows": manifest.total_rows, "files": manifest.num_files})
# {'rows': 4, 'files': 2}
```

{doc}`/user-guide/transform/index` · {doc}`/user-guide/moving-data/writing-data` · {doc}`/cookbook/data-engineering/index`

SQL builds the same plan the verbs build. Start in one and finish in the other:

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

## Text, JSON and time

JSON and nested columns are expressions, so you never bolt a parsing step onto the front:

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

Lists explode into rows:

```python
tagged = bt.from_pydict({"k": [1, 1, 2], "tags": [["a", "b"], ["c"], []]})
print(tagged.explode("tags").to_pydict())
# {'k': [1, 1, 1], 'tags': ['a', 'b', 'c']}
```

{doc}`/api/accessors/nested` · {doc}`/cookbook/expressions/nested/index`

Text cleanup and date parts are accessor calls too:

```python
import datetime as dt

notes = bt.from_pydict({"text": ["Contact bob@x.io now", "no email here"]})
print(notes.select(masked=bt.col("text").str.mask_emails()).to_pydict())
# {'masked': ['Contact [EMAIL] now', 'no email here']}

events = bt.from_pydict({"ts": [dt.datetime(2026, 1, 5, 9, 30), dt.datetime(2026, 3, 1, 17, 0)]})
print(events.select(month=bt.col("ts").dt.month(), hour=bt.col("ts").dt.hour()).to_pydict())
# {'month': [1, 3], 'hour': [9, 17]}
```

{doc}`/user-guide/transform/columns/expressions`

## Streaming

Batch is the bounded case of streaming, so the operators are the same. The `rate` source needs no external service:

```python
print([batch.num_rows for batch in bt.read.rate_micro_batch(4, num_rows=8).iter_batches()])
# [4, 4]
```

Against Kafka only the source line changes:

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

{doc}`/user-guide/moving-data/streaming/index` · {doc}`/integrations/streams/kafka`

## Data quality

A contract is a plan rewrite. The check costs one pass and travels with the query:

```python
raw = bt.from_pydict({"id": [1, 2, 3], "amount": [10.0, -1.0, 30.0]})
print(raw.dq.positive("amount").drop().to_pydict())
# {'id': [1, 3], 'amount': [10.0, 30.0]}
```

`.fail()` raises instead, and `.quarantine()` routes the bad rows aside.

{doc}`/user-guide/trust/data-quality`

## Images, audio, and video

Media columns are binary columns with accessors. A header question never decodes a pixel:

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

## Models and vectors

Anything with the scikit-learn contract runs over a dataset:

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

Similarity is an expression over a list column. Retrieval is a query:

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

A Delta write is a transaction, and `merge_on` makes it a keyed upsert:

```python
bt.from_pydict({"id": [1, 2], "v": ["a", "b"]}).write.delta("lake/t", mode="append")
bt.from_pydict({"id": [2, 3], "v": ["B", "c"]}).write.delta("lake/t", mode="append", merge_on="id")
print(bt.read.delta("lake/t").sort("id").to_pydict())
# {'id': [1, 2, 3], 'v': ['a', 'B', 'c']}
```

{doc}`/user-guide/moving-data/lakehouse` · {doc}`/integrations/lakehouse/index`

## Geospatial and graphs

Both are function libraries over ordinary columns:

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

You don't have to move a whole pipeline. Batcher shares Arrow with the libraries already in your process, so a single step can run here and hand its result straight back:

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

Distribution is an argument on the terminal call, not a rewrite:

```python
# docs: skip
by_region.collect(distributed=True, num_workers=8)
```

Every stateful operator is written once. One core and a whole cluster run the same code.

{doc}`/user-guide/operate/running/index` · {doc}`/architecture/deep-dives/operators/mergeable-algebra`

## Where to go next

| You want | Read |
| --- | --- |
| The five-minute version, with a single pipeline | {doc}`quickstart` |
| The ideas behind the model | {doc}`concepts/index` |
| A guide per capability | {doc}`/user-guide/index` |
| A runnable recipe for your problem | {doc}`/cookbook/index` |
| The equivalent of a call you already know | {doc}`migration/index` |
| How fast it is | {doc}`/benchmarks/index` |
