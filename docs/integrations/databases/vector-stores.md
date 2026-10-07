# Vector stores

This page covers writing embeddings into Qdrant, Pinecone, Milvus and Turbopuffer, and reading them back out. The four connectors share one contract, so a frame that writes to one of them has the shape the others expect, and each one checks your points before it sends the first request.

:::{warning}
None of these connectors has been run against a live service yet. Each one is tested against a recording stand-in for its client library, which pins the requests it sends but cannot prove a server accepts them. Live smoke tests exist under [`tests/integration/live/`](https://github.com/stephenoffer/batcher/tree/main/tests/integration/live) and are tracked in [`tests/PENDING_VERIFICATION.md`](https://github.com/stephenoffer/batcher/blob/main/tests/PENDING_VERIFICATION.md). Verify a pipeline against your own deployment before you rely on it.
:::

The following table shows what each connector reads and writes, and the extra that installs its client:

| Store | Read | Write | `mode` | Extra |
| --- | --- | --- | --- | --- |
| Qdrant | {py:meth}`bt.read.qdrant <batcher.api.io_namespace.reader.Reader.qdrant>`, by scroll | {py:meth}`ds.write.qdrant <batcher.api.io_namespace.writer.Writer.qdrant>` | `upsert` / `delete` | `qdrant` |
| Pinecone | {py:meth}`bt.read.pinecone <batcher.api.io_namespace.reader.Reader.pinecone>`, serverless indexes only | {py:meth}`ds.write.pinecone <batcher.api.io_namespace.writer.Writer.pinecone>` | `upsert` / `delete` | `pinecone` |
| Milvus | {py:meth}`bt.read.milvus <batcher.api.io_namespace.reader.Reader.milvus>`, one split per partition | {py:meth}`ds.write.milvus <batcher.api.io_namespace.writer.Writer.milvus>` | `upsert` / `append` / `delete` | `milvus` |
| Turbopuffer | {py:meth}`bt.read.turbopuffer <batcher.api.io_namespace.reader.Reader.turbopuffer>`, paged by id | {py:meth}`ds.write.turbopuffer <batcher.api.io_namespace.writer.Writer.turbopuffer>` | `upsert` / `delete` | `turbopuffer` |

`pip install 'batcher-engine[vector]'` installs all four clients.

## What shape does a frame need?

A frame bound for a vector store has an *id column*, a *vector column*, and payload. The id column is `id` unless you pass `id_column=`, and it holds integers or strings. The vector column is `embedding` unless you pass `vector_column=`, which is the column {py:func}`batcher.ml.embed` writes. Its default fixed-shape tensor output is accepted as it is. Every other column is payload: Qdrant's payload, Pinecone's metadata, Milvus' scalar fields, or Turbopuffer's attributes.

A vector column is `fixed_size_list<float32, dim>`. A fixed-size list of another numeric type, a one-dimensional fixed-shape tensor column, and a plain list column whose rows all have the same length are accepted and converted to it. Reading a store gives vectors back as `fixed_size_list<float32, dim>`.

```python
# docs: skip
import batcher as bt

docs = bt.read.parquet("s3://<your-bucket>/docs.parquet")
embedded = docs.ml.embed("sentence-transformers/all-MiniLM-L6-v2", column="text")
embedded.write.qdrant("docs", url="http://qdrant.internal:6333", metric="cosine")
```

## What is checked before the first request?

Every store refuses a malformed point, but it refuses it partway through a write, after the batches before it have landed. So the connector checks the whole frame first, in Arrow, and sends nothing if any point is wrong. An id that is null or appears twice is refused, and so is a vector that is null, has the wrong number of values, or holds a NaN or an infinity. A float64 value too large for float32 counts as an infinity.

A refused write raises {py:class}`VectorWriteError <batcher.io.formats.vector.VectorWriteError>`, and its `failures` attribute names every point with its reason. This block runs without a server, because the check happens before the client is opened:

```python
import batcher as bt
from batcher.io.formats.vector import VectorWriteError

frame = bt.from_pydict({"id": [1, 2], "embedding": [[0.1, 0.2], [float("nan"), 0.4]]})
try:
    frame.write.qdrant("docs", url="http://localhost:6333")
except VectorWriteError as err:
    print(err.failures)
    print(err.written)
# (PointFailure(id=2, reason='vector holds a NaN, infinity or null'),)
# 0
```

The connector then asks the store about the target. Qdrant reports the collection's vector size and distance, Pinecone reports the index's dimension and metric, and Milvus reports the vector field's dimension and its index's metric. A frame whose dimension disagrees, or a `metric=` that disagrees, is refused with nothing written. A Turbopuffer namespace is created by its first write, so the dimension is checked only when the namespace already exists.

`metric=` takes `"cosine"`, `"euclidean"` or `"dot"`, or the store's own spelling, such as Pinecone's `"dotproduct"` or Milvus' `"IP"`. Leave it out to skip the metric check. Turbopuffer sends `cosine_distance` when you leave it out, and has no `"dot"`.

## How are retries and failures handled?

Points go out in batches of `batch_size`, and a failed batch is retried up to `max_retries` times with exponential backoff. A retried upsert lands on the same ids, so it can't duplicate a point. This is why every write needs an id column. Qdrant accepts only unsigned integers and UUIDs as ids, so a string id that is neither is sent as a UUID derived from it, the same UUID on every run, and the original string is kept in the payload under the id column's name, which is where `bt.read.qdrant` reads it back from.

Milvus `append` is an `insert`, which doesn't check the primary key, so it is never retried: a retry of a request that landed but whose response was lost would store the rows twice. Use `upsert` when a job may be re-run.

A batch that still fails after its retries doesn't stop the write. The other batches are sent, and the write then raises one `VectorWriteError` whose `failures` list every point of the failed batches and whose `written` counts the points that landed. Re-running the same upsert repairs it.

## Store-specific behavior

Each store maps the frame onto its own model a little differently. The differences that change what you write are listed here.

Qdrant
: Named vectors are written with `vectors={"text": "text_emb", "image": "image_emb"}`, or with `vector_name=` for one of them. `location=":memory:"` uses the client's in-process mode. Reading scrolls one cursor, so it is a single split.

Pinecone
: Metadata holds strings, numbers, booleans and lists of strings. A column of any other type is refused before the write, and a null is left out of that record's metadata. Integer ids are sent as decimal strings. `namespace=` picks the namespace for reads and writes.

Milvus
: The id column is sent as the collection's primary key and the vector column as its float-vector field, whatever they are called there. Pass `vector_field=` when the collection has several. A column the collection has no field for is refused unless the collection's dynamic field is enabled. `partition=` targets one partition on write, and a read fans out across partitions. `uri="./milvus.db"` uses Milvus Lite.

Turbopuffer
: The id column is sent as `id` and the vector column as `vector`, and a payload column named either of those is refused. Pass `region=` or `base_url=`, never both. `schema=` forwards a Turbopuffer attribute schema with every write.

Each sink writes into a target you have already created, except a Turbopuffer namespace. None of them supports `overwrite`, because emptying a collection or an index is an administrative operation rather than a write.

## Requirements and limitations

- Each connector needs its client library: `qdrant-client`, `pinecone`, `pymilvus` 2.5 or later, or `turbopuffer`.
- Reads push no filter to the store. The engine applies the filter after the read, so a selective query still reads the whole collection. Milvus' `filter=` expression is passed to the server when you set it yourself.
- Pinecone reads use `Index.list`, which serverless indexes support and pod-based indexes don't.
- Milvus reads map scalar and float-vector fields. A JSON, array, sparse or binary-vector field needs a declared `schema=`.
- Turbopuffer reads take their columns from the namespace's attribute schema.
- API keys and tokens accept `env:` and `file:` references, resolved on the worker, as {doc}`Secrets and keys </user-guide/trust/secrets>` describes.

## See also

- {doc}`/integrations/databases/writing`: the write modes these sinks share with the database sinks.
- {doc}`/integrations/databases/elasticsearch`: another `_bulk`-style sink with per-item failure checks.
- {doc}`/getting-started/migration/ray-data/index`: `write_turbopuffer` in the Ray Data migration reference.
