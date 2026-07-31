# Cloud storage

Batcher reads and writes object stores through the same API as local files. A path with a cloud scheme is recognized automatically, so only the path changes and the rest of the pipeline is identical.

The accepted schemes are `s3://` / `s3a://`, `gs://` / `gcs://`, `az://` / `abfs://` / `abfss://`, and `hdfs://`, plus `file://` and bare local paths. Reading and writing both go through one `pyarrow.fs`-backed filesystem. Anything pyarrow does not implement natively falls back to fsspec behind the same interface.

## Prerequisites

Object-store access needs the cloud extra:

```bash
pip install 'batcher-engine[cloud]'
```

If a cloud scheme is used without the extra installed, the read fails with a message telling you to install it.

Every example on this page needs a real bucket and credentials, so the blocks are shown but not executed.

## Reading from object storage

{py:obj}`bt.read <batcher.read>` infers the format from the extension. The format-specific readers (`bt.read.parquet`, `bt.read.csv`, `bt.read.json`) take the same cloud paths.

Use a typed reader when the path contains a glob. Format inference reads the extension off the literal path and stops at the first `*`, so `bt.read("s3://b/*.parquet")` has nothing to infer from and raises `FormatError`. `bt.read.parquet(...)` is already explicit. A `*` also matches within one path segment only, so crossing directories in a Hive layout needs `**`.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/events/*.parquet")
out = ds.filter(bt.col("status") == "active").select("user_id", "amount")
print(out.to_pydict())
```

A glob reads many files as one Dataset. Reading stays lazy: no bytes are fetched until a terminal operation runs, and projection and filter pushdown limit what is read.

```python
# docs: skip
ds = bt.read.parquet("s3://bucket/year=2024/month=06/*.parquet")
```

Only the scheme changes between a bucket and a local disk, so the same read is runnable here against local files:

```python
import batcher as bt

bt.from_pydict({"user_id": [1, 2], "status": ["active", "closed"]}).write.parquet("events/a.parquet")
bt.from_pydict({"user_id": [3], "status": ["active"]}).write.parquet("events/b.parquet")

ds = bt.read.parquet("events/*.parquet")
print(ds.filter(bt.col("status") == "active").sort("user_id").to_pydict())
# {'user_id': [1, 3], 'status': ['active', 'active']}
```

Swap `events/*.parquet` for `s3://bucket/events/*.parquet` and nothing else changes.

## Credentials

Credentials are read from the environment, following the conventions of each provider's SDK. They are the same variables the AWS, Google Cloud, and Azure tooling already uses. Set them before starting your process.

| Store | Environment variables and settings |
| --- | --- |
| S3 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` for temporary credentials, `AWS_REGION` / `AWS_DEFAULT_REGION`. Instance and role credentials are picked up automatically when set. |
| S3-compatible (MinIO, Ceph) | The S3 variables above, plus `AWS_ENDPOINT_URL=https://minio.internal:9000`, or `?endpoint_override=...` in the path. |
| Google Cloud Storage | `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service-account JSON, or workload-identity and application-default credentials. |
| Azure (`abfs`/`az`) | `AZURE_STORAGE_ACCOUNT_NAME` plus one of `AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_STORAGE_SAS_TOKEN`, or the AAD service-principal variables `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`. |
| HDFS | `hdfs://namenode:8020/path`, with `HADOOP_HOME` / `CLASSPATH` set so the native client and `core-site.xml` are found. |

```python
# docs: skip
import os

os.environ["AWS_ACCESS_KEY_ID"] = "..."
os.environ["AWS_SECRET_ACCESS_KEY"] = "..."
os.environ["AWS_REGION"] = "us-east-1"

import batcher as bt

ds = bt.read("s3://bucket/events.parquet")
```

For Delta tables read through delta-rs, credentials can also be passed explicitly as `storage_options` instead of through the environment. The keys there are delta-rs's own: `aws_access_key_id`, `aws_secret_access_key`, `azure_storage_account_key`, `google_service_account_token`, and so on.

```python
# docs: skip
ds = bt.read.delta(
    "s3://bucket/table",
    storage_options={"aws_access_key_id": "...", "aws_secret_access_key": "..."},
)
```

## On-prem and S3-compatible stores

`s3a://` and `gcs://` take the same native backends as `s3://` and `gs://`. They are aliases, not a slower path.

Point at your endpoint with `AWS_ENDPOINT_URL`, or per-path with an `endpoint_override` in the URI query string.

```bash
export AWS_ENDPOINT_URL=https://minio.internal:9000
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

Both variables are honored by the Python reader and the native reader. Per-path settings go in the URI query string, including the ones a self-hosted gateway usually needs:

| Option | Use |
| --- | --- |
| `endpoint_override`, `scheme` | Point at your gateway. `scheme=http` for plain HTTP. |
| `force_virtual_addressing` | `false` for path-style (`host/bucket/key`), the MinIO and Ceph default. |
| `access_key`, `secret_key`, `session_token` | Explicit credentials for one path. |
| `anonymous` | Public buckets, no signing. |
| `region`, `connect_timeout`, `request_timeout` | Region pinning and slow-link tuning. |
| `role_arn`, `session_name`, `external_id` | Assume a role for this path. |

```python
# docs: skip
ds = bt.read.parquet(
    "s3://bucket/data/*.parquet"
    "?endpoint_override=https://ceph.internal:8080&force_virtual_addressing=false"
)
```

An unrecognized option is an error naming the option, rather than being silently ignored.

The legacy Azure Blob schemes `wasb://` and `wasbs://` are **not** supported by either backend. Use the current `abfs://` / `abfss://` spelling, which is. A `wasb://` path fails with an error saying exactly that rather than being silently mis-read.

## Bring your own filesystem or credentials

Every reader and writer accepts two optional keywords, so you are never limited to environment variables or a URI query string.

`filesystem=` takes an already-constructed `pyarrow.fs.FileSystem` (or `PyFileSystem`), or an fsspec filesystem instance. Batcher uses it verbatim. Reach for it when you have a handle you have already authenticated, a mocked filesystem in a test, or a backend Batcher does not know.

`storage_options=` takes the portable credential dict the rest of the ecosystem speaks, including fsspec, delta-rs, Polars, and pandas: `key`, `secret`, and `endpoint_override` for S3, `account_name` and `account_key` for Azure, and so on. Prefer it over `filesystem=` for a distributed read. A plain dict rides the split to every worker unchanged, so each one resolves the same backend, whereas a live filesystem object only reaches a worker if it pickles.

```python
# docs: skip
import pyarrow.fs as pafs
import batcher as bt

fs = pafs.S3FileSystem(endpoint_override="https://minio.internal:9000",
                       access_key="...", secret_key="...")
ds = bt.read.parquet("s3://bucket/events/*.parquet", filesystem=fs)

# Or the portable dict, which also works across a Ray cluster:
ds = bt.read.parquet(
    "s3://bucket/events/*.parquet",
    storage_options={"endpoint_override": "https://minio.internal:9000",
                     "force_virtual_addressing": "false"},
)
```

## Writing to object storage

Write helpers take cloud paths as well. Combine with `partition_by` to lay out a partitioned dataset, and `distributed=True` to write across workers. Writes to an object store go straight to the destination, because a single PUT is atomic and leaves no truncated-file window. Local and HDFS writes use temp-then-rename for the same guarantee.

```python
# docs: skip
ds.write.parquet("s3://bucket/curated/events.parquet")
ds.write("s3://bucket/curated/events", fmt="parquet", partition_by=["region"])
```

## Working with a large dataset

Large cloud datasets are split into tasks so the driver never has to materialize a whole file. For distributed reads, the data plane moves Arrow batches directly between workers over Arrow Flight rather than through a scheduler's object store, which keeps per-node memory bounded.

```python
# docs: skip
ds = bt.read("s3://bucket/huge/*.parquet")
result = ds.group_by("region").agg(total=bt.col("amount").sum()).collect(
    distributed=True, num_workers=16
)
```

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`: the full reader
  and writer surface.
- {doc}`Lakehouse </user-guide/moving-data/lakehouse>`: Delta, Iceberg, and Hudi tables on object storage.
- {doc}`IO API </api/relational/io>`: the `bt.read` / `ds.write` reference.
- {doc}`/cookbook/io/sources_and_sinks`: which formats exist, and the objects behind them.
