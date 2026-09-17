# Cloud storage

This page covers reading and writing object storage: the schemes Batcher accepts, where credentials come from, and how to point at an on-prem or S3-compatible store. Object stores use the same API as local files. A path with a cloud scheme is recognized automatically, so only the path changes.

The accepted schemes are `s3://` / `s3a://`, `gs://` / `gcs://`, `az://` / `abfs://` / `abfss://`, and `hdfs://`, plus `file://` and bare local paths. Reading and writing both go through one `pyarrow.fs`-backed filesystem. Anything pyarrow does not implement natively falls back to fsspec behind the same interface.

## Prerequisites

Object-store access needs the cloud extra:

```bash
pip install 'batcher-engine[cloud]'
```

If a cloud scheme is used without the extra installed, the read fails with a message telling you to install it.

Most examples on this page need a real bucket and credentials, so their blocks are shown but not executed. The two that read and write local files run for real, and the prose says so where they appear.

## Read from object storage

{py:obj}`bt.read <batcher.read>` infers the format from the extension. The format-specific readers ({py:meth}`bt.read.parquet <batcher.api.io_namespace.reader.Reader.parquet>`, {py:meth}`bt.read.csv <batcher.api.io_namespace.reader.Reader.csv>`, {py:meth}`bt.read.json <batcher.api.io_namespace.reader.Reader.json>`) take the same cloud paths.

A glob works with `bt.read` and the typed readers alike. A `*` matches within one path segment only, so crossing directories in a Hive layout needs `**`. Reading a Hive layout that way returns every row without the partition columns, and Batcher warns about it. Point `bt.read.parquet` at the directory itself, or use `bt.read.parquet_dataset(...)`, when you need those columns back.

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

bt.from_pydict({"user_id": [1, 2], "status": ["active", "closed"]}).write.parquet(
    "events/a.parquet"
)
bt.from_pydict({"user_id": [3], "status": ["active"]}).write.parquet("events/b.parquet")

ds = bt.read.parquet("events/*.parquet")
print(ds.filter(bt.col("status") == "active").sort("user_id").to_pydict())
# {'user_id': [1, 3], 'status': ['active', 'active']}
```

Swap `events/*.parquet` for `s3://bucket/events/*.parquet` and nothing else changes.

## Credentials

Credentials are read from the environment, following the conventions of each provider's SDK. They are the same variables the AWS, Google Cloud, and Azure tooling already uses. Set them before starting your process.

The environment is the last place Batcher looks, not the first. A filesystem object you pass is used as it is and wins over `storage_options`. Otherwise a scheme pyarrow implements natively is built from the path's query string with your `storage_options` added, and only what those leave unset falls through to the provider SDK's own chain. A scheme pyarrow does not implement goes to fsspec, which takes `storage_options` as keyword arguments. The sections below cover each of these in turn.

![The order a path's filesystem and credentials come from. First, if filesystem= is passed, that pyarrow or fsspec filesystem is used verbatim and wins over storage_options. Second, if the scheme is not native to pyarrow, as with oss, cos, obs, oci, swift and lakefs, an fsspec backend is built with storage_options as keyword arguments. Third, for a native scheme such as s3, gs, abfs or hdfs and their aliases, the backend is built from the URI's query options with storage_options added, so explicit keys, an endpoint_override or a role_arn apply to that path only. Fourth, anything still unset comes from the provider SDK's own chain: environment variables such as AWS_ACCESS_KEY_ID, AZURE_STORAGE_* and GOOGLE_APPLICATION_CREDENTIALS, or instance and role identity. An env:, file: or cmd: value in storage_options resolves on the machine that opens the connection, so a distributed read ships the reference to each worker and never the secret.](/_static/diagrams/credential_resolution.svg)

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

Use `abfs://` or `abfss://` for Azure. The legacy Blob schemes `wasb://` and `wasbs://` aren't supported by either backend, and a `wasb://` path fails with an error that says so.

## Object stores outside the three hyperscalers

Alibaba OSS (`oss://`), Tencent COS (`cos://`, `cosn://`), Huawei OBS (`obs://`), Oracle Cloud Object Storage (`oci://`), OpenStack Swift (`swift://`) and lakeFS (`lakefs://`) are reached through fsspec, so they need that scheme's driver installed. Batcher treats them as object stores: a write publishes in place rather than through a temp-then-rename, which on an object store is a full server-side copy of the object and is not atomic anyway.

Credentials go in `storage_options`, which reaches an fsspec backend as **keyword arguments**. That is the vocabulary fsspec, delta-rs, Polars and pandas already speak, so a credential set that works with any of them works here unchanged.

```python
# docs: skip
ds = bt.read.parquet(
    "oss://bucket/events/*.parquet",
    storage_options={
        "key": "...",
        "secret": "env:OSS_SECRET",
        "endpoint": "oss-cn-hangzhou.aliyuncs.com",
    },
)
```

An option the backend does not accept is an error naming the option, rather than a connection that quietly used none of your settings.

Any value there may be an `env:`, `file:` or `cmd:` reference, resolved on the machine that opens the connection. A distributed read therefore ships the reference to each worker and never the secret. See {doc}`/user-guide/trust/secrets`.

## Bring your own filesystem or credentials

Every reader and writer accepts two optional keywords, so you aren't limited to environment variables or a URI query string.

`filesystem=` takes an already-constructed `pyarrow.fs.FileSystem` (or `PyFileSystem`), or an fsspec filesystem instance. Batcher uses it verbatim. Reach for it when you have a handle you have already authenticated, a mocked filesystem in a test, or a backend Batcher does not know.

`storage_options=` takes the portable credential dict the rest of the ecosystem speaks, including fsspec, delta-rs, Polars, and pandas: `key`, `secret`, and `endpoint_override` for S3, `account_name` and `account_key` for Azure, and so on. Prefer it over `filesystem=` for a distributed read. A plain dict rides the split to every worker unchanged, so each one resolves the same backend, whereas a live filesystem object only reaches a worker if it pickles.

```python
# docs: skip
import pyarrow.fs as pafs
import batcher as bt

fs = pafs.S3FileSystem(
    endpoint_override="https://minio.internal:9000", access_key="...", secret_key="..."
)
ds = bt.read.parquet("s3://bucket/events/*.parquet", filesystem=fs)

# Or the portable dict, which also works across a Ray cluster:
ds = bt.read.parquet(
    "s3://bucket/events/*.parquet",
    storage_options={
        "endpoint_override": "https://minio.internal:9000",
        "force_virtual_addressing": "false",
    },
)
```

## Write to object storage

Write helpers take cloud paths as well. Combine with `partition_by` to lay out a partitioned dataset, and `distributed=True` to write across workers. Writes to an object store go straight to the destination, because a single PUT is atomic and leaves no truncated-file window. Local and HDFS writes use temp-then-rename for the same guarantee.

```python
# docs: skip
ds.write.parquet("s3://bucket/curated/events.parquet")
ds.write("s3://bucket/curated/events", format="parquet", partition_by=["region"])
```

## Read a large dataset

Large cloud datasets are split into tasks so the driver never has to materialize a whole file. For distributed reads, the data plane moves Arrow batches directly between workers over Arrow Flight rather than through a scheduler's object store, which keeps per-node memory bounded.

```python
# docs: skip
ds = bt.read("s3://bucket/huge/*.parquet")
result = (
    ds.group_by("region")
    .agg(total=bt.col("amount").sum())
    .collect(distributed=True, num_workers=16)
)
```

## See also

- {doc}`Reading data </user-guide/moving-data/reading-data>` and {doc}`Writing data </user-guide/moving-data/writing-data>`: the full reader
  and writer surface.
- {doc}`Lakehouse </user-guide/moving-data/lakehouse>`: Delta, Iceberg, and Hudi tables on object storage.
- {doc}`IO API </api/relational/io>`: the {py:obj}`bt.read <batcher.read>` / {py:obj}`ds.write <batcher.Dataset.write>` reference.
- {doc}`/cookbook/io/sources_and_sinks`: which formats exist, and the objects behind them.
