# Reading data

A pipeline starts by building a `Dataset` from a source. Sources come in two groups:
in-memory constructors, which wrap data already in the process, and path readers,
which load from disk or object storage. Both are lazy.

## In-memory constructors

These wrap data the process already holds, so they need no files and no credentials. They
are what the rest of the documentation uses for its runnable examples.

### From a column dict

`from_pydict` takes a column-oriented dictionary. This is the constructor used
throughout the docs because it needs no files.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "id": [1, 2, 3],
        "name": ["alice", "bob", "carol"],
        "value": [100, 200, 300],
    }
)
print(ds.to_pydict())
# {'id': [1, 2, 3], 'name': ['alice', 'bob', 'carol'], 'value': [100, 200, 300]}
```

### From arrow

`from_arrow` wraps a `pyarrow.Table`, a `RecordBatch`, or a list of batches with
no copy of the underlying buffers.

```python
import pyarrow as pa

table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
ds = bt.from_arrow(table)
print(ds.to_pydict())
# {'x': [1, 2, 3], 'y': ['a', 'b', 'c']}
```

### From a streaming factory

`from_batches` builds a streaming source from a callable that returns a fresh
iterator of Arrow batches each time it is called, plus the schema those batches
follow.

```python
schema = pa.schema([("n", pa.int64())])


def make_batches():
    for start in (0, 3):
        yield pa.record_batch({"n": [start, start + 1, start + 2]}, schema=schema)


ds = bt.from_batches(make_batches, schema)
print(ds.to_pydict())
# {'n': [0, 1, 2, 3, 4, 5]}
```

### From items and generators

`from_items` builds a `Dataset` from a Python list, one row per item.
A dict item expands to columns, and a scalar becomes a single `item` column. `date_range`
generates a calendar dimension, the date-typed sibling of `range`.

```python
print(bt.from_items([1, 2, 3]).to_pydict())
# {'item': [1, 2, 3]}
print(bt.date_range("2024-01-01", "2024-01-03").count())
# 3
```

### From other frameworks

Adapters convert a frame from another library into a `Dataset`:
`from_pandas`, `from_polars`, `from_numpy`, `from_spark`, `from_dask`,
`from_huggingface`, `from_torch`, and `from_tf`. They require the corresponding
library to be installed.

```python
# docs: skip
import pandas as pd

ds = bt.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
```

## File and path readers

File readers take a local path, a glob pattern, or an object-store URL. They need
real files, so the examples below are shown but not executed here.

{py:obj}`bt.read(path, format=None, **opts) <batcher.read>` detects the format from the path when
`format` is omitted. The format-specific helpers `read.parquet`, `read.csv`, `read.json`,
and `read.table` accept the same path and option style.

```python
# docs: skip
ds = bt.read("data/events.parquet")          # format inferred from extension
ds = bt.read("data/*.parquet")               # glob across many files
ds = bt.read("s3://bucket/events.parquet")   # object storage (needs [cloud])
```

```python
# docs: skip
ds = bt.read.parquet("data/events.parquet")
ds = bt.read.csv("data/events.csv")
ds = bt.read.json("data/events.jsonl")
```

Many more readers cover the columnar and table formats, and the multimodal ones:
`read.orc`, `read.arrow`, `read.avro`, `read.lance`, `read.delta`, `read.iceberg`,
`read.hudi`, `read.sql`, `read.snowflake`, `read.bigquery`, `read.kafka`,
`read.images`, `read.audio`, and `read.video`. Each takes a path or connection plus
format-specific options.

```python
# docs: skip
ds = bt.read.delta("s3://lake/events")
frames = bt.read.images("s3://bucket/photos/*.jpg")
```

## Databases, warehouses, and specialized formats

The same `bt.read` namespace also reaches databases and warehouses, plus a handful
of scientific container formats. They share one shape: `bt.read.<name>(path_or_uri,
**opts)` hands back a lazy `Dataset`, and nothing is fetched until a terminal op
runs. The database connectors (`mongo`, `cassandra`, `dynamodb`, `elasticsearch`)
take their connection as keyword options rather than a path.

| Reader | Reads | Needs |
| --- | --- | --- |
| `read.parquet_dataset(dir)` | A Hive-partitioned Parquet directory, partition columns recovered from the layout | nothing extra |
| `read.webdataset(path)` | WebDataset `.tar` shards, one row per sample | nothing extra |
| `read.warc(path)` | Web-archive (WARC) files, one row per crawl record; `.warc.gz` read transparently | nothing extra |
| `read.tfrecord(path)` | TFRecord files from Waymo, TFDS, or RLDS, one row per record | nothing extra |
| `read.excel(path)` | Excel workbooks via python-calamine | `[excel]` |
| `read.hdf5(path)` | HDF5 files, datasets as columns | `[hdf5]` |
| `read.zarr(path)` | A Zarr store of chunked n-dimensional arrays | `[zarr]` |
| `read.numpy(path)` | NumPy `.npy` and `.npz` files as tensor rows | nothing extra |
| `read.point_cloud(path)` | LiDAR point-cloud files in `.pcd`, `.ply`, or raw `.bin`, one row per point | nothing extra |
| `read.mcap(path)` | MCAP robot and vehicle logs from ROS 2 or ADAS, one row per message | `[robotics]` |
| `read.mdf(path)` | ASAM MDF4 vehicle measurements over CAN/LIN and sensors, one row per sample | `[robotics]` |
| `read.delta_sharing(url)` | A Delta Sharing `<profile>#<share>.<schema>.<table>` | `[delta-sharing]` |
| `read.sql(query, uri=...)` | Any ADBC/FlightSQL database addressed by a connection URI | driver + reachable DB |
| `read.clickhouse(query)` | A ClickHouse query result over the Arrow-native interface | a running ClickHouse |
| `read.databricks(table)` | A Databricks or Unity Catalog table, with credential vending | a Databricks workspace |
| `read.mongo(...)` | A MongoDB collection via pymongoarrow | a running MongoDB |
| `read.cassandra(...)` | A Cassandra/Scylla table, fanned out across token ranges | a running Cassandra |
| `read.dynamodb(...)` | A DynamoDB table via native parallel scan segments | AWS DynamoDB |
| `read.elasticsearch(...)` | An Elasticsearch index via ES\|QL Arrow output | a running Elasticsearch |

Each connector needs its service reachable (or its optional extra installed), so
these are shown but not executed:

```python
# docs: skip
rows = bt.read.sql("SELECT * FROM events", uri="postgresql://localhost:5432/app")
sales = bt.read.clickhouse("SELECT * FROM sales", host="localhost")
orders = bt.read.databricks("main.sales.orders")
events = bt.read.mongo(uri="mongodb://localhost:27017", database="app", collection="events")
shared = bt.read.delta_sharing("config.share#share.schema.table")
grids = bt.read.zarr("s3://bucket/array.zarr")
```

### Web crawls (WARC)

`bt.read.warc(path)` reads the format every web-scale crawler ships, Common Crawl
included. Each record becomes a row: the named WARC headers as typed columns, every other
header as JSON in `warc_headers`, and the payload as `warc_content`. `.warc.gz` is read
transparently, including the per-record gzip members a crawler normally writes.

That makes a corpus-building pass a plain query. Filter to the response records, decode
the payload, and hand it to the string accessors.

```python
import gzip
import os
import tempfile

# Build a two-record crawl so the example runs without a download.
def _record(kind, uri, body):
    head = (
        f"WARC/1.0\r\nWARC-Type: {kind}\r\nWARC-Target-URI: {uri}\r\n"
        f"WARC-Date: 2024-03-15T13:45:30Z\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode()
    return head + body + b"\r\n\r\n"

crawl = os.path.join(tempfile.mkdtemp(), "segment.warc")
with open(crawl, "wb") as fh:
    fh.write(_record("response", "https://example.com/a", b"<html><p>Hello</p></html>"))
    fh.write(_record("request", "https://example.com/a", b"GET /a HTTP/1.1"))

pages = (
    bt.read.warc(crawl)
    .filter(bt.col("warc_type") == "response")
    .select(
        url=bt.col("warc_target_uri"),
        text=bt.col("warc_content").cast("string").str.strip_html(),
    )
)
print(pages.to_pydict())
# {'url': ['https://example.com/a'], 'text': ['Hello']}
```

A WARC carries no index, so a reader cannot start in the middle of one: each file is one
split. Parallelism therefore comes from the number of files, which is how crawls are
published anyway, in many segments rather than one archive.

### Avro unions

Most Avro fields are written as `["null", T]`, which is Avro's spelling of "a nullable
`T`". `read.avro` reads those as exactly that, a nullable Arrow column of `T`.

A union with **more than one** real branch, such as `["null", "long", "string"]`, has no
single Arrow type. Rather than pick one branch and lose the rest, Batcher reads it as a
struct with one nullable `memberN` field per branch, in declaration order, exactly one of
which is set on any given row. This is the same mapping Spark's Avro reader uses.

```python
import fastavro
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "events.avro")
schema = fastavro.parse_schema(
    {"type": "record", "name": "E", "fields": [{"name": "v", "type": ["null", "long", "string"]}]}
)
with open(path, "wb") as fh:
    fastavro.writer(fh, schema, [{"v": 1}, {"v": "retry"}, {"v": None}])

events = bt.read.avro(path)
print(events.schema.field("v").type)
```

Pick a branch out with the usual struct accessor, so `col("v").struct.field("member1")` is
the string arm. Or `coalesce` the arms together once you know they are compatible.

### Point clouds and sensor arrays for robotics

`read.point_cloud` reads the native LiDAR and autonomous-driving point-cloud formats,
`.pcd` (PCL/ROS), `.ply`, and raw KITTI-style `.bin`, with no third-party dependency.
Each file is one frame, and every point becomes a row with a column per field, such as
`x`, `y`, `z`, and `intensity`, plus a `frame` column naming the source file. The cloud is
columnar, so the usual robotics preprocessing is a native engine operator: crop a region,
remove the ground plane, bin into voxels. A directory of sweeps stays separable with
`group_by("frame")`. A raw `.bin` buffer carries no schema, so pass its `columns=` layout.
That defaults to `x, y, z, intensity`.

```python
import os
import tempfile
import numpy as np

path = os.path.join(tempfile.mkdtemp(), "0000.bin")  # a KITTI-style Velodyne sweep
np.array([[3.0, 1.0, -1.8, 0.2], [4.0, 2.0, 0.5, 0.9]], dtype=np.float32).tofile(path)

sweep = bt.read.point_cloud(path)
# Ground-plane removal is a filter on z, done natively and in parallel.
above_ground = sweep.filter(bt.col("z") > -1.5)
print(above_ground.select("x", "z").to_pydict())
# {'x': [4.0], 'z': [0.5]}
```

### The LiDAR preprocessing chain is native

Because the cloud is columnar, the standard per-frame preprocessing is engine operators
end to end. No Python runs per point, and one lazy plan fuses the stages rather than
materializing a cloud for each. Two of the idioms are worth spelling out because they are
not obvious:

```python
# docs: skip
from batcher import col, lit

roi = sweep.filter((col("x") > 0) & (col("x") < 30) & (col("y").abs() < 10))
above_ground = roi.filter(col("z") > -1.5)

# Voxel downsample: snap each point to a grid cell with `floor`, then average the points
# that land in the same cell. This is the canonical LiDAR reduction.
VOXEL = 0.5
downsampled = (
    above_ground.with_columns(
        vx=(col("x") / VOXEL).floor(),
        vy=(col("y") / VOXEL).floor(),
        vz=(col("z") / VOXEL).floor(),
    )
    .group_by("vx", "vy", "vz")
    .agg(x=col("x").mean(), y=col("y").mean(), z=col("z").mean(), n=col("x").count())
)

# Ego frame -> world frame: a rigid transform is three projections, given a pose.
import math
yaw, tx, ty = 0.3, 10.0, 20.0
cos, sin = math.cos(yaw), math.sin(yaw)
world = above_ground.with_columns(
    wx=col("x") * lit(cos) - col("y") * lit(sin) + lit(tx),
    wy=col("x") * lit(sin) + col("y") * lit(cos) + lit(ty),
    wz=col("z"),
)

# Range gating is plain arithmetic on the coordinates.
near = above_ground.with_columns(
    rho=(col("x") ** 2 + col("y") ** 2 + col("z") ** 2).sqrt()
).filter(col("rho") < 25)
```

### Robot and vehicle logs (MCAP)

`read.mcap` reads the container format ROS 2 records into and autonomous-driving stacks
exchange. One log multiplexes every sensor as timestamped messages on named *topics*,
covering camera, LiDAR, radar, CAN, GPS/IMU, and planner state, so **a row is a message**:
`{topic, log_time, publish_time, sequence, schema_name, message_encoding, data}`.

Payloads stay encoded in `data`, so a query that wants `/gps` never pays to deserialize
`/camera`. MCAP is indexed, and the reader uses that. `count()` and the `log_time` bounds
come from the summary without reading a message, and a filter on `topic` or `log_time` is
pushed into the reader as a *seek*. That is the difference between reading five seconds of
a two-hour drive and reading the whole drive. Naming `topics=` up front does the same
thing explicitly.

Topic discovery is a property of the *source*, not of the `Dataset` it produces.
`MCAPSource.topics()` reads only the file summary, so it costs no message decode. The
multi-sensor idiom follows: a filter per topic and then an as-of join, which is how
sensors sampled at different rates get onto a common clock.

```python
# docs: skip
from batcher.io.formats import MCAPSource

print(MCAPSource("s3://drives/2026-07-18/").topics())
# ['/camera/front', '/gps', '/imu', '/lidar/top']

log = bt.read.mcap("s3://drives/2026-07-18/")
imu = log.filter(bt.col("topic") == "/imu").select("log_time", imu_seq=bt.col("sequence"))
lidar = log.filter(bt.col("topic") == "/lidar/top").select("log_time", sweep=bt.col("sequence"))

# Put the 100 Hz IMU onto each 10 Hz LiDAR sweep: one row per sweep, carrying the most
# recent IMU sample at or before it.
aligned = lidar.join_asof(imu, on="log_time")
```

A drive-day directory usually contains one recording that was cut short; pass
`on_error="skip"` to drop it and keep the rest, then check `corrupt_files()` to see what
was dropped.

### Vehicle measurements and CAN (MDF4)

`read.mdf` reads ASAM MDF4 (`.mf4`), what automotive OEMs and test fleets log CAN/LIN
signals and sensor channels to. A file holds several *channel groups*, each with **its own
sampling raster**, so there is no single wide table for it. Powertrain might sample at
100 Hz while chassis samples at 4 Hz. The reader emits **long format**, one row per
sample: `{signal, timestamp, value, unit}`. That gives one schema whatever the file's
raster count, and keeps resampling an explicit choice you make rather than something the
reader did silently.

`timestamp` is absolute, which is what makes the cross-format ADAS query work: a CAN
measurement and an MCAP log from the same drive align on one clock.

```python
# docs: skip
can = bt.read.mdf("s3://fleet/drive.mf4", signals=["VehicleSpeed"])
can = can.select("timestamp", speed=bt.col("value"))

lidar = (bt.read.mcap("s3://fleet/drive.mcap")
         .filter(bt.col("topic") == "/lidar/top")
         .select(timestamp=bt.col("log_time"), sweep=bt.col("sequence")))

# Attach the vehicle speed to every LiDAR sweep, then keep the hard-braking ones.
# This is scenario extraction across two file formats.
fused = lidar.join_asof(can, on="timestamp")
braking = fused.filter(bt.col("speed") < 20)
```

`MDFSource.signals()` and `MCAPSource.topics()` list what a file carries without reading
data. That is the discovery step before a query names the handful of channels it wants.

### Hive-partitioned Parquet directories

Partitioned Parquet needs only local files, so it runs end to end. `parquet_dataset`
recovers the partition columns from the Hive directory layout and prunes whole
partitions when a filter matches the partition key:

```python
import os
import tempfile
import pyarrow.parquet as pq

root = os.path.join(tempfile.mkdtemp(), "events")
pq.write_to_dataset(
    pa.table({"value": [1, 2], "day": ["mon", "tue"]}),
    root,
    partition_cols=["day"],
)
ds = bt.read.parquet_dataset(root)
print(ds.select("value").sort("value").to_pydict())
# {'value': [1, 2]}
```

## What you get back

Every constructor hands back a lazy `Dataset`. Inspect the column names with the
`columns` property. Nothing is read until a terminal operation runs.

```python
people = bt.from_pydict({"id": [1, 2], "name": ["alice", "bob"]})
print(people.columns)
# ['id', 'name']
```

The reads above run on the compiled Rust data plane. `engine_version` reports which
engine build is loaded, distinct from the Python package version:

```python
print(isinstance(bt.engine_version(), str))
# True
```

## See also

- {doc}`Transformations </user-guide/transform/transformations>`: reshape and derive columns.
- {doc}`Filtering </user-guide/transform/filtering>`: select rows, drop duplicates.
- {doc}`Lakehouse tables </user-guide/moving-data/lakehouse>`: read Delta and Iceberg tables, and travel back
  through their versions.
- {doc}`Data quality </user-guide/trust/data-quality>`: validate inputs as they arrive.
- {doc}`IO API </api/relational/io>`: the full `bt.read` reader reference.
- {doc}`Agent skills </agents/index>`: `read-and-write-data` covers picking a reader or
  sink, cloud paths, globs, schema evolution, and error tolerance.
- {doc}`/cookbook/io/index`: 6 runnable recipes for readers, writers, and the registries.
