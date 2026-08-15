# Reading databases, warehouses, and scientific formats

This page describes the `bt.read` entry points that reach a database, a warehouse, or a
scientific container format, rather than a file of rows.

```python
import batcher as bt
import pyarrow as pa
```

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
| `read.documents(path)` | PDF documents, one row per page as `{path, page, text}` | `[pdf]` |
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

## Web crawls (WARC)

{py:meth}`bt.read.warc(path) <batcher.api.io_namespace.reader.Reader.warc>` reads the format every web-scale crawler ships, Common Crawl
included. Each record becomes a row: the named WARC headers as typed columns, every other
header as JSON in `warc_headers`, and the payload as `warc_content`. `.warc.gz` is read
transparently, including the per-record gzip members a crawler normally writes.

A response record's payload is not the page. It is the whole HTTP exchange: the status
line, the response headers, a blank line, and only then the body. `http_status`,
`http_content_type` and `http_body` are that exchange taken apart, so a corpus-building
pass is a filter and a projection rather than a parse.

Use `http_body`, not `warc_content`. Handing the raw payload to `strip_html()` extracts
`HTTP/1.1 200 OK Content-Type: text/html Server: nginx` along with the prose, and nothing
about the result says so: the read succeeds, the column is text, and the prefix reads like
a sentence. Every length metric, embedding and dedup hash downstream then sees the crawl's
server banners.

```python
import os
import tempfile

# Build a small crawl so the example runs without a download.
def _record(kind, uri, payload):
    head = (
        f"WARC/1.0\r\nWARC-Type: {kind}\r\nWARC-Target-URI: {uri}\r\n"
        f"WARC-Date: 2024-03-15T13:45:30Z\r\nContent-Length: {len(payload)}\r\n\r\n"
    ).encode()
    return head + payload + b"\r\n\r\n"

def _response(status, body):
    return f"HTTP/1.1 {status}\r\nContent-Type: text/html\r\nServer: nginx\r\n\r\n".encode() + body

crawl = os.path.join(tempfile.mkdtemp(), "segment.warc")
with open(crawl, "wb") as fh:
    fh.write(_record("response", "https://example.com/a", _response("200 OK", b"<html><p>Hello</p></html>")))
    fh.write(_record("response", "https://example.com/b", _response("404 Not Found", b"<html>gone</html>")))
    fh.write(_record("request", "https://example.com/a", b"GET /a HTTP/1.1\r\n\r\n"))

pages = (
    bt.read.warc(crawl)
    .filter(bt.col("http_status") == 200)
    .select(
        url=bt.col("warc_target_uri"),
        text=bt.col("http_body").cast("string").str.strip_html(),
    )
)
print(pages.to_pydict())
# {'url': ['https://example.com/a'], 'text': ['Hello']}
```

`warc_content` is still there beside them, exactly as the crawler recorded it, because a
provenance or replay pass needs the whole exchange.

A WARC carries no index, so a reader cannot start in the middle of one: each file is one
split. Parallelism therefore comes from the number of files, which is how crawls are
published anyway, in many segments rather than one archive.

## Avro unions

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

## PDF documents

`read.documents` extracts a PDF corpus into `{path, page, text}`, one row per page. That
shape is what makes the rest of a document pipeline relational: chunking, embedding, and
retrieval all run as expressions over the `text` column, and `path` keeps a page attached
to the document it came from.

```python
# docs: skip
import batcher as bt
import pyarrow as pa
from batcher import col

pages = bt.read.documents("s3://bucket/reports/")
chunks = pages.with_columns(chunk=col("text").str.chunk(512, overlap=64, boundary="sentence"))
```

Two properties are worth knowing before you point it at a large corpus.

**Extraction is skipped when you do not ask for the text.** Laying a page out into reading
order is most of the cost of reading a PDF, so `select("path", "page")` and `count()` walk
the page tree and stop. Surveying a corpus is therefore cheap, and it is the right first
step: `group_by("path").agg(pages=col("page").count())` tells you the shape of what you
have without extracting a word.

**Encrypted documents need their password.** A PDF encrypted for *permissions* only, to
restrict printing or copying, carries an empty user password and opens without anything
extra. One with a real password takes `password=`:

```python
# docs: skip
locked = bt.read.documents("s3://bucket/contracts/", password="s3cret")
```

A page that will not extract is a page-level failure rather than a document-level one:
under `on_error="skip"` its text is null, distinguishable from a genuinely empty page, and
one bad page in a 900-page report does not cost the other 899. The document is recorded in
`corrupt_files()`.

## Point clouds and sensor arrays for robotics

`read.point_cloud` reads the native LiDAR and autonomous-driving point-cloud formats,
`.pcd` (PCL/ROS), `.ply`, and raw KITTI-style `.bin`, with no third-party dependency.
Each file is one frame, and every point becomes a row with a column per field, such as
`x`, `y`, `z`, and `intensity`, plus a `frame` column naming the source file. The cloud is
columnar, so the usual robotics preprocessing is a native engine operator: crop a region,
remove the ground plane, bin into voxels. A directory of sweeps stays separable with
{py:meth}`group_by("frame") <batcher.Dataset.group_by>`. A raw `.bin` buffer carries no schema, so pass its `columns=` layout.
That defaults to `x, y, z, intensity`.

Reading the schema does not read the points. PCD and PLY declare their fields in an ASCII
header, and a raw `.bin` has no header at all because you supplied the layout, so
`ds.schema` costs a few hundred bytes per file rather than a parse of every sweep in the
directory. That matters at corpus scale: an autonomous-driving dataset is thousands of
files, and each one is millions of points.

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

## The LiDAR preprocessing chain is native

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

## Robot and vehicle logs (MCAP)

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

## Vehicle measurements and CAN (MDF4)

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

## Hive-partitioned Parquet directories

`read.parquet` already recognizes a Hive layout: point it at a directory whose children
are `col=value` subdirectories and it recovers the partition columns from the layout and
prunes whole partitions when a filter matches the partition key. That makes a partitioned
write and its read symmetric: no column goes missing on the way back.

```python
import os
import tempfile

events_root = os.path.join(tempfile.mkdtemp(), "events")
bt.from_pydict({"value": [1, 2], "day": ["mon", "tue"]}).write.parquet(
    events_root, partition_by=["day"]
)
print(sorted(bt.read.parquet(events_root).columns))
# ['day', 'value']
```

A partition column's *type* is a different question from its presence, because a Hive path
segment carries only text, and the type has to be inferred back out of it. Strings,
integers, and dates come back as they went in. A date key is recognized only when *every*
directory value under that key is a full ``YYYY-MM-DD`` date, so a key that is a date in
one branch and something else in another stays text rather than failing to parse later:

```python
import datetime

day_root = tempfile.mkdtemp()
_ = bt.from_pydict(
    {"day": [datetime.date(2024, 1, 1), datetime.date(2024, 2, 10)], "n": [1, 2]}
).write.parquet(day_root, partition_by=["day"])
back = bt.read.parquet(day_root)
print(back.schema.field("day").type)
# date32[day]
print(back.filter(bt.col("day") > datetime.date(2024, 1, 1)).to_pydict()["n"])
# [2]
```

A float, boolean, or timestamp partition key still comes back as a **string**, since
nothing in the directory names distinguishes those from text. Cast the column explicitly
if you need the original type:

```python
# docs: skip
ds = bt.read.parquet("events/").with_columns(ratio=bt.col("ratio").cast("float64"))
```

An integer key loses zero-padding, because the padding is not part of the number: a tree
of ``month=01`` through ``month=12`` reads back as 1 through 12, and re-writing it would
produce unpadded directory names. Where the padding is part of the identifier rather than
a formatting choice, write the key as a string with a non-numeric marker in it, or use a
format that records the partition schema.

Partition by a string, an integer, or a date where you can. A table whose partition types
must survive exactly wants a format that records them: Delta and Iceberg keep the
partition schema in their logs, so they have nothing to infer.

Recovery is Parquet only. Partitioning a CSV, JSON, or Arrow directory writes the same
`col=value` layout, but reading it back returns every row *without* the partition columns,
with a warning saying so. Partition those formats only if you are willing to re-derive the
columns yourself, or write the values into the files as well as the path.

{py:meth}`parquet_dataset <batcher.api.io_namespace.reader.Reader.parquet_dataset>` names
that reader explicitly. Reach for it when the layout is unusual, or when a reader option
such as `n_rows` would otherwise keep the read on the flat reader:

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

## See also

- {doc}`/user-guide/moving-data/reading-data`: the in-memory constructors and file readers.
- {doc}`/user-guide/moving-data/cloud-storage`: paths, credentials and object stores.
- {doc}`/user-guide/moving-data/custom-connectors`: adding a source of your own.
