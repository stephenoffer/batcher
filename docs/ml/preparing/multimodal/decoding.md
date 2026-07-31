# Fetching and decoding

Getting the bytes, and turning them into tensors.

## Fetch remote bytes

`ds.ml.download(url_column)` fetches the bytes at each URL or path into a binary
column, reading `s3://` / `gs://` / `az://` / `http(s)://` / local paths through the
shared filesystem resolver. Each batch's rows are fetched concurrently, and the stage
parallelizes across workers the way any operator does. `on_error="null"` turns a failed
fetch into a null so one bad URL does not fail the job.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/catalog.parquet")  # has a "url" column
with_bytes = ds.ml.download("url", output_column="bytes", max_concurrency=32)
```

`ds.ml.upload(data_column, directory)` is the counterpart. It writes a bytes column such
as decoded thumbnails or re-encoded media back to object storage, and appends the written
paths. Names come from a `name_column` or a content hash, and writes are concurrent and
atomic.

```python
# docs: skip
written = with_bytes.ml.upload("thumbnail", "s3://bucket/thumbs/", extension=".jpg")
```

## Decode images, audio, video

Multimodal readers list files and expose header metadata without decoding pixels.
Pass `decode=True` to append decoded tensors:

```python
# docs: skip
import batcher as bt

# Image bytes -> a (224, 224, 3) uint8 tensor column, decoded/resized in the engine.
images = bt.read.images("s3://bucket/images/", decode=True, size=(224, 224))

# Audio -> a list<float32> waveform column; video -> sampled (N, H, W, 3) frames.
audio = bt.read.audio("data/clips/", decode=True, sample_rate=16000)
video = bt.read.video("data/videos/", decode=True, size=(112, 112), num_frames=8)
```

Image and audio decode run natively in the engine. Audio goes through the pure-Rust
`symphonia` decoder, and an explicit `sample_rate` resamples natively too, with a sinc
resampler in the data plane. Only multi-channel output falls back to Python. Always
pass a `size`, or a batch of full-resolution frames can exhaust memory. Video decode uses
`PyAV` behind the `batcher-engine[video]` extra, one clip at a time, so a batch of large
clips never all co-resides. Keep `batch_size` small for multi-GB clips.

Decode is fast because it runs in the data plane, not a Python loop. Image decode uses
SIMD JPEG, including a DCT-scaled path for large frames feeding small model inputs, and
SIMD resize, fanned out per row across every core. The result crosses into a shaped
tensor column with no per-batch re-type step. On a 96-core node that decodes and resizes
**5,693 images per second**, which is **2.4x Daft**, and streams LiDAR point clouds at
**21,467 frames per second**. See
{doc}`Multimodal ingest benchmarks </benchmarks/multimodal-ingest>` and the reproducible
head-to-heads under `benchmarks/scenarios/`.

You can also decode inside a pipeline with the `.image` expression after a download:

```python
# docs: skip
import batcher as bt
from batcher import col

ds = bt.read.parquet("s3://bucket/catalog.parquet")
tensors = (
    ds.ml.download("url", output_column="bytes")
    .with_columns(image=col("bytes").image.to_tensor(224, 224))
)
```

The `.image.to_tensor(width, height)` expression decodes and resizes natively in the
engine, with no per-row Python and no model needed, so it runs here on a handful of
in-memory PNG bytes:

```python
import io

import numpy as np
from PIL import Image

import batcher as bt
from batcher import col

# Synthesize two tiny PNGs so the example needs no files.
raw = (np.random.default_rng(0).random((10, 12, 3)) * 255).astype("uint8")
buf = io.BytesIO()
Image.fromarray(raw).save(buf, format="PNG")
png = buf.getvalue()

ds = bt.from_pydict({"bytes": [png, png]})
decoded = ds.with_columns(image=col("bytes").image.to_tensor(8, 8)).collect()
print(decoded.num_rows, decoded.schema.field("image").type)
# 2 extension<arrow.fixed_shape_tensor[value_type=uint8, shape=[8,8,3]]>
```

Each row is now a flat `8 * 8 * 3 = 192`-element RGB block. `bt.read.images(...,
decode=True, size=(h, w))` re-types that flat result into a fixed-shape `(h, w, 3)`
tensor column so the shape travels with the data, as the tensor-column section below
describes. The bare expression leaves it flat for when you reshape it yourself in a
downstream `map_batches`.

`.image.resize(width, height)` is the other half of the pair. It decodes, resizes, then
**re-encodes to PNG bytes**, so the column stays a compact `binary` blob instead of
becoming a tensor. Reach for `resize` when you want to shrink payloads before a
shuffle, a spill, or a write. Use `to_tensor` when the next stage is a model.

```python
resized = ds.with_columns(small=col("bytes").image.resize(4, 4)).collect()
thumbnail = resized.column("small")[0].as_py()
print(resized.schema.field("small").type, thumbnail[:4] == b"\x89PNG")
# binary True
```

Two more expressions stay in the bytes-to-bytes lane. `.image.crop(x, y, width, height)`
cuts a named window out of each image, which is what a detection pipeline does with a
bounding box. `.image.encode(format)` rewrites the container without touching the pixels,
for normalizing a mixed-format corpus onto one codec or trading a PNG for a smaller JPEG.

```python
cut = ds.with_columns(
    region=col("bytes").image.crop(0, 0, 4, 4),
    as_jpeg=col("bytes").image.encode("jpeg"),
)
print(cut.select(d=col("region").image.decode()).to_pydict())
# {'d': [{'width': 4, 'height': 4, 'channels': 4, 'mode': 'RGBA'}]}
```

`.image.convert(mode)` completes the set: it changes only the channels, which is what
normalizing a corpus that mixes RGB and RGBA needs before a model that wants one of them.
The mode names are the ones `decode` reports, so a mode read off one goes straight into
the other, and grayscale uses the same Rec. 601 luma as `to_grayscale` and `dhash`.

`crop` clips at the edge rather than padding. `center_crop` pads, because it feeds a model
that needs a fixed input size; a cropped image is something a person or another tool will
look at, and inventing black pixels there would be inventing data. A window starting past
the image entirely yields null.

The audio counterpart is `.audio.to_waveform()`, which decodes an encoded clip and
averages its channels down to a single mono PCM signal. That is a `list<float>` per row,
the shape most audio models take as input. To feed a model that expects a fixed rate,
such as the 16 kHz Whisper and wav2vec want, `.audio.resample(16000)` decodes and
band-limit-resamples in the same native pass. `.video.decode()` is the video equivalent.

```python
import math
import struct
import wave

buf = io.BytesIO()
with wave.open(buf, "wb") as clip_writer:
    clip_writer.setnchannels(2)
    clip_writer.setsampwidth(2)
    clip_writer.setframerate(8000)
    samples = [int(3000 * math.sin(i / 10)) for i in range(16)]
    clip_writer.writeframes(b"".join(struct.pack("<hh", s, s) for s in samples))

signal = bt.from_pydict({"clip": [buf.getvalue()]}).with_columns(
    mono=col("clip").audio.to_waveform()
)
decoded_audio = signal.collect()
print(decoded_audio.schema.field("mono").type, len(decoded_audio.column("mono")[0].as_py()))
# list<item: float> 16
```

## Tensor columns

A column where every row is a same-shape `N`-dimensional tensor is stored as Arrow's
canonical fixed-shape-tensor type, so the shape travels with the data across the
engine boundary and converts to a correctly-shaped training tensor. `from_numpy` and
the NumPy reader build them for rows of rank 2 or higher:

```python
import batcher as bt
import numpy as np

imgs = np.zeros((4, 8, 8, 3), dtype=np.uint8)
ds = bt.from_numpy(imgs, column="image")
print(ds.collect().schema.field("image").type.shape)  # [8, 8, 3]
```

The `tensor_type`, `to_tensor_column`, `as_tensor_column`, and `is_tensor_column` helpers
in `batcher.io.formats.ml.tensor` build and classify these columns when you construct data
yourself.

When a tensor column reaches a `map_batches` model stage with `batch_format="numpy"`
or `"torch"`, the per-row tensors arrive **stacked** into one leading-batch array, a
`(batch, H, W, 3)` block, which is exactly the shape a vision model's forward pass
wants. No manual stacking or reshaping in the UDF.

A nested list column holds several small vectors per row, such as per-frame features or a
ragged batch of patches. `.list.flatten()` collapses it into one flat list per row,
removing a single level of nesting and keeping element order. Use it before a
downstream stage that wants one contiguous vector rather than a list of lists:

```python
import batcher as bt
from batcher import col

# Each row is a list of per-frame feature vectors; flatten to one vector per row.
frames = bt.from_pydict({"clip": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]})
flat = frames.select(vec=col("clip").list.flatten())
print(flat.to_pydict())
# {'vec': [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]]}
```
