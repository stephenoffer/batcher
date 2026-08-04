# Fetching and decoding

Getting the bytes, and turning them into tensors.

## Fetch remote bytes

{py:meth}`ds.ml.download(url_column) <batcher.api.dataset.ml.DatasetML.download>` fetches the bytes at each URL or path into a binary
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

{py:meth}`ds.ml.upload(data_column, directory) <batcher.api.dataset.ml.DatasetML.upload>` is the counterpart. It writes a bytes column such
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
pass a `size`, or a batch of full-resolution frames can exhaust memory.

Video decodes natively too, on an engine built with the `video` cargo feature, which links
the system FFmpeg. A build without it has no video codec to reach, so the same call falls
back to `PyAV` behind the `batcher-engine[video]` extra, one clip at a time. Both give the
same answer. {doc}`/ml/preparing/multimodal/video` covers which one you are running and
what changes. Keep `batch_size` small for multi-GB clips either way.

Decode is fast because it runs in the data plane, not a Python loop. Image decode uses
SIMD JPEG, including a DCT-scaled path for large frames feeding small model inputs, and
SIMD resize, fanned out per row across every core. The result crosses into a shaped
tensor column with no per-batch re-type step. On a 96-core node that decodes and resizes
**5,693 images per second**, which is **2.4x Daft**, and streams LiDAR point clouds at
**21,467 frames per second**. See
{doc}`Multimodal ingest benchmarks </benchmarks/results/multimodal-ingest>` and the reproducible
head-to-heads under `benchmarks/scenarios/`.

You can also decode inside a pipeline with the {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>` expression after a download:

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

The {py:meth}`.image.to_tensor(width, height) <batcher.plan.expr_ir.image._ImageNamespace.to_tensor>` expression decodes and resizes natively in the
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

{py:meth}`.image.resize(width, height) <batcher.plan.expr_ir.image._ImageNamespace.resize>` is the other half of the pair. It decodes, resizes, then
**re-encodes to PNG bytes**, so the column stays a compact `binary` blob instead of
becoming a tensor. Reach for `resize` when you want to shrink payloads before a
shuffle, a spill, or a write. Use {py:meth}`to_tensor <batcher.plan.expr_ir.image._ImageNamespace.to_tensor>` when the next stage is a model.

```python
resized = ds.with_columns(small=col("bytes").image.resize(4, 4)).collect()
thumbnail = resized.column("small")[0].as_py()
print(resized.schema.field("small").type, thumbnail[:4] == b"\x89PNG")
# binary True
```

Two more expressions stay in the bytes-to-bytes lane. {py:meth}`.image.crop(x, y, width, height) <batcher.plan.expr_ir.image._ImageNamespace.crop>`
cuts a named window out of each image, which is what a detection pipeline does with a
bounding box. {py:meth}`.image.encode(format) <batcher.plan.expr_ir.image._ImageNamespace.encode>` rewrites the container without touching the pixels,
for normalizing a mixed-format corpus onto one codec or trading a PNG for a smaller JPEG.

```python
cut = ds.with_columns(
    region=col("bytes").image.crop(0, 0, 4, 4),
    as_jpeg=col("bytes").image.encode("jpeg"),
)
print(cut.select(d=col("region").image.decode()).to_pydict())
# {'d': [{'width': 4, 'height': 4, 'channels': 4, 'mode': 'RGBA'}]}
```

{py:meth}`.image.convert(mode) <batcher.plan.expr_ir.image._ImageNamespace.convert>` completes the set: it changes only the channels, which is what
normalizing a corpus that mixes RGB and RGBA needs before a model that wants one of them.
The mode names are the ones `decode` reports, so a mode read off one goes straight into
the other, and grayscale uses the same Rec. 601 luma as `to_grayscale` and `dhash`.

`crop` clips at the edge rather than padding. `center_crop` pads, because it feeds a model
that needs a fixed input size; a cropped image is something a person or another tool will
look at, and inventing black pixels there would be inventing data. A window starting past
the image entirely yields null.

The audio counterpart is {py:meth}`.audio.to_waveform() <batcher.plan.expr_ir.audio._AudioNamespace.to_waveform>`, which decodes an encoded clip and
averages its channels down to a single mono PCM signal. That is a `list<float>` per row,
the shape most audio models take as input. To feed a model that expects a fixed rate,
such as the 16 kHz Whisper and wav2vec want, {py:meth}`.audio.resample(16000) <batcher.plan.expr_ir.audio._AudioNamespace.resample>` decodes and
band-limit-resamples in the same native pass. {py:meth}`.video.decode() <batcher.plan.expr_ir.video._VideoNamespace.decode>` is the video equivalent.

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

### What a bad row does

Every decode operation answers **null** for a row it cannot read: null input bytes,
truncated files, a codec the build does not have. The batch is never failed, because one
corrupt file in a scrape of millions is normal and losing the other millions to it is not.

That extends to a column of nothing but nulls, which is a shape a media pipeline produces
constantly — a download stage where every fetch failed, an outer join that matched nothing,
a partition filtered empty upstream. Such a column is typed `null` rather than `binary`,
and the decode operations read it as "all rows are null" rather than as a type mismatch.

An image too large to decode is a bad row too. The decoders carry a 512 MiB ceiling on the
pixel data one image may produce, so a "decompression bomb" — a small file declaring
enormous dimensions, such as the gigapixel scans and panoramas a real corpus contains
without malice — nulls its row rather than allocating. `.image.decode()` still reads its
header, so a corpus can be surveyed for oversized images before it is decoded:

```python
# docs: skip
from batcher import col

too_big = photos.filter(
    col("bytes").image.decode().struct.field("width")
    * col("bytes").image.decode().struct.field("height")
    > 100_000_000
)
```

Null is deliberate rather than a zero-filled tensor. Zeros are indistinguishable from a
legitimately black image or a silent clip, so they put blank samples into a training set
with nothing to detect them by. Count them instead:

```python
import batcher as bt
from batcher import col

photos = bt.from_pydict({"bytes": [b"", b""]})
undecodable = photos.filter(col("bytes").image.decode().is_null())
```

### Cutting out a detection's bounding box

`.image.crop(x, y, width, height)` takes its window from **columns**, not just constants,
which is what makes the central operation of a detection pipeline an engine operation: cut
the box a model predicted out of the frame it was predicted in. The boxes are data, one per
row, so a fixed window could never express it.

```python
# docs: skip
from batcher import col

patches = detections.with_columns(
    patch=col("frame").image.crop(col("box_x"), col("box_y"), col("box_w"), col("box_h"))
)
```

Constants and columns mix, so a fixed-size patch at a per-row position is
`crop(col("cx"), col("cy"), 64, 64)`.

The result is encoded bytes rather than a tensor, because rows genuinely differ in size.
Follow it with `letterbox` or `to_tensor` to get back to one shape a model can batch:

```python
# docs: skip
ready = patches.with_columns(x=col("patch").image.letterbox(224, 224))
```

A window that runs past an edge is clipped to what exists, rather than padded — a crop is
something you look at, and inventing black pixels there invents data. A window that is
null, negative, empty, or entirely outside the image nulls **that row only**. That last
part matters at corpus scale: boxes come from a model that sometimes declines to predict,
or from a join that sometimes matches nothing, and one unusable box should not cost the
batch it travelled in.

### Choosing how an image is resized

Three operations resize, and picking the wrong one is a mistake the output's shape cannot
show you:

| Call | Aspect ratio | Output | Use it for |
| --- | --- | --- | --- |
| `.image.to_tensor(w, h)` | stretched to fit | uint8 tensor | a classifier fed square crops |
| `.image.letterbox(w, h, fill=114)` | preserved, padded | uint8 tensor | object detection |
| `.image.thumbnail(max_size)` | preserved, never upscaled | PNG bytes | anything a person looks at |

`to_tensor` and `resize` take both dimensions, so they squash whatever is not already at
the target ratio. That is right for a classifier and wrong for a detector, because a
stretched image moves every box the model predicts off its object. `center_crop` is not the
answer either: it discards the border, which is where the missed detections are.

`letterbox` is the standard detection preprocessing. It scales the whole image to fit,
centres it on the canvas, and fills the remainder with a constant the model learns to
ignore. The default fill of `114` is the YOLO family's grey, so a model trained against
that preprocessing sees the padding it expects.

```python
# docs: skip
from batcher import col

# Two orientations, one canvas, so the rows batch together.
frames = photos.with_columns(x=col("bytes").image.letterbox(640, 640))
```

`thumbnail` scales so the longest side is `max_size` and hands back encoded bytes rather
than pixels, because its output is for review rather than for a model. It never upscales.
That split is the rule across the whole media surface: an operation returning an **encoded
still** takes a longest side and keeps the shape, one returning a **tensor** takes exact
dimensions.

```python
# docs: skip
from batcher import col

sheet = photos.select(uri=col("uri"), small=col("bytes").image.thumbnail(256))
```

## Tensor columns

A column where every row is a same-shape `N`-dimensional tensor is stored as Arrow's
canonical fixed-shape-tensor type, so the shape travels with the data across the
engine boundary and converts to a correctly-shaped training tensor. {py:func}`from_numpy <batcher.from_numpy>` and
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
ragged batch of patches. {py:meth}`.list.flatten() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.flatten>` collapses it into one flat list per row,
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
