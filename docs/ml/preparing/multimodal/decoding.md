# Fetching and decoding

This page covers the first two links of every media pipeline: fetching the bytes a URL or path points at, and decoding those bytes into tensor columns a model can read.

## Fetch remote bytes

{py:meth}`ds.ml.download(url_column) <batcher.api.dataset.ml.DatasetML.download>` fetches the bytes at each URL or path into a binary column. It reads `s3://`, `gs://`, `az://`, `http(s)://` and local paths through the shared filesystem resolver. Each batch's rows are fetched concurrently, and the stage parallelizes across workers the way any operator does. `on_error="null"` turns a failed fetch into a null, so one bad URL doesn't fail the job.

```python
# docs: skip
import batcher as bt

ds = bt.read.parquet("s3://bucket/catalog.parquet")  # has a "url" column
with_bytes = ds.ml.download("url", output_column="bytes", max_concurrency=32)
```

{py:meth}`ds.ml.upload(data_column, directory) <batcher.api.dataset.ml.DatasetML.upload>` goes the other way. It writes a bytes column, such as decoded thumbnails or re-encoded media, back to object storage and appends the written paths. Names come from a `name_column` or a content hash, and writes are concurrent and atomic.

```python
# docs: skip
written = with_bytes.ml.upload("thumbnail", "s3://bucket/thumbs/", extension=".jpg")
```

## Decode with a reader

Multimodal readers list files and expose header metadata without decoding pixels. Pass `decode=True` to append decoded tensors:

```python
# docs: skip
import batcher as bt

# Image bytes -> a (224, 224, 3) uint8 tensor column, decoded/resized in the engine.
images = bt.read.images("s3://bucket/images/", decode=True, size=(224, 224))

# Audio -> a list<float32> waveform column; video -> sampled (N, H, W, 3) frames.
audio = bt.read.audio("data/clips/", decode=True, sample_rate=16000)
video = bt.read.video("data/videos/", decode=True, size=(112, 112), num_frames=8)
```

Always pass a `size` for images, or a batch of full-resolution frames can exhaust memory. `size=(height, width)` says what shape every row must be, and `fit` says how an image with a different aspect ratio gets there. All three modes in the following table produce the same fixed-shape column, so nothing downstream can tell them apart. That's why the choice is worth making on purpose:

| `fit` | What happens to a mismatched ratio | Reach for it when |
|---|---|---|
| `"stretch"` (default) | squashed to the exact size | the model was trained the same way |
| `"letterbox"` | scaled inside the box, remainder padded gray | detection: stretching moves every predicted box off its object |
| `"center_crop"` | center kept at native resolution, border discarded | the subject is centered and the border is clutter |

```python
# docs: skip
import batcher as bt

frames = bt.read.images("s3://bucket/frames/", size=(640, 640), fit="letterbox")
```

Stretch is the default because it's right for a classifier. For anything else it's a *silent* distortion. The decode succeeds, the tensor is the right size, and a detector trained on letterboxed frames predicts every box in the wrong place with no error anywhere.

Image and audio decode run natively in the engine. Audio goes through the pure-Rust `symphonia` decoder, and a `sample_rate` resamples natively too, with a sinc resampler in the data plane. The one audio path that falls back to Python is multi-channel output, `batcher.ml.decode.audio_dataset(mono=False)`. Video decodes natively on an engine built with the `video` cargo feature, which links the system FFmpeg. A build without it falls back to `PyAV` behind the `batcher-engine[video]` extra. Both give the same answer, and {doc}`/ml/preparing/multimodal/video` covers how to tell which one you're running.

Native decode is fast because no Python loop sits in it. Image decode uses SIMD JPEG, including a DCT-scaled path for large frames feeding small model inputs, plus SIMD resize, fanned out per row across every core. The result becomes a shaped tensor column with no per-batch re-type step. On an idle 96-core node that decoded and resized 5,693 images per second, 2.4x Daft, and streamed LiDAR point clouds at 21,467 frames per second. {doc}`Multimodal ingest benchmarks </benchmarks/results/multimodal-ingest>` has the full results, and the reproducible head-to-heads live under `benchmarks/scenarios/`.

## Decode inside a pipeline

Bytes that arrive from a download or a table column decode with expressions instead of a reader. You can decode right after a download with the {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>` accessor:

```python
# docs: skip
import batcher as bt
from batcher import col

ds = bt.read.parquet("s3://bucket/catalog.parquet")
tensors = ds.ml.download("url", output_column="bytes").with_columns(
    image=col("bytes").image.to_tensor(224, 224)
)
```

{py:meth}`.image.to_tensor(width, height) <batcher.plan.expr_ir.image._ImageNamespace.to_tensor>` decodes and resizes natively, with no per-row Python and no model, so it runs here on a handful of in-memory PNG bytes:

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

Each row holds 8x8x3 RGB values tagged as a fixed-shape `(8, 8, 3)` tensor, so the shape travels with the data. {ref}`tensor-columns` below covers what that buys you.

{py:meth}`.image.resize(width, height) <batcher.plan.expr_ir.image._ImageNamespace.resize>` is the other half of the pair. It decodes, resizes, then **re-encodes**, to PNG by default, so the column stays a compact `binary` blob instead of becoming a tensor. Use `resize` to shrink payloads before a shuffle, a spill, or a write, and {py:meth}`to_tensor <batcher.plan.expr_ir.image._ImageNamespace.to_tensor>` when the next stage is a model.

```python
resized = ds.with_columns(small=col("bytes").image.resize(4, 4)).collect()
thumbnail = resized.column("small")[0].as_py()
print(resized.schema.field("small").type, thumbnail[:4] == b"\x89PNG")
# binary True
```

Two more expressions stay in the bytes-to-bytes lane. {py:meth}`.image.crop(x, y, width, height) <batcher.plan.expr_ir.image._ImageNamespace.crop>` cuts a named window out of each image. {py:meth}`.image.encode(format) <batcher.plan.expr_ir.image._ImageNamespace.encode>` rewrites the container without touching the pixels, for moving a mixed-format corpus onto one codec or trading a PNG for a smaller JPEG.

```python
cut = ds.with_columns(
    region=col("bytes").image.crop(0, 0, 4, 4),
    as_jpeg=col("bytes").image.encode("jpeg"),
)
print(cut.select(d=col("region").image.decode()).to_pydict())
# {'d': [{'width': 4, 'height': 4, 'channels': 4, 'mode': 'RGBA'}]}
```

{py:meth}`.image.convert(mode) <batcher.plan.expr_ir.image._ImageNamespace.convert>` changes only the channels, which is what a corpus mixing RGB and RGBA needs before a model that wants one of them. The mode names are the ones `decode` reports, so a mode read from one goes straight into the other. Grayscale uses the same Rec. 601 luma as `to_grayscale` and `dhash`.

The audio counterpart is {py:meth}`.audio.to_waveform() <batcher.plan.expr_ir.audio._AudioNamespace.to_waveform>`. It decodes an encoded clip and averages its channels into one mono PCM signal, a `list<float>` per row, which is the input most audio models take. For a model that expects a fixed rate, such as the 16 kHz that Whisper and wav2vec want, {py:meth}`.audio.resample(16000) <batcher.plan.expr_ir.audio._AudioNamespace.resample>` decodes and band-limit-resamples in the same native pass. {py:meth}`.video.decode() <batcher.plan.expr_ir.video._VideoNamespace.decode>` is the video equivalent of a header decode.

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

### Cutting out a detection's bounding box

`.image.crop(x, y, width, height)` takes its window from **columns** as well as constants. That makes the central operation of a detection pipeline an engine operation: cut the box a model predicted out of the frame it was predicted in. The boxes are data, one per row, which a fixed window could never express.

```python
# docs: skip
from batcher import col

patches = detections.with_columns(
    patch=col("frame").image.crop(col("box_x"), col("box_y"), col("box_w"), col("box_h"))
)
```

Constants and columns mix, so a fixed-size patch at a per-row position is `crop(col("cx"), col("cy"), 64, 64)`.

The result is encoded bytes rather than a tensor, because rows genuinely differ in size. Follow it with `letterbox` or `to_tensor` to get back to one shape a model can batch:

```python
# docs: skip
ready = patches.with_columns(x=col("patch").image.letterbox(224, 224))
```

A window that runs past an edge is clipped to what exists, not padded. `center_crop` pads, because it feeds a model that needs a fixed input size, but a crop is something a person or another tool looks at, and black pixels there would be invented data. A window that is null, negative, empty, or entirely outside the image nulls **that row only**. Boxes come from a model that sometimes declines to predict, or a join that sometimes matches nothing, and one unusable box shouldn't cost the batch it traveled in.

### Choosing how an image is resized

Three operations resize, and the output's shape won't reveal a wrong choice. The following table compares them:

| Call | Aspect ratio | Output | Use it for |
| --- | --- | --- | --- |
| `.image.to_tensor(w, h)` | stretched to fit | uint8 tensor | a classifier fed square crops |
| `.image.letterbox(w, h, fill=114)` | preserved, padded | uint8 tensor | object detection |
| `.image.thumbnail(max_size)` | preserved, never upscaled | PNG bytes | anything a person looks at |

`to_tensor` and `resize` take both dimensions, so they squash whatever isn't already at the target ratio. That's right for a classifier and wrong for a detector, because a stretched image moves every predicted box off its object. `center_crop` doesn't fix it either: it discards the border, which is where the missed detections are.

`letterbox` is the standard detection preprocessing. It scales the whole image to fit, centers it on the canvas, and fills the remainder with a constant the model learns to ignore. The default fill of `114` is the YOLO family's gray, so a model trained with that preprocessing sees the padding it expects.

```python
# docs: skip
from batcher import col

# Two orientations, one canvas, so the rows batch together.
frames = photos.with_columns(x=col("bytes").image.letterbox(640, 640))
```

`thumbnail` scales so the longest side is `max_size`, never upscales, and hands back encoded bytes, because its output is for review. {doc}`/ml/preparing/multimodal/video` explains why every still-returning operation takes a longest side while every tensor-returning one takes exact dimensions.

```python
# docs: skip
from batcher import col

sheet = photos.select(uri=col("uri"), small=col("bytes").image.thumbnail(256))
```

## Identify bytes that didn't come from a file

The media and blob readers give you a `mime` column, sniffed from each file's leading bytes rather than its name. Bytes that arrive any other way have no such column and no filename to guess from: a download, a blob column in a Parquet table, a payload pulled out of an archive. `.str.mime_type()` reads the same magic-number table as an expression:

```python
import batcher as bt
from batcher import col

blobs = bt.from_pydict({"b": [b"\x89PNG\r\n\x1a\n" + bytes(8), b"unknown"]})
print(blobs.select(m=col("b").str.mime_type()).to_pydict())
# {'m': ['image/png', None]}
```

Routing a mixed corpus becomes a filter instead of a UDF:

```python
# docs: skip
typed = downloaded.with_columns(kind=col("bytes").str.mime_type())
images = typed.filter(col("kind").str.starts_with("image/"))
video = typed.filter(col("kind").str.starts_with("video/"))
```

Unrecognized bytes are **null**, not `application/octet-stream`. An expression sees only bytes, so it has nothing left to try. A reader still has a filename, and turns the same null into a guess from the extension before giving up. Keeping the two distinct lets you `coalesce` in whatever you know instead.

## What a bad row does

Every decode operation answers null for a row it can't read: null input bytes, a truncated file, a codec the build doesn't have. The batch never fails, because one corrupt file in a scrape of millions is normal, and losing the other millions to it isn't.

That extends to a column of nothing but nulls, a shape media pipelines produce constantly: a download stage where every fetch failed, an outer join that matched nothing, a partition filtered empty upstream. Such a column is typed `null` rather than `binary`, and the decode operations read it as all-null rows, not as a type mismatch.

An image too large to decode is a bad row too. The decoders cap the pixel data one image may produce at 512 MiB, so a small file declaring enormous dimensions nulls its row instead of allocating. Real corpora contain such files without any malice, in the form of gigapixel scans and panoramas. `.image.decode()` still reads the header, so you can survey a corpus for oversized images before decoding it:

```python
# docs: skip
from batcher import col

too_big = photos.filter(
    col("bytes").image.decode().struct.field("width")
    * col("bytes").image.decode().struct.field("height")
    > 100_000_000
)
```

Null is deliberate, not a zero-filled tensor. Zeros are indistinguishable from a legitimately black image or a silent clip, so they'd put blank samples into a training set with nothing to detect them by. Count the nulls instead:

```python
import batcher as bt
from batcher import col

photos = bt.from_pydict({"bytes": [b"", b""]})
undecodable = photos.filter(col("bytes").image.decode().is_null())
```

(tensor-columns)=
## Tensor columns

A column where every row is a same-shape `N`-dimensional tensor is stored as Arrow's canonical fixed-shape-tensor type. The shape travels with the data across the engine boundary, and the column converts to a correctly shaped training tensor. {py:func}`from_numpy <batcher.from_numpy>` and the NumPy reader build one for rows of rank 2 or higher:

```python
import batcher as bt
import numpy as np

imgs = np.zeros((4, 8, 8, 3), dtype=np.uint8)
ds = bt.from_numpy(imgs, column="image")
print(ds.collect().schema.field("image").type.shape)  # [8, 8, 3]
```

When you construct data yourself, the `tensor_type`, `to_tensor_column`, `as_tensor_column`, and `is_tensor_column` helpers in `batcher.io.formats.ml.tensor` build and classify these columns.

When a tensor column reaches a `map_batches` model stage with `batch_format="numpy"` or `"torch"`, the per-row tensors arrive **stacked** into one array with a leading batch dimension, such as `(batch, H, W, 3)`. That's the shape a vision model's forward pass wants, with no stacking or reshaping in the UDF.

### Images of different sizes

The canonical type carries one shape for a whole column, so a corpus at native resolution has no fixed-shape form. Return the arrays anyway. A `map_batches` whose column holds arrays of differing shape produces a variable-shape tensor column, and `to_numpy` and `batch_format="numpy"` decode it back to one array per row:

```python
import numpy as np

sizes = bt.from_pydict({"id": [1, 2, 3]})


def decode(batch):
    shapes = [(2, 2), (3, 4), (1, 5)]
    return {"img": [np.zeros(s, dtype=np.uint8) for s in shapes]}


out = sizes.map_batches(decode, output_columns=["img"])
print([a.shape for a in out.to_numpy()["img"]])
# [(2, 2), (3, 4), (1, 5)]
```

The column is an ordinary Arrow struct underneath, so it filters, joins, shuffles, and writes to Parquet like any other, and a distributed run returns exactly what a single-node run does. It can't become one torch tensor, because rows of different shape have no stacked form, so `batch_format="torch"` drops such a column and says so. Resize when a model needs a batch, and let the pipeline carry the originals until then.

A nested list column holds several small vectors per row, such as per-frame features or a ragged batch of patches. {py:meth}`.list.flatten() <batcher.plan.expr_ir.namespaces.collections._ListNamespace.flatten>` removes one level of nesting and keeps element order, for a downstream stage that wants one contiguous vector instead of a list of lists:

```python
import batcher as bt
from batcher import col

# Each row is a list of per-frame feature vectors; flatten to one vector per row.
frames = bt.from_pydict({"clip": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]})
flat = frames.select(vec=col("clip").list.flatten())
print(flat.to_pydict())
# {'vec': [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]]}
```

## See also

- {doc}`/ml/preparing/multimodal/curating`: screening decoded images and scraped text for rows worth keeping.
- {doc}`/ml/preparing/multimodal/augmenting`: geometry, color, and perceptual hashes on image columns.
- {doc}`/ml/preparing/multimodal/audio`: level, normalization, and spectral features for audio.
- {doc}`/ml/preparing/multimodal/video`: frame sampling, stills, and which video decoder is running.
- {doc}`/ml/preparing/multimodal/pipelines`: moving decoded media through a plan and into a model stage.
- {doc}`/architecture/deep-dives/memory/tensor-columns`: how tensor columns are represented in the engine.
