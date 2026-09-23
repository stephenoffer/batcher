# Media columns

This page covers the three accessors that read inside a binary media column: `.image` on an encoded image, `.audio` on an encoded waveform, and `.video` on a clip. They turn media work into expressions, so a decode, a resize, and a quality filter sit in the same plan as a `filter` and a `join`.

A media column is ordinary Arrow binary. Nothing about it is a separate execution path, which is the point: the optimizer sees a decode the way it sees a multiplication, and a predicate over an image's shape is a predicate the planner can push down.

## Header facts are free

Every image format writes its dimensions and color layout into a header before the pixel data. The `.image` accessor reads that header directly, so the questions below cost a few bytes per row and never decode a pixel.

```python
import base64

import batcher as bt

# Two tiny PNGs: one 8x2, one 2x8.
wide = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAACCAIAAADq9gq6AAAAEUlEQVR4nGO4o6GBFTHgkgAA4EESwW1PLREAAAAASUVORK5CYII="
)
tall = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAAICAIAAABcT7kVAAAAEElEQVR4nGPQCLgDRAykUgBIFhVB7m0LTAAAAABJRU5ErkJggg=="
)
photos = bt.from_pydict({"name": ["wide", "tall"], "img": [wide, tall]})

img = bt.col("img").image
print(
    photos.select(
        name=bt.col("name"), format=img.format(), aspect=img.aspect_ratio(), alpha=img.has_alpha()
    ).to_pydict()
)
# {'name': ['wide', 'tall'], 'format': ['png', 'png'], 'aspect': [4.0, 0.25], 'alpha': [False, False]}
```

That makes a shape filter cheap enough to run over a whole catalog before any of it is decoded:

```python
print(photos.filter(img.aspect_ratio() > 1).select(bt.col("name")).to_pydict())
# {'name': ['wide']}
```

Run the filter first and the decode never happens for the rows it drops. Run it after one and you have paid for every row, including the ones you were about to throw away. Ordering a media pipeline is mostly this one decision.

## Decoding into tensors

{py:obj}`to_tensor_f32(width, height) <batcher.plan.expr_ir.image._ImageNamespace.to_tensor_f32>` turns encoded bytes into a fixed-shape float tensor column, with optional per-channel `mean` and `std` so a model's normalization stays in the plan. It takes the size rather than inferring one, because a batch of full-resolution frames is the most common way to exhaust memory in a media pipeline, and a fixed shape is what lets the column stay a tensor rather than a ragged list.

```python
# docs: skip
import batcher as bt

ds = bt.read.images("s3://bucket/photos/")  # bytes plus header metadata, nothing decoded
ratio = bt.col("image").image.aspect_ratio()
ready = ds.filter((ratio >= 0.5) & (ratio <= 2.0)).with_columns(
    pixels=bt.col("image").image.to_tensor_f32(224, 224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
```

The decoded column is Arrow, so it reaches a model without a copy and without leaving the engine. {doc}`/ml/preparing/multimodal/decoding` covers the reader-side form of the same thing, where `decode=True` folds the decode into the scan.

The audio and video accessors work the same way. `.video` samples a frame as an encoded image, so that frame becomes an ordinary image column every `.image` method then applies to. `.audio` decodes to a waveform, and the rest of the namespace is the signal processing a model's front end would otherwise do in Python: {py:obj}`resample <batcher.plan.expr_ir.audio._AudioNamespace.resample>`, {py:obj}`trim_silence <batcher.plan.expr_ir.audio._AudioNamespace.trim_silence>`, {py:obj}`pad_or_trim <batcher.plan.expr_ir.audio._AudioNamespace.pad_or_trim>`, and {py:obj}`mel_spectrogram <batcher.plan.expr_ir.audio._AudioNamespace.mel_spectrogram>`.

```python
# docs: skip
clips = bt.read.video("s3://bucket/clips/")
frames = clips.with_columns(thumb=bt.col("video").video.frame_at(0.0, 112).image.to_tensor_f32(112, 112))

speech = bt.read.audio("s3://bucket/calls/")
features = speech.with_columns(
    mels=bt.col("audio").audio.resample(16000).audio.trim_silence().audio.mel_spectrogram(16000)
)
```

Passing `decode=True` to {py:obj}`bt.read.audio <batcher.api.io_namespace.reader.Reader.audio>` or {py:obj}`bt.read.video <batcher.api.io_namespace.reader.Reader.video>` folds the decode into the scan instead, which is the better form when every row needs it.

## Curating before you spend

The quality signals are the reason to reach for these accessors rather than a Python loop over files. Each one is an aggregate-friendly scalar, so a corpus can be profiled, filtered, and deduplicated with the relational verbs already in hand.

| Signal | Answers |
| --- | --- |
| {py:obj}`entropy() <batcher.plan.expr_ir.image._ImageNamespace.entropy>` | Is this image blank, or a solid placeholder? |
| {py:obj}`sharpness() <batcher.plan.expr_ir.image._ImageNamespace.sharpness>` | Is it out of focus? |
| {py:obj}`brightness() <batcher.plan.expr_ir.image._ImageNamespace.brightness>`, {py:obj}`colorfulness() <batcher.plan.expr_ir.image._ImageNamespace.colorfulness>` | Is it under-exposed, or greyscale in all but name? |
| {py:obj}`phash() <batcher.plan.expr_ir.image._ImageNamespace.phash>`, {py:obj}`dhash() <batcher.plan.expr_ir.image._ImageNamespace.dhash>`, {py:obj}`ahash() <batcher.plan.expr_ir.image._ImageNamespace.ahash>` | Is it a near-duplicate of one we already have? |

A perceptual hash is a value like any other, so near-duplicate removal is a `group_by` rather than a special operator:

```python
# docs: skip
deduped = ds.with_columns(h=bt.col("image").image.phash()).distinct(["h"])
```

{doc}`/ml/preparing/multimodal/curating` works this through on a real corpus, including what each threshold costs you in recall.

## Requirements and limitations

Decoding needs the media extras. `pip install "batcher-engine[multimodal]"` installs the image, audio, and video codecs together, and the `image`, `audio`, and `video` extras install them one at a time. A call whose codec is missing raises {py:obj}`MissingDependencyError <batcher.MissingDependencyError>`, which names the extra to install. {doc}`/getting-started/install/packages-and-extras` lists the full set.

The accessors report what a format records rather than inferring it. A field a particular container leaves out comes back null.

## See also

- {doc}`/api/accessors/media`: every `.image`, `.audio`, and `.video` method, with signatures.
- {doc}`/ml/preparing/multimodal/index`: fetching, decoding, curating, and augmenting media at scale.
- {doc}`/user-guide/moving-data/reading-data`: the readers that produce a media column.
- {doc}`/cookbook/ml/pipelines/multimodal/index`: captioning, classification, detection, and transcription end to end.
- {doc}`/architecture/deep-dives/memory/tensor-columns`: how a decoded tensor is laid out, and what it costs.
